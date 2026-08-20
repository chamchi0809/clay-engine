import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline, TgpuMutable, TgpuReadonly, TgpuUniform } from 'typegpu';
import { Brush, BrushSet, defaultBrushSet, type BrushDesc, type BrushValue } from './brush.ts';
import { MipRefiner } from './mips.ts';
import { TileGrid, unflattenAt } from './tilegrid.ts';
import { SdfVolume, TILE, volumeLevelLayout, volumeWriteLayout } from './volume.ts';

const VOXELS_PER_TILE = TILE * TILE * TILE;

export const EditParams = d.struct({
  editCount: d.u32,
});

export interface SdfEditorOptions {
  /** Edits applied in one flush. */
  maxEdits?: number;
  /** Edits recorded per tile. */
  tileCapacity?: number;
  /**
   * Tiles that can be edited in one flush. Defaults to every mip-0 tile, which cannot
   * overflow; lower it to trade edit reach for the staging buffer's 2 KB per tile.
   */
  maxTiles?: number;
  /**
   * Primitives the edits may use. Compiled into the apply pipeline, so it has to be the
   * same set the volume was built with - otherwise a brush kind that bakes into the world
   * would silently read as empty when it is carved with. Defaults to the builtins.
   */
  brushSet?: BrushSet;
}

/**
 * Applies brush edits to a live volume, in place, over only the tiles they touch.
 *
 * The awkward part is that WebGPU has no read-write storage access for `rgba16float`
 * (nor for any filterable format), so a tile cannot be read and written by the same
 * dispatch. Claybook hit the same wall on DX11.1 and solved it the same way: fold the
 * edit into a staging buffer, then copy whole tiles back (GDC'18 slide 18). The staging
 * value is `pack2x16float`-ed, which is exactly the precision the texture stores anyway.
 *
 * After mip 0 is patched, the dirty-tile mask walks down the mip chain - dilated by one
 * tile per level, because a coarser band reaches further - and {@link MipRefiner}
 * rebuilds each level. That keeps the cost proportional to the edited surface: a
 * pencil-sized carve touches a handful of tiles no matter how big the world is.
 */
export class SdfEditor {
  readonly volume: SdfVolume;
  readonly grid: TileGrid;
  readonly refiner: MipRefiner;
  readonly maxEdits: number;
  readonly maxTiles: number;
  readonly brushSet: BrushSet;
  readonly edits: TgpuReadonly<d.WgslArray<typeof Brush>>;
  readonly params: TgpuUniform<typeof EditParams>;
  /** Per-voxel `pack2x16float(distance, material)` staging area, indexed by tile slot. */
  readonly staging: TgpuMutable<d.WgslArray<d.U32>>;

  private readonly binPipeline: TgpuComputePipeline;
  private readonly applyPipeline: TgpuComputePipeline;
  private readonly copyPipeline: TgpuComputePipeline;
  private readonly binGroups: number;
  private pending: BrushValue[] = [];

  constructor(volume: SdfVolume, options: SdfEditorOptions = {}) {
    const root = volume.root;
    this.volume = volume;
    this.maxEdits = options.maxEdits ?? 64;
    this.brushSet = options.brushSet ?? defaultBrushSet;
    const applyBrush = this.brushSet.applyBrush;
    const tileRes0 = volume.tileResPerMip[0];
    this.maxTiles = options.maxTiles ?? tileRes0 * tileRes0 * tileRes0;
    this.grid = new TileGrid(volume, options.tileCapacity ?? 8);
    this.refiner = new MipRefiner(volume, this.grid);
    this.binGroups = this.grid.cellDispatchGroups(0);

    const edits = root.createReadonly(d.arrayOf(Brush, this.maxEdits)).$name('sdfEdits');
    const params = root.createUniform(EditParams, { editCount: 0 });
    const staging = root
      .createMutable(d.arrayOf(d.u32, this.maxTiles * VOXELS_PER_TILE))
      .$name('sdfEditStaging');
    this.edits = edits;
    this.params = params;
    this.staging = staging;

    const grid = this.grid;
    const origin = d.vec3f(...volume.origin);
    const voxel = volume.voxelSize;
    const tileSize = voxel * TILE;
    const tileHalf = tileSize / 2;
    const bandWorld = volume.band * voxel;
    const cells0 = tileRes0 * tileRes0 * tileRes0;
    const unflatten0 = unflattenAt(tileRes0);
    const cellIndex0 = grid.cellIndexAt(0);
    const tileAt0 = grid.tileAt(0);
    const maxTiles = this.maxTiles;

    // --- bin: which mip-0 tiles does each edit reach into? -------------------
    this.binPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= d.u32(cells0)) {
          return;
        }
        const tile = unflatten0(i);
        const tileCenter = origin + (d.vec3f(tile) * tileSize + d.vec3f(tileHalf));
        const cell = cellIndex0(tile);
        let n = d.u32(0);
        for (let b = d.u32(0); b < params.$.editCount; b++) {
          // Box-vs-sphere: distance from the brush origin to the tile box, widened by
          // the band, because a brush just outside a tile still changes its band values.
          const near = std.max(
            std.abs(edits.$[b].pos - tileCenter) - d.vec3f(tileHalf),
            d.vec3f(),
          );
          if (std.length(near) < edits.$[b].bound + bandWorld) {
            grid.writeCell(cell, n, b);
            n = n + 1;
          }
        }
        if (n > 0) {
          grid.setCellCount(cell, n);
        }
      }),
    });

    // --- apply: fold the edits into the staging buffer -----------------------
    this.applyPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [TILE, TILE, TILE / 2],
        in: {
          wid: d.builtin.workgroupId,
          lid: d.builtin.localInvocationId,
        },
      })(({ wid, lid }) => {
        'use gpu';
        const slot = wid.x;
        if (slot >= d.u32(maxTiles)) {
          return;
        }
        const tile = tileAt0(slot);
        const cell = cellIndex0(tile);
        const n = grid.cellCount(cell);
        for (const half of std.range(2)) {
          const local = d.vec3u(lid.x, lid.y, lid.z * 2 + d.u32(half));
          const coord = tile * d.u32(TILE) + local;
          const p = origin + (d.vec3f(coord) + d.vec3f(0.5)) * voxel;
          const raw = std.textureLoad(volumeLevelLayout.$.src, coord, 0);
          let acc = d.vec2f(raw.x * bandWorld, raw.y);
          for (let i = d.u32(0); i < n; i++) {
            acc = applyBrush(acc, edits.$[grid.cellId(cell, i)], p);
          }
          const flat = local.x + d.u32(TILE) * (local.y + d.u32(TILE) * local.z);
          staging.$[slot * d.u32(VOXELS_PER_TILE) + flat] = std.pack2x16float(
            d.vec2f(std.clamp(acc.x / bandWorld, -1, 1), acc.y),
          );
        }
      }),
    });

    // --- copy: staging -> texture, now that nothing reads the old values -----
    this.copyPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [TILE, TILE, TILE / 2],
        in: {
          wid: d.builtin.workgroupId,
          lid: d.builtin.localInvocationId,
        },
      })(({ wid, lid }) => {
        'use gpu';
        const slot = wid.x;
        if (slot >= d.u32(maxTiles)) {
          return;
        }
        const tile = tileAt0(slot);
        for (const half of std.range(2)) {
          const local = d.vec3u(lid.x, lid.y, lid.z * 2 + d.u32(half));
          const flat = local.x + d.u32(TILE) * (local.y + d.u32(TILE) * local.z);
          const v = std.unpack2x16float(
            staging.$[slot * d.u32(VOXELS_PER_TILE) + flat],
          );
          std.textureStore(
            volumeWriteLayout.$.out,
            tile * d.u32(TILE) + local,
            d.vec4f(v.x, v.y, 0, 1),
          );
        }
      }),
    });
  }

  /** Queues one edit for the next {@link flush}. */
  push(edit: BrushDesc | BrushValue): void {
    const brush = 'kind' in edit && typeof edit.kind === 'string'
      ? this.brushSet.make(edit as BrushDesc)
      : (edit as BrushValue);
    if (this.pending.length >= this.maxEdits) {
      // ponytail: dropping is better than growing the buffer mid-frame. Edits come
      // from input, so a frame that queues 64 of them has a bug upstream.
      console.warn(`SdfEditor: dropping edit, ${this.maxEdits} already queued this frame`);
      return;
    }
    this.pending.push(brush);
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  /** Applies and clears the queue. Returns false when there was nothing to do. */
  flush(pass: TgpuComputePass): boolean {
    if (this.pending.length === 0) {
      return false;
    }
    this.edits.write(this.pending);
    this.params.write({ editCount: this.pending.length });
    this.pending = [];

    this.grid.reset(pass);
    this.binPipeline.with(pass).dispatchWorkgroups(this.binGroups);
    this.grid.compact(pass, 0);
    let apply = this.applyPipeline.with(this.volume.levelGroups[0]);
    // Empty unless the set has a mesh atlas, in which case the fold samples it.
    for (const g of this.brushSet.groups) {
      apply = apply.with(g);
    }
    this.grid.dispatchTiles(pass, apply, 0);
    this.grid.dispatchTiles(pass, this.copyPipeline.with(this.volume.writeGroups[0]), 0);
    for (let m = 1; m < this.volume.mipLevels; m++) {
      this.grid.dilate(pass, m);
      this.grid.compact(pass, m);
      this.refiner.refresh(pass, m);
    }
    return true;
  }

  destroy(): void {
    this.grid.destroy();
  }
}
