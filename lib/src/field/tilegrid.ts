import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline, TgpuMutable, TgpuRoot } from 'typegpu';
import type { SdfVolume } from './volume.ts';

/** Packs a tile coordinate into one u32. Tile resolution is <= 1024 per axis. */
export const packTile = tgpu.fn([d.vec3u], d.u32)((c) => {
  'use gpu';
  return c.x | (c.y << 10) | (c.z << 20);
});

export const unpackTile = tgpu.fn([d.u32], d.vec3u)((v) => {
  'use gpu';
  return d.vec3u(v & 1023, (v >>> 10) & 1023, (v >>> 20) & 1023);
});

/** `flatIndex -> 3D coordinate` inside a cubic grid of edge `res`. */
export function unflattenAt(res: number): (i: number) => d.v3u {
  return (i: number) => {
    'use gpu';
    const plane = d.u32(res * res);
    const rem = i % plane;
    return d.vec3u(rem % d.u32(res), rem / d.u32(res), i / plane);
  };
}

/**
 * The sparse-tile scheduler shared by every field-writing pass.
 *
 * This is the scheduling half of Claybook's GDC'18 slides 11-16, with the brush
 * specifics factored out:
 *   1. producers bucket their sources (brushes / edits / fluid particles) into an
 *      8^3-voxel tile grid using {@link TileGrid.append},
 *   2. {@link TileGrid.compact} turns the non-empty tiles into a dense list plus
 *      indirect dispatch args,
 *   3. consumers run one workgroup per live tile via {@link TileGrid.dispatchTiles},
 *   4. {@link TileGrid.dilate} pushes the live-tile mask down the mip chain (2x2x2
 *      downsample of the "not empty" mask, dilated by one tile).
 *
 * Typically only a few percent of tiles are live, which is the whole point of the
 * indirect dispatch: cost tracks the edited surface, not the volume.
 *
 * Every mip's tile records live in one shared buffer at {@link SdfVolume.cellOffsets},
 * so a full chain needs a single reset.
 */
export class TileGrid {
  readonly root: TgpuRoot;
  readonly volume: SdfVolume;
  /** Max source ids recorded per tile. */
  readonly capacity: number;

  readonly cellCounts: TgpuMutable<d.WgslArray<d.Atomic<d.U32>>>;
  readonly cellIds: TgpuMutable<d.WgslArray<d.U32>>;
  readonly tileList: TgpuMutable<d.WgslArray<d.U32>>;
  readonly dispatchArgs: TgpuMutable<d.WgslArray<d.Atomic<d.U32>>>;
  /** Raw handle for `dispatchWorkgroupsIndirect`. */
  readonly argsBuffer: GPUBuffer;

  /** `(cellIndex, sourceId) -> void`. Overflowing ids are dropped. */
  readonly append: (cell: number, id: number) => void;
  /** How many source ids a tile stored, clamped to {@link capacity}. */
  readonly cellCount: (cell: number) => number;
  /** Reads one recorded source id. */
  readonly cellId: (cell: number, slot: number) => number;
  /** Writes one id at a fixed slot. For producers that bin a cell from a single thread. */
  readonly writeCell: (cell: number, slot: number, id: number) => void;
  /** Publishes a cell's id count. Pairs with {@link writeCell}. */
  readonly setCellCount: (cell: number, n: number) => void;

  private readonly resetPipeline: TgpuComputePipeline;
  private readonly resetGroups: number;
  private readonly compactPipelines: TgpuComputePipeline[] = [];
  private readonly dilatePipelines: (TgpuComputePipeline | undefined)[] = [];
  private readonly cellGroups: number[] = [];

  constructor(volume: SdfVolume, capacity = 32) {
    const root = volume.root;
    this.root = root;
    this.volume = volume;
    this.capacity = capacity;

    const { totalCells, mipLevels, tileResPerMip, cellOffsets } = volume;

    const cellCounts = root
      .createMutable(d.arrayOf(d.atomic(d.u32), totalCells))
      .$name('tileCellCounts');
    const cellIds = root
      .createMutable(d.arrayOf(d.u32, totalCells * capacity))
      .$name('tileCellIds');
    const tileList = root.createMutable(d.arrayOf(d.u32, totalCells)).$name('tileList');
    // INDIRECT is not expressible through createMutable, so the buffer is made by
    // hand and wrapped. Layout is (x, y, z, unused) per mip level.
    this.argsBuffer = root.device.createBuffer({
      size: mipLevels * 16,
      usage: GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      label: 'tileDispatchArgs',
    });
    const dispatchArgs = root.createMutable(
      d.arrayOf(d.atomic(d.u32), mipLevels * 4),
      this.argsBuffer,
    );
    this.cellCounts = cellCounts;
    this.cellIds = cellIds;
    this.tileList = tileList;
    this.dispatchArgs = dispatchArgs;

    this.append = (cell: number, id: number) => {
      'use gpu';
      const slot = std.atomicAdd(cellCounts.$[cell], 1);
      // ponytail: fixed per-tile capacity instead of Claybook's exact global list.
      // Overflow silently drops a source; raise `capacity` if a scene ever puts more
      // than 32 overlapping brushes inside one 8^3 tile.
      if (slot < capacity) {
        cellIds.$[cell * capacity + slot] = id;
      }
    };

    this.cellCount = (cell: number) => {
      'use gpu';
      return std.min(std.atomicLoad(cellCounts.$[cell]), d.u32(capacity));
    };

    this.cellId = (cell: number, slot: number) => {
      'use gpu';
      return cellIds.$[cell * capacity + slot];
    };

    // Producers that own a whole cell from one thread use these instead of `append`:
    // the ids stay in source order, which matters as soon as the fold is a CSG stack
    // (an `add` after a `cut` is not the same shape as a `cut` after an `add`).
    this.writeCell = (cell: number, slot: number, id: number) => {
      'use gpu';
      if (slot < capacity) {
        cellIds.$[cell * capacity + slot] = id;
      }
    };

    this.setCellCount = (cell: number, n: number) => {
      'use gpu';
      std.atomicStore(cellCounts.$[cell], std.min(n, d.u32(capacity)));
    };

    // --- reset: zero the counts and re-arm the indirect args -----------------
    const resetLanes = Math.max(totalCells, mipLevels * 4);
    this.resetGroups = Math.ceil(resetLanes / 64);
    this.resetPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i < d.u32(totalCells)) {
          std.atomicStore(cellCounts.$[i], 0);
        }
        if (i < d.u32(mipLevels * 4)) {
          const lane = i & 3;
          std.atomicStore(
            dispatchArgs.$[i],
            std.select(d.u32(0), d.u32(1), lane === 1 || lane === 2),
          );
        }
      }),
    });

    // --- per-mip compaction and mask dilation -------------------------------
    // The mip index is a JS constant, so each level gets its own tiny specialised
    // pipeline with grid strides folded into the WGSL. Cheaper than threading a
    // uniform through, and there are only a handful of levels.
    for (let m = 0; m < mipLevels; m++) {
      const tr = tileResPerMip[m];
      const off = cellOffsets[m];
      const cells = tr * tr * tr;
      const unflatten = unflattenAt(tr);
      this.cellGroups.push(Math.ceil(cells / 64));

      this.compactPipelines.push(
        root.createComputePipeline({
          compute: tgpu.computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
          })(({ gid }) => {
            'use gpu';
            const i = gid.x;
            if (i >= d.u32(cells)) {
              return;
            }
            if (std.atomicLoad(cellCounts.$[d.u32(off) + i]) === 0) {
              return;
            }
            // One workgroup per live tile. This mip's list region is exactly `cells`
            // entries long, so the append can never run past it.
            const slot = std.atomicAdd(dispatchArgs.$[m * 4], 1);
            tileList.$[d.u32(off) + slot] = packTile(unflatten(i));
          }),
        }),
      );

      if (m === 0) {
        this.dilatePipelines.push(undefined);
        continue;
      }
      const prevTr = tileResPerMip[m - 1];
      const prevOff = cellOffsets[m - 1];
      this.dilatePipelines.push(
        root.createComputePipeline({
          compute: tgpu.computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
          })(({ gid }) => {
            'use gpu';
            const i = gid.x;
            if (i >= d.u32(cells)) {
              return;
            }
            const base = d.vec3i(unflatten(i)) * 2;
            // The 2x2x2 children dilated by one tile, i.e. the range [2c-1, 2c+2]:
            // a parent tile must be rebuilt whenever anything in its neighbourhood
            // moved, because its band reaches one tile further out.
            let live = false;
            for (const dz of std.range(-1, 3)) {
              for (const dy of std.range(-1, 3)) {
                for (const dx of std.range(-1, 3)) {
                  const q = base + d.vec3i(dx, dy, dz);
                  if (std.all(std.ge(q, d.vec3i(0))) && std.all(std.lt(q, d.vec3i(d.i32(prevTr))))) {
                    const u = d.vec3u(q);
                    const idx = d.u32(prevOff) + u.x + d.u32(prevTr) * (u.y + d.u32(prevTr) * u.z);
                    if (std.atomicLoad(cellCounts.$[idx]) !== 0) {
                      live = true;
                    }
                  }
                }
              }
            }
            if (live) {
              std.atomicStore(cellCounts.$[d.u32(off) + i], 1);
            }
          }),
        }),
      );
    }
  }

  /**
   * Shader-side accessor factory. `mip` is a JS constant so the strides fold into the
   * WGSL. Returns `(tile) -> flat cell index` for the shared grid buffers.
   */
  cellIndexAt(mip: number): (tile: d.v3u) => number {
    const tr = this.volume.tileResPerMip[mip];
    const off = this.volume.cellOffsets[mip];
    return (tile: d.v3u) => {
      'use gpu';
      return d.u32(off) + tile.x + d.u32(tr) * (tile.y + d.u32(tr) * tile.z);
    };
  }

  /** Shader-side accessor factory: `(listIndex) -> tile coordinate` for one mip. */
  tileAt(mip: number): (listIndex: number) => d.v3u {
    const off = this.volume.cellOffsets[mip];
    const tileList = this.tileList;
    return (listIndex: number) => {
      'use gpu';
      return unpackTile(tileList.$[d.u32(off) + listIndex]);
    };
  }

  /** Zeroes tile counts and re-arms the indirect args for every mip level. */
  reset(pass: TgpuComputePass): void {
    this.resetPipeline.with(pass).dispatchWorkgroups(this.resetGroups);
  }

  /** Turns "count != 0" tiles at `mip` into a dense list + indirect workgroup count. */
  compact(pass: TgpuComputePass, mip: number): void {
    this.compactPipelines[mip].with(pass).dispatchWorkgroups(this.cellGroups[mip]);
  }

  /** Marks tiles at `mip` whose mip-1 children (dilated by one tile) are live. */
  dilate(pass: TgpuComputePass, mip: number): void {
    const p = this.dilatePipelines[mip];
    if (!p) {
      throw new Error('TileGrid.dilate: mip 0 has no parent level');
    }
    p.with(pass).dispatchWorkgroups(this.cellGroups[mip]);
  }

  /** Workgroups of 64 needed to cover every cell at `mip`. */
  cellDispatchGroups(mip: number): number {
    return this.cellGroups[mip];
  }

  /**
   * Runs `pipeline` once per live tile at `mip`. The pipeline must already carry any
   * bind groups it needs; `workgroupId.x` indexes the tile list via {@link tileAt}.
   */
  dispatchTiles(pass: TgpuComputePass, pipeline: TgpuComputePipeline, mip: number): void {
    pipeline.with(pass).dispatchWorkgroupsIndirect(this.argsBuffer, mip * 16);
  }

  /** Live tile count per mip. Debug/tests only - blocks on a GPU readback. */
  async readTileCounts(): Promise<number[]> {
    const device = this.root.device;
    const staging = device.createBuffer({
      size: this.argsBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.argsBuffer, 0, staging, 0, this.argsBuffer.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const view = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return Array.from({ length: this.volume.mipLevels }, (_, m) => view[m * 4]);
  }

  destroy(): void {
    this.argsBuffer.destroy();
  }
}
