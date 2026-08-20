import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline, TgpuReadonly, TgpuUniform } from 'typegpu';
import { Brush, applyBrush, type BrushValue } from './brush.ts';
import { SdfVolume, TILE, volumeWriteLayout } from './volume.ts';

/** Brushes considered per 8^3 tile. */
export const MAX_TILE_BRUSHES = 64;

export const BuildParams = d.struct({
  brushCount: d.u32,
});

export interface SdfBuilderOptions {
  /** Capacity of the brush buffer. */
  maxBrushes?: number;
}

/**
 * Bakes a brush list into an {@link SdfVolume}, one dispatch per mip level.
 *
 * Claybook (GDC'18 slides 11-16) culled brushes into a sparse tile grid and used an
 * indirect dispatch, because it re-baked the world while the game ran. Here a full
 * bake only happens on level load, so the levels are written densely - every tile gets
 * a workgroup - which removes the grid, the compaction and the clear pass. The sparse
 * machinery still exists in {@link TileGrid} and is what the per-frame edit path
 * (`modify.ts`) and the fluid bake use, which is where sparsity actually pays.
 *
 * The per-tile brush cull is kept: one thread walks the brush list in order and writes
 * the survivors to workgroup memory, so the 512 voxel threads only fold the handful of
 * brushes that reach into their tile. Cost per voxel tracks local brush density, not
 * total brush count - the property that mattered in the talk.
 *
 * Each mip is evaluated analytically at its own voxel size and band width. That is
 * exact (no downsample error to re-normalise) and only possible because brushes are
 * primitives rather than Claybook's baked brush volumes.
 */
export class SdfBuilder {
  readonly volume: SdfVolume;
  readonly maxBrushes: number;
  readonly brushes: TgpuReadonly<d.WgslArray<typeof Brush>>;
  readonly params: TgpuUniform<typeof BuildParams>;

  private readonly genPipelines: TgpuComputePipeline[] = [];
  private brushCount = 0;

  constructor(volume: SdfVolume, options: SdfBuilderOptions = {}) {
    const root = volume.root;
    this.volume = volume;
    this.maxBrushes = options.maxBrushes ?? 256;

    const brushes = root
      .createReadonly(d.arrayOf(Brush, this.maxBrushes))
      .$name('brushes');
    const params = root.createUniform(BuildParams, { brushCount: 0 });
    this.brushes = brushes;
    this.params = params;

    // ponytail: one thread builds the per-tile brush list so the CSG fold stays in
    // brush order (an atomic append would scramble add-then-cut layering). It costs
    // `brushCount` serial tests per tile at load time; if bake time ever matters,
    // replace with a workgroup prefix-sum compaction.
    const tileIds = tgpu.workgroupVar(d.arrayOf(d.u32, MAX_TILE_BRUSHES));
    const tileCount = tgpu.workgroupVar(d.u32);

    const [ox, oy, oz] = volume.origin;
    for (let m = 0; m < volume.mipLevels; m++) {
      const voxel = volume.voxelSizeAt(m);
      const tileSize = voxel * TILE;
      const bandWorld = volume.band * voxel;
      // Tiles are culled as boxes, widened by the stored band: a brush further away
      // than the band cannot change any value in this tile.
      const tileHalf = tileSize / 2;
      // Any accumulator value beyond the band clamps to "empty", so this is just an
      // identity element for the smooth-min fold.
      const emptyDist = bandWorld * 4;

      this.genPipelines.push(
        root.createComputePipeline({
          compute: tgpu.computeFn({
            workgroupSize: [TILE, TILE, TILE / 2],
            in: {
              wid: d.builtin.workgroupId,
              lid: d.builtin.localInvocationId,
            },
          })(({ wid, lid }) => {
            'use gpu';
            const tileMin = d.vec3f(
              d.f32(ox) + d.f32(wid.x) * tileSize,
              d.f32(oy) + d.f32(wid.y) * tileSize,
              d.f32(oz) + d.f32(wid.z) * tileSize,
            );
            const tileCenter = tileMin + d.vec3f(tileHalf);

            if (lid.x + lid.y + lid.z === 0) {
              let n = d.u32(0);
              for (let b = d.u32(0); b < params.$.brushCount; b++) {
                const near = std.max(
                  std.abs(brushes.$[b].pos - tileCenter) - d.vec3f(tileHalf),
                  d.vec3f(),
                );
                if (
                  std.length(near) < brushes.$[b].bound + bandWorld &&
                  n < d.u32(MAX_TILE_BRUSHES)
                ) {
                  tileIds.$[n] = b;
                  n = n + 1;
                }
              }
              tileCount.$ = n;
            }
            std.workgroupBarrier();

            const n = tileCount.$;
            // 8^3 voxels per tile but only 256 invocations allowed per workgroup, so
            // each thread covers two z-slices.
            for (const half of std.range(2)) {
              const local = d.vec3u(lid.x, lid.y, lid.z * 2 + d.u32(half));
              const coord = wid * d.u32(TILE) + local;
              const p = tileMin + (d.vec3f(local) + d.vec3f(0.5)) * voxel;
              let acc = d.vec2f(emptyDist, 0);
              for (let i = d.u32(0); i < n; i++) {
                acc = applyBrush(acc, brushes.$[tileIds.$[i]], p);
              }
              std.textureStore(
                volumeWriteLayout.$.out,
                coord,
                d.vec4f(std.clamp(acc.x / bandWorld, -1, 1), acc.y, 0, 1),
              );
            }
          }),
        }),
      );
    }
  }

  /** Uploads the brush list. Extra brushes past `maxBrushes` are dropped. */
  setBrushes(list: readonly BrushValue[]): void {
    if (list.length > this.maxBrushes) {
      console.warn(
        `SdfBuilder: ${list.length} brushes exceeds capacity ${this.maxBrushes}, extras dropped`,
      );
    }
    const used = list.slice(0, this.maxBrushes);
    this.brushCount = used.length;
    if (used.length > 0) {
      this.brushes.write(used);
    }
    this.params.write({ brushCount: this.brushCount });
  }

  /** Re-bakes every mip level of the volume from the current brush list. */
  rebuild(pass: TgpuComputePass): void {
    for (let m = 0; m < this.volume.mipLevels; m++) {
      const tiles = this.volume.tileResPerMip[m];
      this.genPipelines[m]
        .with(this.volume.writeGroups[m])
        .with(pass)
        .dispatchWorkgroups(tiles, tiles, tiles);
    }
  }
}
