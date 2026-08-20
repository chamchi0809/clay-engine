import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline } from 'typegpu';
import type { TileGrid } from './tilegrid.ts';
import { SdfVolume, TILE, volumeLevelLayout, volumeWriteLayout } from './volume.ts';

/** Tile edge plus the border the eikonal sweeps need on each side (Claybook: 12^3). */
const GSM_EDGE = TILE + 4;
const GSM_CELLS = GSM_EDGE * GSM_EDGE * GSM_EDGE;
const BORDER = 2;
const THREADS = TILE * TILE * (TILE / 2);

/**
 * Rebuilds coarse mip levels of a volume from the level below, for the tiles a
 * {@link TileGrid} marked live. This is Claybook GDC'18 slides 19-21.
 *
 * Two things have to happen per level, and the second is the non-obvious one:
 *
 *  1. **Downsample.** A 2x2x2 box filter, which a linear sampler gives for free in a
 *     single fetch when read exactly at the corner shared by the eight children.
 *  2. **Re-expand the band.** Values are stored normalised to the level's own band,
 *     and the band doubles in world size per level, so a downsampled `+-1` becomes
 *     `+-0.5` - half the level's band is suddenly unused, and everything that was
 *     saturated below now reads as "closer than it is". The fix is an eikonal
 *     (`|grad d| = 1`) relaxation that pushes distance back outwards until the band is
 *     full again: `band / voxel = 4` normalised units, of which the downsample keeps
 *     0.5, so two sweeps of 0.25 restore exactly 1.0.
 *
 * The 12^3 neighbourhood lives in workgroup memory and is double-buffered, because an
 * in-place relaxation would race. Two buffers of 12^3 f32 is 13.5 KB, which is why the
 * border is 2 and not 3 - the 16 KB workgroup limit is the binding constraint, and two
 * sweeps is exactly what a 2-voxel border supports.
 *
 * Unlike {@link SdfBuilder}, this path never looks at what produced the field, so it
 * works for brush edits, baked particle clouds and fluid alike.
 */
export class MipRefiner {
  readonly volume: SdfVolume;
  readonly grid: TileGrid;
  private readonly pipelines: (TgpuComputePipeline | undefined)[] = [];

  constructor(volume: SdfVolume, grid: TileGrid, options: { safety?: number } = {}) {
    const root = volume.root;
    this.volume = volume;
    this.grid = grid;
    // ponytail: chamfer distances on a 26-neighbour stencil slightly over-estimate
    // true Euclidean distance, and an over-estimate makes the tracer overshoot. The
    // safety factor trades a little band range for conservatism; drop it below 1 only
    // if you see surfaces being stepped through.
    const safety = options.safety ?? 0.98;

    // One array holding both ping-pong halves rather than two workgroupVars in a JS
    // array: TypeGPU cannot index a host-side array of references from GPU code, and
    // an offset is cheaper to read than a swap.
    const gsm = tgpu.workgroupVar(d.arrayOf(d.f32, GSM_CELLS * 2));

    const flatGsm = (c: d.v3i) => {
      'use gpu';
      return d.u32(c.x + GSM_EDGE * (c.y + GSM_EDGE * c.z));
    };
    const unflatGsm = (i: number) => {
      'use gpu';
      const plane = d.u32(GSM_EDGE * GSM_EDGE);
      const rem = i % plane;
      return d.vec3i(
        d.i32(rem % d.u32(GSM_EDGE)),
        d.i32(rem / d.u32(GSM_EDGE)),
        d.i32(i / plane),
      );
    };

    this.pipelines.push(undefined);
    for (let m = 1; m < volume.mipLevels; m++) {
      const mipRes = volume.resolution >> m;
      // One normalised unit is the level's band; one voxel is `1 / band` of it.
      const step = (1 / volume.band) * safety;
      const tileOf = grid.tileAt(m);

      this.pipelines.push(
        root.createComputePipeline({
          compute: tgpu.computeFn({
            workgroupSize: [TILE, TILE, TILE / 2],
            in: {
              wid: d.builtin.workgroupId,
              lid: d.builtin.localInvocationId,
              li: d.builtin.localInvocationIndex,
            },
          })(({ wid, lid, li }) => {
            'use gpu';
            const tileMin = d.vec3i(tileOf(wid.x)) * TILE;

            // --- gather: 2x2x2 downsample of the level below -------------------
            for (let i = li; i < d.u32(GSM_CELLS); i = i + d.u32(THREADS)) {
              const g = tileMin + unflatGsm(i) - d.vec3i(BORDER);
              // Clamp rather than treat the outside as empty: a level whose content
              // reaches the volume wall (a ground plane, say) must not get a ring of
              // "air" pulled in from beyond it.
              const c = std.clamp(g, d.vec3i(0), d.vec3i(d.i32(mipRes - 1)));
              // Reading at the corner shared by the eight children of this voxel makes
              // the linear filter return their average.
              const uv = (d.vec3f(c) + d.vec3f(0.5)) * (1 / mipRes);
              const v = std.textureSampleLevel(
                volumeLevelLayout.$.src,
                volumeLevelLayout.$.srcSamp,
                uv,
                0,
              ).x;
              // The band doubles in world size, so the same distance is half as far
              // through this level's band.
              gsm.$[i] = v * 0.5;
            }

            // --- eikonal re-expansion -----------------------------------------
            // Sweep k may only touch cells that still have a full neighbourhood, so
            // the writable region shrinks by one shell per sweep and the tile core is
            // exactly covered by the last one.
            for (let k = 0; k < BORDER; k++) {
              const src = d.u32((k % 2) * GSM_CELLS);
              const dst = d.u32(((k + 1) % 2) * GSM_CELLS);
              std.workgroupBarrier();
              for (let i = li; i < d.u32(GSM_CELLS); i = i + d.u32(THREADS)) {
                const c = unflatGsm(i);
                let v = gsm.$[src + i];
                const margin = d.i32(k + 1);
                if (
                  std.all(std.ge(c, d.vec3i(margin))) &&
                  std.all(std.lt(c, d.vec3i(d.i32(GSM_EDGE) - margin)))
                ) {
                  let lo = d.f32(-1e9);
                  let hi = d.f32(1e9);
                  // 26-neighbour chamfer: axis, edge and corner steps cost 1, sqrt2
                  // and sqrt3 voxels. Axis-only propagation would over-estimate
                  // diagonal distances by up to 73%.
                  for (const dz of std.range(-1, 2)) {
                    for (const dy of std.range(-1, 2)) {
                      for (const dx of std.range(-1, 2)) {
                        if (dx !== 0 || dy !== 0 || dz !== 0) {
                          // `std.length`, not a host-side `Math.hypot`: loop indices
                          // in GPU scope are GPU values, so the weight has to be
                          // computed on the GPU too.
                          const w = std.length(d.vec3f(d.f32(dx), d.f32(dy), d.f32(dz))) * step;
                          const nv = gsm.$[src + flatGsm(c + d.vec3i(dx, dy, dz))];
                          hi = std.min(hi, nv + w);
                          lo = std.max(lo, nv - w);
                        }
                      }
                    }
                  }
                  v = std.clamp(v, lo, hi);
                }
                gsm.$[dst + i] = v;
              }
            }
            std.workgroupBarrier();
            const out = d.u32((BORDER % 2) * GSM_CELLS);

            // --- store the tile core ------------------------------------------
            for (const half of std.range(2)) {
              const local = d.vec3u(lid.x, lid.y, lid.z * 2 + d.u32(half));
              const coord = d.vec3u(tileMin) + local;
              const gi = flatGsm(d.vec3i(local) + d.vec3i(BORDER));
              // Materials do not obey the eikonal equation; they just come along with
              // the same box filter.
              const mat = std.textureSampleLevel(
                volumeLevelLayout.$.src,
                volumeLevelLayout.$.srcSamp,
                (d.vec3f(coord) + d.vec3f(0.5)) * (1 / mipRes),
                0,
              ).y;
              std.textureStore(
                volumeWriteLayout.$.out,
                coord,
                d.vec4f(std.clamp(gsm.$[out + gi], -1, 1), mat, 0, 1),
              );
            }
          }),
        }),
      );
    }
  }

  /** Rebuilds `mip` from `mip - 1` over the grid's live tiles at `mip`. */
  refresh(pass: TgpuComputePass, mip: number): void {
    const p = this.pipelines[mip];
    if (!p) {
      throw new Error('MipRefiner.refresh: mip 0 has no parent level');
    }
    this.grid.dispatchTiles(
      pass,
      p
        .with(this.volume.levelGroups[mip - 1])
        .with(this.volume.writeGroups[mip]),
      mip,
    );
  }
}
