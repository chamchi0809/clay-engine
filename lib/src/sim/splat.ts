import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline, TgpuRoot } from 'typegpu';
import { SdfVolume, TILE, volumeWriteLayout } from '../field/volume.ts';
import { volumeField, type TracedField } from '../trace/field.ts';
import type { ParticleSet } from './particles.ts';

/**
 * A particle cloud, as seen by {@link SplatField}. Two shader closures and a bound.
 *
 * Deliberately not "a buffer of struct X": the fluid's particles and a shape-matched
 * body's particles are different structs living in different buffers, and neither needs
 * to know that the other exists in order to be turned into a surface.
 */
export interface ParticleCloud {
  /** Particles the bake must cover. A JS constant: it becomes the dispatch size. */
  readonly capacity: number;
  /** GPU scope: world position of particle `i`. */
  readonly positionAt: (i: number) => d.v3f;
  /** GPU scope: whether particle `i` contributes to the surface this frame. */
  readonly liveAt: (i: number) => boolean;
}

/**
 * One shape-matched body of a {@link ParticleSet}, as a {@link ParticleCloud}.
 *
 * The live range comes from the body record rather than from the whole buffer: the
 * extractor writes `first`/`count` on the GPU and re-extraction shrinks them, so
 * splatting the buffer would keep painting last extraction's leftovers.
 */
export function bodyCloud(set: ParticleSet, body: number): ParticleCloud {
  const particles = set.particles;
  const bodies = set.bodies;
  return {
    capacity: set.capacity,
    positionAt: (i: number) => {
      'use gpu';
      return d.vec3f(particles.$[i].pos);
    },
    liveAt: (i: number) => {
      'use gpu';
      const b = bodies.$[body];
      return i >= b.first && i < b.first + b.count;
    },
  };
}

export interface SplatFieldOptions {
  /** Radius of the sphere each particle contributes. Sets the surface thickness. */
  radius: number;
  /** Material id written into the volume. */
  material?: number;
  /** Volume resolution. Must suit `TILE * 2^(mipLevels-1)`. */
  resolution?: number;
  mipLevels?: number;
  /**
   * Band half-width in voxels. It sets the splat footprint cubically, so it is the
   * expensive knob - but it is also a hard correctness constraint for anything that uses
   * the result as a *collider* rather than as something to look at.
   *
   * Outside the band a volume reads as saturated, which is `band * voxel` exactly. A
   * collider testing `distance < myRadius` therefore sees a contact everywhere inside
   * the box whenever `band * voxel <= myRadius`, and resolves it along a gradient that
   * is flat there - i.e. it shoves things in an arbitrary direction for no reason. So:
   * `band * voxel` must exceed the largest radius any collider will test with.
   */
  band?: number;
  /** World-space region covered. A particle outside it simply does not appear. */
  origin?: readonly [number, number, number];
  worldSize?: number;
  label?: string;
}

/** `enc(v) = (clamp(v, -1, 1) + 1) * 32767.5`, so `atomicMin` on the u32 is `min` on `v`. */
const ENC_SCALE = 32767.5;
const ENC_SAT = 65535;

/**
 * Turns any {@link ParticleCloud} into a mip-mapped {@link TracedField}: a union of
 * spheres, scattered into a fixed-point grid with `atomicMin` and resolved into an
 * {@link SdfVolume}. Claybook GDC'18 slides 57-59.
 *
 * Scatter, not gather, and one splat per mip level:
 *
 *  - Scatter is `O(particles x footprint)`. A gather would be `O(voxels x local
 *    density)` and falls over exactly where a particle cloud is interesting.
 *  - Every level is splatted at its own voxel size and band width, so there is no
 *    downsample/re-normalise step and each level is a valid band-limited SDF on its own.
 *  - `atomicMin` on the fixed-point encoding *is* the distance union, which is why the
 *    encoding has to be monotone in the distance.
 *
 * This is the engine's answer to "my simulated thing should exist to the renderer":
 * `unionField` the result with the world and it is traced, shadowed, AO'd and collided
 * against like anything else - a fluid, a clay body, a debris cloud, all the same code.
 *
 * ponytail: a hard union of spheres, so a thin sheet shows the individual blobs.
 * `atomicMin` cannot express a smooth min; the upgrade is to scatter a fixed-point
 * *density* with `atomicAdd` and take an iso-surface, at the cost of no longer being a
 * Lipschitz-1 distance the tracer can trust directly.
 */
export class SplatField {
  readonly volume: SdfVolume;
  readonly field: TracedField;

  private readonly clear: TgpuComputePipeline;
  private readonly splat: TgpuComputePipeline[] = [];
  private readonly resolve: TgpuComputePipeline[] = [];
  private readonly cloudGroups: number;
  private readonly clearGroups: number;
  private readonly resolveGroups: number[] = [];

  constructor(root: TgpuRoot, cloud: ParticleCloud, options: SplatFieldOptions) {
    const res = options.resolution ?? 64;
    const mips = options.mipLevels ?? Math.max(1, Math.log2(res / TILE) | 0);
    const band = options.band ?? 2;
    const material = options.material ?? 0;
    const radius = options.radius;
    const label = options.label ?? 'splat';

    this.volume = new SdfVolume(root, {
      resolution: res,
      mipLevels: mips,
      band,
      worldSize: options.worldSize ?? 24,
      origin: options.origin ?? [-12, -4, -12],
      label: `${label}Volume`,
    });
    this.field = volumeField(this.volume);
    this.cloudGroups = Math.ceil(cloud.capacity / 64);

    const levelRes: number[] = [];
    const levelOffsets: number[] = [];
    let total = 0;
    for (let m = 0; m < mips; m++) {
      const r = res >> m;
      levelRes.push(r);
      levelOffsets.push(total);
      total += r * r * r;
    }
    this.clearGroups = Math.ceil(total / 64);
    const grid = root
      .createMutable(d.arrayOf(d.atomic(d.u32), total))
      .$name(`${label}Grid`);

    this.clear = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x < d.u32(total)) {
          std.atomicStore(grid.$[gid.x], d.u32(ENC_SAT));
        }
      }),
    });

    // The origin comes from the volume's uniform, not from JS, so `setRegion` can move
    // the box between frames without rebuilding a single pipeline.
    const vparams = this.volume.params;
    for (let m = 0; m < mips; m++) {
      const r = levelRes[m]!;
      const off = levelOffsets[m]!;
      const voxel = this.volume.voxelSizeAt(m);
      const bandWorld = band * voxel;
      const cells = r * r * r;
      this.resolveGroups.push(Math.ceil(cells / 64));

      this.splat.push(
        root.createComputePipeline({
          compute: tgpu.computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
          })(({ gid }) => {
            'use gpu';
            const i = gid.x;
            if (i >= d.u32(cloud.capacity) || !cloud.liveAt(i)) {
              return;
            }
            const p = cloud.positionAt(i) - vparams.$.origin;
            // Only voxels whose value this particle could lower: anything further than
            // `radius + band` already clamps to saturation.
            const reach = radius + bandWorld;
            const lo = d.vec3i(std.floor((p - d.vec3f(reach)) * (1 / voxel)));
            const hi = d.vec3i(std.floor((p + d.vec3f(reach)) * (1 / voxel)));
            const last = d.i32(r - 1);
            for (let z = std.max(lo.z, 0); z <= std.min(hi.z, last); z++) {
              for (let y = std.max(lo.y, 0); y <= std.min(hi.y, last); y++) {
                for (let x = std.max(lo.x, 0); x <= std.min(hi.x, last); x++) {
                  const c = (d.vec3f(d.f32(x), d.f32(y), d.f32(z)) + d.vec3f(0.5)) * voxel;
                  const sd = std.length(c - p) - radius;
                  if (sd < bandWorld) {
                    const e = d.u32((std.clamp(sd / bandWorld, -1, 1) + 1) * ENC_SCALE);
                    const idx = d.u32(off) + d.u32(x) + d.u32(r) * (d.u32(y) + d.u32(r) * d.u32(z));
                    std.atomicMin(grid.$[idx], e);
                  }
                }
              }
            }
          }),
        }),
      );

      this.resolve.push(
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
            const plane = d.u32(r * r);
            const rem = i % plane;
            const coord = d.vec3u(rem % d.u32(r), rem / d.u32(r), i / plane);
            const v = d.f32(std.atomicLoad(grid.$[d.u32(off) + i])) * (1 / ENC_SCALE) - 1;
            std.textureStore(volumeWriteLayout.$.out, coord, d.vec4f(v, d.f32(material), 0, 1));
          }),
        }),
      );
    }
  }

  /**
   * Centres the covered box on `center`. Cheap - one uniform write - and the next
   * {@link bake} fills the new region, so a body-sized volume can follow the body.
   */
  setCenter(center: readonly [number, number, number]): void {
    const half = this.volume.worldSize / 2;
    this.volume.setOrigin([center[0] - half, center[1] - half, center[2] - half]);
  }

  /**
   * Rebuilds the volume from the cloud's current positions, into a caller-owned pass.
   *
   * The clear is unconditional, including on a frame with nothing alive: an unwritten
   * `rgba16float` volume reads as zero, and zero means "surface right here" to the
   * tracer, so an un-cleared volume unioned with the world hides the world completely.
   */
  bake(pass: TgpuComputePass): void {
    this.clear.with(pass).dispatchWorkgroups(this.clearGroups);
    for (let m = 0; m < this.splat.length; m++) {
      this.splat[m]!.with(pass).dispatchWorkgroups(this.cloudGroups);
      this.resolve[m]!.with(pass)
        .with(this.volume.writeGroups[m]!)
        .dispatchWorkgroups(this.resolveGroups[m]!);
    }
  }

  destroy(): void {
    this.volume.destroy();
  }
}
