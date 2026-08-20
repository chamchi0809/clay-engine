import { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import * as sdf from '@typegpu/sdf';
import type { SdfVolume } from '../field/volume.ts';

/** Bind groups a field's shader closures need at draw/dispatch time. */
export type FieldGroups = readonly ReturnType<TgpuRoot['createBindGroup']>[];

/**
 * Everything the tracer needs from "a signed distance field". Deliberately narrower
 * than {@link SdfVolume}: the marcher never touches a texture, a bind group or a tile
 * grid, so the exact same tracing code runs on a baked volume, on a purely analytic
 * field, or on a blend of the two.
 *
 * `sample` returns both representations because the hierarchical loop needs both:
 * `.x` is a world-space conservative distance (how far it may step), `.y` is the
 * band-normalised value in `[-1, 1]` (whether the level is saturated, which is what
 * decides a mip hop - Claybook GDC'18 slide 24).
 */
export interface TracedField {
  /** Coarsest mip index. A JS constant, so the tracer can bake loop bounds. */
  readonly maxMip: number;
  /**
   * Bind groups every pipeline using this field must be `.with()`-ed. Empty for a
   * purely analytic field, one entry for a volume - the caller does not need to know
   * which.
   */
  readonly groups: FieldGroups;
  /** `(worldDistance, normalisedBandValue)` at `mip`. */
  readonly sample: (p: d.v3f, mip: number) => d.v2f;
  /** `(worldDistance, materialId)` at `mip`. */
  readonly field: (p: d.v3f, mip: number) => d.v2f;
  /** World-space half-width of the stored band at `mip`. */
  readonly bandWorld: (mip: number) => number;
  /** World-space voxel size at `mip`. For an analytic field, the sampling epsilon. */
  readonly voxelWorld: (mip: number) => number;
  /** Unit surface normal. */
  readonly normal: (p: d.v3f) => d.v3f;
}

/** Adapts a mip-mapped {@link SdfVolume} to {@link TracedField}. */
export function volumeField(volume: SdfVolume): TracedField {
  const layout = volume.layout;
  const sampleRaw = volume.sampleRaw;
  const bandWorld = volume.bandWorld;
  const voxel0 = volume.voxelSize;

  return {
    maxMip: volume.mipLevels - 1,
    groups: [volume.bindGroup],
    sample: (p: d.v3f, mip: number) => {
      'use gpu';
      // Outside the volume box, the distance to the box is a valid lower bound and
      // reads as saturated so the tracer stays at the coarsest level while it flies in.
      const dBox = sdf.sdBox3d(p - layout.$.params.center, layout.$.params.halfExtent);
      if (dBox > 0) {
        return d.vec2f(dBox, 1);
      }
      const raw = sampleRaw(p, mip).x;
      return d.vec2f(raw * bandWorld(mip), raw);
    },
    field: volume.sampleField,
    bandWorld,
    voxelWorld: (mip: number) => {
      'use gpu';
      return voxel0 * std.exp2(d.f32(mip));
    },
    normal: volume.gradient,
  };
}

export interface AnalyticFieldOptions {
  /** Distance beyond which the field is treated as saturated. Defaults to `worldSize`. */
  band?: number;
  /** Central-difference epsilon for normals. */
  epsilon?: number;
  /** Material id reported by `field`. Defaults to 0. */
  material?: number;
}

/**
 * Wraps a plain analytic SDF closure as a {@link TracedField}: single level, no
 * texture, no mip hopping. Useful on its own (analytic-only scenes) and as the
 * proof that the tracer is not coupled to the volume representation.
 */
export function analyticField(
  dist: (p: d.v3f) => number,
  options: AnalyticFieldOptions = {},
): TracedField {
  const band = options.band ?? 64;
  const eps = options.epsilon ?? 1e-3;
  const material = options.material ?? 0;
  const invBand = 1 / band;

  return {
    maxMip: 0,
    groups: [],
    sample: (p: d.v3f, _mip: number) => {
      'use gpu';
      const v = dist(p);
      return d.vec2f(v, std.clamp(v * invBand, -1, 1));
    },
    field: (p: d.v3f, _mip: number) => {
      'use gpu';
      return d.vec2f(dist(p), material);
    },
    bandWorld: (_mip: number) => {
      'use gpu';
      return band;
    },
    voxelWorld: (_mip: number) => {
      'use gpu';
      return eps;
    },
    normal: (p: d.v3f) => {
      'use gpu';
      const g = d.vec3f(
        dist(p + d.vec3f(eps, 0, 0)) - dist(p - d.vec3f(eps, 0, 0)),
        dist(p + d.vec3f(0, eps, 0)) - dist(p - d.vec3f(0, eps, 0)),
        dist(p + d.vec3f(0, 0, eps)) - dist(p - d.vec3f(0, 0, eps)),
      );
      const len = std.length(g);
      return std.select(d.vec3f(0, 1, 0), g * (1 / len), len > 1e-9);
    },
  };
}

/**
 * Union of two fields, traced as one. Both must agree on mip structure, so the
 * coarser one is queried at its own `maxMip` clamp. Claybook needed exactly this for
 * "clay world + fluid" (slide 60); nothing about it is Claybook-specific.
 */
export function unionField(a: TracedField, b: TracedField): TracedField {
  const maxMip = Math.min(a.maxMip, b.maxMip);
  return {
    maxMip,
    groups: [...a.groups, ...b.groups],
    sample: (p: d.v3f, mip: number) => {
      'use gpu';
      const sa = a.sample(p, mip);
      const sb = b.sample(p, mip);
      // Distance takes the nearer surface; the normalised value must take the
      // *smaller* saturation or the tracer would hop up over a nearby surface.
      return d.vec2f(std.min(sa.x, sb.x), std.min(sa.y, sb.y));
    },
    field: (p: d.v3f, mip: number) => {
      'use gpu';
      const fa = a.field(p, mip);
      const fb = b.field(p, mip);
      return std.select(fb, fa, fa.x < fb.x);
    },
    bandWorld: (mip: number) => {
      'use gpu';
      return std.min(a.bandWorld(mip), b.bandWorld(mip));
    },
    // `max`, not `min`: the tracer discounts every step by the interpolation slack of
    // the level it read, `0.866 * voxel`. If the nearer surface came from the coarser
    // child while the discount was computed from the finer child's voxel, the step
    // overshoots by the difference and the ray tunnels through the surface - a coarse
    // fluid volume unioned with a fine world punched hard-edged wedges through the
    // water. The larger voxel is the only conservative answer.
    voxelWorld: (mip: number) => {
      'use gpu';
      return std.max(a.voxelWorld(mip), b.voxelWorld(mip));
    },
    normal: (p: d.v3f) => {
      'use gpu';
      const fa = a.sample(p, 0);
      const fb = b.sample(p, 0);
      return std.select(b.normal(p), a.normal(p), fa.x < fb.x);
    },
  };
}

/**
 * Linear blend of two fields. `t = 0` is `a`, `t = 1` is `b`.
 *
 * A lerp of two distance functions is not itself a distance function - it is only
 * Lipschitz-1 if both inputs are, which they are, so the tracer stays safe; what it is
 * not is *exact*, so the intermediate surface is a plausible morph rather than a
 * metrically correct one. That is precisely what Claybook's shape morphing wanted
 * (GDC'18 slide 48): re-extract the particle cloud from `lerpField(a, b, t)` every
 * frame and the body flows from one shape into the other.
 */
export function lerpField(a: TracedField, b: TracedField, t: () => number): TracedField {
  const maxMip = Math.min(a.maxMip, b.maxMip);
  return {
    maxMip,
    groups: [...a.groups, ...b.groups],
    sample: (p: d.v3f, mip: number) => {
      'use gpu';
      const k = std.clamp(t(), 0, 1);
      const sa = a.sample(p, mip);
      const sb = b.sample(p, mip);
      // Saturation takes the min, as in `unionField`: a level may only be skipped when
      // *both* inputs promise a full band of empty space.
      return d.vec2f(std.mix(sa.x, sb.x, k), std.min(sa.y, sb.y));
    },
    field: (p: d.v3f, mip: number) => {
      'use gpu';
      const k = std.clamp(t(), 0, 1);
      const fa = a.field(p, mip);
      const fb = b.field(p, mip);
      return std.mix(fa, fb, k);
    },
    bandWorld: (mip: number) => {
      'use gpu';
      return std.min(a.bandWorld(mip), b.bandWorld(mip));
    },
    // `max`, not `min`: the tracer discounts every step by the interpolation slack of
    // the level it read, `0.866 * voxel`. If the nearer surface came from the coarser
    // child while the discount was computed from the finer child's voxel, the step
    // overshoots by the difference and the ray tunnels through the surface - a coarse
    // fluid volume unioned with a fine world punched hard-edged wedges through the
    // water. The larger voxel is the only conservative answer.
    voxelWorld: (mip: number) => {
      'use gpu';
      return std.max(a.voxelWorld(mip), b.voxelWorld(mip));
    },
    normal: (p: d.v3f) => {
      'use gpu';
      const k = std.clamp(t(), 0, 1);
      return std.normalize(std.mix(a.normal(p), b.normal(p), k));
    },
  };
}

/**
 * The same field, moved. `offset` is evaluated in GPU scope, so it can be a uniform and
 * the field follows something around at the cost of a 16-byte upload per frame.
 *
 * Translation is an isometry, so every promise the child makes survives it untouched -
 * distances, bands and voxel sizes are all unchanged, and only the sample point moves.
 * That is what lets a body-local volume (a baked shape, a morph pair) be used by a
 * system that works in world space.
 */
export function offsetField(field: TracedField, offset: () => d.v3f): TracedField {
  return {
    maxMip: field.maxMip,
    groups: field.groups,
    sample: (p: d.v3f, mip: number) => {
      'use gpu';
      return field.sample(p - offset(), mip);
    },
    field: (p: d.v3f, mip: number) => {
      'use gpu';
      return field.field(p - offset(), mip);
    },
    bandWorld: (mip: number) => {
      'use gpu';
      return field.bandWorld(mip);
    },
    voxelWorld: (mip: number) => {
      'use gpu';
      return field.voxelWorld(mip);
    },
    normal: (p: d.v3f) => {
      'use gpu';
      return field.normal(p - offset());
    },
  };
}
