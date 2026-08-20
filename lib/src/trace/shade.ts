import { d, std } from 'typegpu';
import type { TgpuReadonly, TgpuRoot } from 'typegpu';
import { cosineHemisphere, hash3 } from '../math/gpu.ts';
import type { TracedField } from './field.ts';
import { makeTracer } from './march.ts';

/** A directional light. `size` is the tangent of its angular radius: 0 = hard shadows. */
export const DirLight = d.struct({
  dir: d.vec3f,
  size: d.f32,
  color: d.vec3f,
  intensity: d.f32,
});
export type DirLightValue = d.InferInput<typeof DirLight>;

/**
 * One entry of a material palette, indexed by the field's material channel.
 *
 * The last three fields are what make a surface see-through, and they live *here* rather
 * than on an object for a reason: the renderer knows a scene as one distance field with a
 * material channel, not as a list of things. Put transparency in the palette and every
 * kind of object gets it at once - an authored solid, a simulated body, a fluid bake, and
 * anything a third party writes against `Entity` - with no per-type render code.
 */
export const Material = d.struct({
  albedo: d.vec3f,
  roughness: d.f32,
  emissive: d.vec3f,
  metallic: d.f32,
  /**
   * How much of the surface is the surface, `0..1`. 1 is opaque and takes the fast path.
   * Below 1 the shading of the surface is mixed over whatever is behind it.
   */
  opacity: d.f32,
  /**
   * Index of refraction. 1 bends nothing, water is 1.33, glass ~1.5. Only meaningful
   * below `opacity` 1.
   */
  ior: d.f32,
  /**
   * Beer-Lambert absorption per world unit travelled through the body, tinted by
   * `1 - albedo`. 0 is perfectly clear glass; a thick pool of water at 0.6 goes visibly
   * blue at the bottom while its shallow edge stays clear, which is the whole reason
   * absorption is a length integral rather than a constant tint.
   */
  absorption: d.f32,
});
export type MaterialValue = d.InferInput<typeof Material>;

/**
 * A material as a game writes one: everything but `albedo` has a default, so an opaque
 * material stays four lines and transparency is opt-in per entry.
 */
export interface MaterialSpec {
  albedo: readonly [number, number, number];
  roughness?: number;
  emissive?: readonly [number, number, number];
  metallic?: number;
  /** See {@link Material.opacity}. Defaults to 1 - opaque. */
  opacity?: number;
  /** See {@link Material.ior}. Defaults to 1.33, which only matters once `opacity < 1`. */
  ior?: number;
  /** See {@link Material.absorption}. Defaults to 0.5. */
  absorption?: number;
}

/** Fills a {@link MaterialSpec}'s defaults. */
export function normalizeMaterial(spec: MaterialSpec): MaterialValue {
  return {
    albedo: [spec.albedo[0], spec.albedo[1], spec.albedo[2]],
    roughness: spec.roughness ?? 0.7,
    emissive: [...(spec.emissive ?? [0, 0, 0])] as [number, number, number],
    metallic: spec.metallic ?? 0,
    opacity: spec.opacity ?? 1,
    ior: spec.ior ?? 1.33,
    absorption: spec.absorption ?? 0.5,
  };
}

/** True when this material transmits anything at all. A JS test, run when a game boots. */
export const isTransparent = (spec: MaterialSpec): boolean => (spec.opacity ?? 1) < 1;

/** Upper bound on palette entries; the g-buffer stores the id in a `unorm8`. */
export const MAX_MATERIALS = 256;
export type MaterialPalette = TgpuReadonly<d.WgslArray<typeof Material>>;

export function createPalette(root: TgpuRoot, entries: readonly MaterialSpec[]): MaterialPalette {
  if (entries.length > MAX_MATERIALS) {
    throw new Error(`createPalette: ${entries.length} materials exceeds ${MAX_MATERIALS}`);
  }
  const count = Math.max(1, entries.length);
  const filled = entries.map(normalizeMaterial);
  // An empty palette still needs slot 0, or the array type has length 0 and nothing binds.
  while (filled.length < count) {
    filled.push(normalizeMaterial({ albedo: [0.5, 0.5, 0.5] }));
  }
  return root.createReadonly(d.arrayOf(Material, count), filled);
}

/**
 * Reads a palette entry for a fractional id. The field's material channel is a
 * *blend* of the ids of the brushes that meet at a voxel, so a seam between two
 * materials interpolates instead of stair-stepping.
 */
export function makePaletteSampler(palette: MaterialPalette, count: number) {
  const last = Math.max(0, count - 1);
  return (id: number) => {
    'use gpu';
    const clamped = std.clamp(id, 0, last);
    const i0 = d.u32(std.floor(clamped));
    const i1 = std.min(i0 + 1, d.u32(last));
    const f = clamped - std.floor(clamped);
    const a = palette.$[i0];
    const b = palette.$[i1];
    return Material({
      albedo: std.mix(a.albedo, b.albedo, f),
      roughness: std.mix(a.roughness, b.roughness, f),
      emissive: std.mix(a.emissive, b.emissive, f),
      metallic: std.mix(a.metallic, b.metallic, f),
      opacity: std.mix(a.opacity, b.opacity, f),
      ior: std.mix(a.ior, b.ior, f),
      absorption: std.mix(a.absorption, b.absorption, f),
    });
  };
}

export interface ShadingOptions {
  /** Max AO ray length in world units. */
  aoDistance?: number;
  /** AO cone half-angle tangent. Wider = softer, cheaper, less contact detail. */
  aoAperture?: number;
  /**
   * Finest mip AO may sample. Claybook traced these coarse for bandwidth (slide 35),
   * but a coarse level also blurs away every concavity narrower than its voxel, so 0
   * is the default and raising it is the explicit perf trade.
   */
  aoMinMip?: number;
  aoSteps?: number;
  shadowSteps?: number;
  shadowMinMip?: number;
  /**
   * Ray start offset in *mip-0* voxels - the scale the geometry is actually stored at.
   * Scaling this by the trace's `minMip` instead would widen the blind zone around
   * every contact point by 2^minMip, which reads as a bright gap between an object and
   * its own shadow, and as a missing crease in AO.
   */
  bias?: number;
}

/**
 * Screen-space-free lighting queries: everything is a distance-field trace, which is
 * what let Claybook ship with no baked lighting, AO or shadows at all (slide 5).
 * Both queries are stochastic by design - one sample per pixel per frame plus
 * temporal accumulation, exactly as in slides 35 and 39.
 */
export function makeShading(field: TracedField, options: ShadingOptions = {}) {
  const aoDistance = options.aoDistance ?? 2.5;
  // Narrow, because the direction is now cosine-distributed over the whole hemisphere:
  // a wide cone around a near-tangent direction clips the surface it started on and
  // reports occlusion that is not there.
  const aoAperture = options.aoAperture ?? 0.08;
  // A single deterministic cone has to cover the hemisphere on its own, so it is wide
  // where the stochastic one is narrow. 0.45 is a ~24 degree half-angle.
  const fixedAperture = Math.max(aoAperture, 0.45);
  const aoMinMip = Math.min(options.aoMinMip ?? 0, field.maxMip);
  const biasVoxels = options.bias ?? 1;
  const shadowMinMip = Math.min(options.shadowMinMip ?? 0, field.maxMip);

  const aoTracer = makeTracer(field, {
    maxSteps: options.aoSteps ?? 32,
    minMip: aoMinMip,
  });
  const shadowTracer = makeTracer(field, {
    maxSteps: options.shadowSteps ?? 64,
    minMip: shadowMinMip,
  });

  /**
   * Cone-traced ambient occlusion along the normal (slide 35). One jittered cone per
   * pixel; the temporal filter turns the noise into a smooth long-range term.
   */
  const coneAO = (p: d.v3f, n: d.v3f, seed: number) => {
    'use gpu';
    const bias = field.voxelWorld(0) * biasVoxels;
    const r = hash3(seed);
    // Cosine-weighted over the hemisphere: that is the weighting the AO integral
    // actually wants, so the per-pixel mean converges to the right value. Jittering a
    // fixed direction around the normal instead clusters samples at the pole, which
    // both under-occludes and leaves a bimodal (hit / escape) estimator whose noise
    // survives temporal accumulation as blotches.
    const dir = cosineHemisphere(n, r.xy);
    // Start *at* the offset origin. Also skipping the first `bias` of the ray would
    // blind the cone to everything within ~2 voxels - exactly the contact region AO
    // exists to darken.
    const hit = aoTracer.cone(p + n * bias, dir, d.f32(0), aoDistance, aoAperture);
    // Contact at t=0 is fully occluded, escaping the interval is fully open.
    return std.select(d.f32(1), std.clamp(hit.t / aoDistance, 0, 1), hit.hit === 1);
  };

  /**
   * Soft shadow: one ray per pixel per frame at a random direction inside the light's
   * angular disc, integrated by the temporal filter into a real penumbra.
   *
   * Claybook's triangulated coverage estimator (slide 39) gets a smooth penumbra out of
   * a *single* ray, but it needs its own copy of the marching loop - and a second
   * marcher is a second place for the mip hopping and the coarse-level discount to be
   * wrong. It was: its penumbra term only ever fired within `lightSize * t` of an
   * occluder, so in practice it was a hard shadow whose hit test disagreed with the
   * tracer's, and it bit hard-edged notches out of every cast shadow. Since temporal
   * accumulation is not optional in this renderer anyway, jittered rays through the one
   * tested loop are both shorter and unbiased.
   */
  const softShadow = (
    p: d.v3f,
    n: d.v3f,
    dir: d.v3f,
    tMax: number,
    lightSize: number,
    seed: number,
  ) => {
    'use gpu';
    const bias = field.voxelWorld(0) * biasVoxels;
    const r = hash3(seed);
    // Uniform point on the light's disc, as an angular offset from `dir`.
    const up = std.select(d.vec3f(1, 0, 0), d.vec3f(0, 1, 0), std.abs(dir.y) < 0.9);
    const tx = std.normalize(std.cross(up, dir));
    const ty = std.cross(dir, tx);
    const a = r.x * 6.2831853;
    const rad = lightSize * std.sqrt(r.y);
    const jittered = std.normalize(
      dir + tx * (rad * std.cos(a)) + ty * (rad * std.sin(a)),
    );
    const hit = shadowTracer.ray(p + n * bias, jittered, d.f32(0), tMax);
    return std.select(d.f32(1), d.f32(0), hit.hit === 1);
  };

  /**
   * The same shadow query with no jitter and a point-sized light.
   *
   * Exists because the transparency layer is composited *after* the temporal filter (see
   * `render/composite.ts`), so it has nothing to average its noise into. A stochastic
   * shadow there would be salt-and-pepper on every pane of glass; one deterministic ray
   * is stable at the cost of a hard shadow edge, which on a surface you can see through
   * is the trade nobody notices.
   */
  const hardShadow = (p: d.v3f, n: d.v3f, dir: d.v3f, tMax: number) => {
    'use gpu';
    const bias = field.voxelWorld(0) * biasVoxels;
    const hit = shadowTracer.ray(p + n * bias, dir, d.f32(0), tMax);
    return std.select(d.f32(1), d.f32(0), hit.hit === 1);
  };

  /**
   * One cone straight up the normal. Deterministic, and for the same reason as
   * {@link hardShadow}: it is the AO term for surfaces that never reach the temporal
   * filter. The aperture is widened, because a single cone has to stand in for the whole
   * hemisphere integral that {@link coneAO} converges to over many frames.
   */
  const axisAO = (p: d.v3f, n: d.v3f) => {
    'use gpu';
    const bias = field.voxelWorld(0) * biasVoxels;
    const hit = aoTracer.cone(p + n * bias, n, d.f32(0), aoDistance, fixedAperture);
    return std.select(d.f32(1), std.clamp(hit.t / aoDistance, 0, 1), hit.hit === 1);
  };

  return { coneAO, softShadow, hardShadow, axisAO, aoTracer, shadowTracer };
}

export type Shading = ReturnType<typeof makeShading>;

/**
 * Schlick's approximation of the Fresnel term: how much of the light bounces off a
 * surface instead of going into it. `f0` is the reflectance head-on (0.04 for a
 * dielectric); at a grazing angle everything reflects, which is why the far edge of a
 * pool is a mirror and the near edge is not.
 */
export const fresnelSchlick = (cosTheta: number, f0: number) => {
  'use gpu';
  const c = std.clamp(1 - cosTheta, 0, 1);
  return f0 + (1 - f0) * (c * c) * (c * c) * c;
};

/** Lambert + GGX-ish specular. Enough for clay; swap out per game. */
export const shadeDirect = (
  n: d.v3f,
  view: d.v3f,
  light: d.v3f,
  albedo: d.v3f,
  roughness: number,
) => {
  'use gpu';
  const ndl = std.max(std.dot(n, light), 0);
  const h = std.normalize(light + view);
  const ndh = std.max(std.dot(n, h), 0);
  const a = std.max(roughness * roughness, 1e-3);
  const dTerm = a / (3.14159265 * std.pow(ndh * ndh * (a - 1) + 1, 2) + 1e-6);
  const fresnel = std.pow(1 - std.max(std.dot(h, view), 0), 5) * 0.9 + 0.04;
  return (albedo * (1 / 3.14159265) + d.vec3f(dTerm * fresnel)) * ndl;
};
