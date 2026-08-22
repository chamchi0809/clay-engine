import { d, std } from 'typegpu';
import type { TracedField } from './field.ts';

/** Result of one sweep. `hit == 0` means the sweep left the interval without touching. */
export const SweepHit = d.struct({
  t: d.f32,
  hit: d.u32,
  /** Iterations consumed. Useful as a heat-map / perf overlay. */
  steps: d.u32,
  /** Field distance at the stop point, world units. */
  dist: d.f32,
});
export type SweepHitValue = d.Infer<typeof SweepHit>;

/**
 * Worst-case over-estimate of a trilinearly interpolated distance field, in voxels of
 * the level being read: half the cell diagonal, `sqrt(3)/2`. Reached when the nearest
 * feature is a point at a cell centre.
 */
export const INTERP_SLACK = 0.866;

export interface TracerOptions {
  /** Iteration cap. Claybook shipped ~100 for primary rays. */
  maxSteps?: number;
  /** Minimum forward progress per step, in mip-0 voxels. Guards against stalls. */
  minStep?: number;
  /**
   * Hit threshold in voxels of `minMip`, added on top of the swept radius. A plain ray
   * has zero radius and converges on the surface asymptotically, so without this it
   * would never report a hit.
   */
  hitEps?: number;
  /**
   * Finest mip the trace is allowed to reach. Claybook ran AO and shadow rays at a
   * coarse level for cache locality and bandwidth (slide 35); the cost is a blurrier
   * surface, which is free for those two.
   */
  minMip?: number;
}

/**
 * Tangent of the half-angle of the cone subtended by one pixel. Pass the result as
 * `aperture` to get Claybook's "perfect LOD" (slide 26): the trace stops as soon as
 * the remaining distance is below the pixel footprint, so cost scales with screen
 * area rather than scene complexity.
 */
export function pixelAperture(fovYRadians: number, pixelHeight: number): number {
  return Math.tan(fovYRadians / 2) / (0.5 * pixelHeight);
}

/**
 * Hierarchical sphere/cone/sphere-sweep tracer over any {@link TracedField}.
 *
 * The three trace kinds are one loop, because they only differ in how the swept
 * radius grows with `t`:
 *
 *   ray          radius(t) = 0
 *   sphere sweep radius(t) = radius
 *   cone         radius(t) = aperture * t
 *
 * With the free-space radius `D` at the current point, the furthest the swept shape
 * can advance is the tangency point:
 *
 *   (t' - t) + radius + aperture*t' = D   ->   t' = (t + D - radius) / (1 + aperture)
 *
 * That reduces to `t + D` for a ray, `t + D - radius` for a sphere sweep, and
 * `(t + D) / (1 + aperture)` for a cone. Slide 27 prints the cone factor as
 * `C / (C - aperture)` with `C = sqrt(aperture^2 + 1)`, which is greater than 1 and
 * would overshoot; the tangency condition above gives `C / (C + aperture)`, i.e.
 * `1 / (1 + sin(theta))`, which is what this uses.
 */
export function makeTracer(field: TracedField, options: TracerOptions = {}) {
  const maxSteps = options.maxSteps ?? 192;
  const minStepVoxels = options.minStep ?? 0.25;
  const hitEpsVoxels = options.hitEps ?? 0.5;
  const maxMip = field.maxMip;
  const minMip = Math.min(options.minMip ?? 0, maxMip);
  // Hops move by 2, so start on a level that lands exactly on `minMip` on the way down.
  const startMip = minMip + (maxMip - minMip - ((maxMip - minMip) % 2));
  const hitMip = minMip + 0.5;
  /**
   * Where a sweep is worth starting. A field that bounds itself says so; one that does not
   * gets marched in from `tMin` as before. Resolved here, when the tracer is built, so the
   * shader carries no branch for it.
   */
  const startAt = field.entry
    ?? ((_ro: d.v3f, _rd: d.v3f, tMin: number, _radius: number, _aperture: number) => {
      'use gpu';
      return tMin;
    });

  /**
   * `radius` grows the swept sphere by a constant, `aperture` grows it with distance.
   * Both zero = plain sphere tracing.
   */
  const sweep = (
    ro: d.v3f,
    rd: d.v3f,
    tMin: number,
    tMax: number,
    radius: number,
    aperture: number,
  ) => {
    'use gpu';
    const invGrowth = 1 / (1 + aperture);
    const minStep = field.voxelWorld(0) * minStepVoxels;
    const hitEps = field.voxelWorld(minMip) * hitEpsVoxels;
    // Nothing before this can be touched, so nothing before it is worth sampling. For a
    // field with no bounds of its own this is `tMin` and the loop is unchanged.
    const tStart = startAt(ro, rd, tMin, radius, aperture);
    let t = tStart;
    // Start coarse, on an even level so `mip -= 2` lands on 0 exactly.
    let mip = d.f32(startMip);
    let dist = d.f32(0);
    let hit = d.u32(0);
    let steps = d.u32(0);

    for (let i = d.u32(0); i < d.u32(maxSteps); i++) {
      steps = i + 1;
      const s = field.sample(ro + rd * t, mip);
      dist = s.x;
      // Trilinear interpolation of a *coarse* level over-estimates the true distance by
      // up to 0.87 of that level's voxel - worst case a feature thinner than the voxel,
      // which the level cannot represent at all. Stepping by the raw value there marches
      // straight through such features: a 0.7-radius torus tube is invisible at mip 2+,
      // so shadow and AO rays punched dashed holes through their own contact shadows.
      // Discount every level down to the finest one this trace trusts; at `minMip` the
      // discount is zero, so the finest level is still taken at face value (that is what
      // `hitEps` is for). Only ever shrinks the step, so the no-overshoot invariant below
      // still holds.
      const safe = dist - INTERP_SLACK * (field.voxelWorld(mip) - field.voxelWorld(minMip));

      // Tangency target for this sweep kind: 0 for a ray, `radius` for a sphere sweep,
      // `aperture * t` for a cone. `hitEps` is what makes a zero-radius ray terminate.
      const target = radius + aperture * t;
      // The discount can swallow a coarse level's whole reading, and then the step below
      // collapses to `minStep` and the trace crawls. A saturated *texture* sample can never
      // do that - a band is four voxels wide, so `safe` there is at least three of them -
      // but a field answering with a conservative bound of its own can, and a sweep that
      // leaves a bounded field re-enters that case for the rest of the interval. Going
      // finer is what makes the reading trustworthy again, so do that instead of crawling.
      if (safe <= 0 && mip >= hitMip) {
        mip = std.max(mip - 2, minMip);
        t = std.max(t - 0.5 * field.voxelWorld(mip), tStart);
        continue;
      }
      // A saturated sample means "at least a full band of empty space", so it can never
      // be a surface. That is also what keeps a field's conservative bounds (the volume
      // box in `volumeField`) from reading as geometry: they report saturation.
      if (safe < target + hitEps && s.y < 0.999) {
        if (mip >= hitMip) {
          // A coarse level's band is 2^mip times wider, so "close" there is only a
          // maybe. Drop two levels and step back half a voxel of the finer level
          // before believing it (slide 24). Reaching here on the conservative `safe`
          // value is also the only hop-down trigger needed: it fires exactly when this
          // level can no longer promise empty space.
          mip = std.max(mip - 2, minMip);
          t = std.max(t - 0.5 * field.voxelWorld(mip), tStart);
          continue;
        }
        hit = 1;
        // Close the `hitEps` gap, conservatively. Slide 25 extrapolates the whole
        // geometric tail, `(dist - target) / (1 - dist/prev)`, which is a *forward*
        // jump past the loop's own bound: where the nearest feature changes between
        // two samples the ratio goes to 1 and the jump lands deep inside the surface.
        // That showed up as a stable dither of black pixels on the ground wherever an
        // object was near enough to own the distance field.
        // One Lipschitz step is all that is needed and cannot overshoot: the surface is
        // at least `dist` away, so advancing by `dist - target` leaves at least
        // `target` of clearance.
        t = t + std.max(dist - target, 0);
        break;
      }

      // The tangency step never crosses the surface: with `D = dist(p(t))` and
      // `t' = (t + D - radius)/(1 + aperture)`, the Lipschitz bound gives
      // `dist(p(t')) >= D - (t' - t) = radius + aperture*t'`. So every `t` the loop
      // reaches is at or before the true tangency point - the invariant the pre-pass
      // depends on. `minStep` is the one exception, hence the tiny backoff callers
      // apply when they reuse `t` as a bound.
      t = std.max((t + safe - radius) * invGrowth, t + minStep);

      if (s.y >= 0.999) {
        // Saturated: at least a full band of empty space here, so a coarser level
        // (band twice as wide) is safe and steps twice as far.
        mip = std.min(mip + 2, d.f32(startMip));
      }

      if (t > tMax) {
        break;
      }
    }

    return SweepHit({ t: std.min(t, tMax), hit, steps, dist });
  };

  return {
    field,
    maxSteps,
    /**
     * Forward slack a caller must subtract before reusing a `t` as a lower bound.
     * A shader closure, not a number: the voxel size lives in the field's uniform.
     */
    stepSlack: () => {
      'use gpu';
      return field.voxelWorld(0) * minStepVoxels;
    },
    sweep,
    /** Plain hierarchical sphere trace. */
    ray: (ro: d.v3f, rd: d.v3f, tMin: number, tMax: number) => {
      'use gpu';
      return sweep(ro, rd, tMin, tMax, d.f32(0), d.f32(0));
    },
    /**
     * Cone trace. For primary rays pass {@link pixelAperture}; for the coarse
     * pre-pass, that times the tile size (slide 28).
     */
    cone: (ro: d.v3f, rd: d.v3f, tMin: number, tMax: number, aperture: number) => {
      'use gpu';
      return sweep(ro, rd, tMin, tMax, d.f32(0), aperture);
    },
    /** Swept sphere - character/projectile collision, one query per body. */
    sphere: (ro: d.v3f, rd: d.v3f, tMin: number, tMax: number, radius: number) => {
      'use gpu';
      return sweep(ro, rd, tMin, tMax, radius, d.f32(0));
    },
  };
}

export type Tracer = ReturnType<typeof makeTracer>;
