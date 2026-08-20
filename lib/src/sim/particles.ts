import { d, std } from 'typegpu';
import type { TgpuMutable, TgpuRoot } from 'typegpu';

/** `Particle.body` value for a particle that belongs to no shape-matched body. */
export const FREE_PARTICLE = 0xffffffff;

/**
 * One simulation particle.
 *
 * Claybook's clay bodies are particle clouds that double as mesh vertices (GDC'18
 * slides 45-47), so a particle carries both halves: the physics state (`pos`, `prev`,
 * `vel`, `velPrev` - four fields because the integrator is BDF2, which needs step
 * `n-1`) and the render state (`restNormal`, `material`).
 *
 * `rest` is the position in the body's *rest* frame. Shape matching pulls the particle
 * back toward `com + R * (rest - restCom)`; plasticity moves `rest` itself.
 */
export const Particle = d.struct({
  pos: d.vec3f,
  /** Owning body index, or {@link FREE_PARTICLE}. */
  body: d.u32,
  /** Position at step `n-1`. */
  prev: d.vec3f,
  material: d.f32,
  vel: d.vec3f,
  _padA: d.f32,
  /** Velocity at step `n-1`. */
  velPrev: d.vec3f,
  _padB: d.f32,
  rest: d.vec3f,
  _padC: d.f32,
  restNormal: d.vec3f,
  _padD: d.f32,
});
export type ParticleValue = d.InferInput<typeof Particle>;

/** Bit flags in {@link Body}`.flags`. */
export const BodyFlags = {
  /** Skip integration and shape matching; still extracted and drawn. */
  kinematic: 1,
  /** Ignore gravity. */
  weightless: 2,
} as const;

/**
 * A shape-matched particle body. Claybook GDC'18 slides 49-51.
 *
 * `rot` is stored rather than recomputed from scratch because the polar decomposition
 * is iterative: warm-starting from last frame's rotation converges in one or two
 * iterations instead of five or six.
 */
export const Body = d.struct({
  /** Current centre of mass, recomputed each substep. */
  com: d.vec3f,
  /** Index of this body's first particle. */
  first: d.u32,
  /** Current orientation, xyzw. */
  rot: d.vec4f,
  /** Centre of mass of the rest shape. */
  restCom: d.vec3f,
  count: d.u32,
  /** Fraction of the shape-match correction applied per substep, `[0, 1]`. */
  stiffness: d.f32,
  /** Rest-shape drift per second, `[0, 1]`. 0 keeps the body rigid, 1 is putty. */
  plasticity: d.f32,
  material: d.f32,
  flags: d.u32,
});

/** CPU-owned half of a {@link Body}. `first`/`count`/`com`/`rot` belong to the GPU. */
export interface BodyDesc {
  /** Fraction of the shape-match correction applied per substep, `[0, 1]`. */
  stiffness?: number;
  /** Rest-shape drift per second, `[0, 1]`. 0 keeps the body rigid, 1 is putty. */
  plasticity?: number;
  material?: number;
  flags?: number;
}

/** Shared GPU allocation counters. Reset per extraction, read by the indirect draws. */
export const SimCounters = d.struct({
  particles: d.atomic(d.u32),
  quads: d.atomic(d.u32),
});

export interface ParticleSetOptions {
  /** Maximum particles. Extraction stops emitting past this. */
  capacity?: number;
  /** Maximum shape-matched bodies. */
  maxBodies?: number;
  label?: string;
}

/**
 * Storage for a particle cloud plus the bodies that own slices of it.
 *
 * Nothing here knows how the particles were produced or what they will be used for -
 * {@link SurfaceExtractor} fills them from any field, {@link ClaySolver} simulates
 * them, {@link ParticleMesh} draws them, and a game can do all, some, or none of that.
 *
 * Body records are mirrored on the CPU because there are only a handful of them and a
 * whole-array upload is cheaper than tracking dirty ranges. Particle data never comes
 * back to the CPU.
 */
export class ParticleSet {
  readonly root: TgpuRoot;
  readonly capacity: number;
  readonly maxBodies: number;
  readonly particles: TgpuMutable<d.WgslArray<typeof Particle>>;
  readonly bodies: TgpuMutable<d.WgslArray<typeof Body>>;

  /** `(index) -> Particle` shader accessor. */
  readonly at: (i: number) => d.Infer<typeof Particle>;

  private declared = 0;

  constructor(root: TgpuRoot, options: ParticleSetOptions = {}) {
    this.root = root;
    this.capacity = options.capacity ?? 1 << 16;
    this.maxBodies = options.maxBodies ?? 32;
    const label = options.label ?? 'particles';

    this.particles = root
      .createMutable(d.arrayOf(Particle, this.capacity))
      .$name(label);
    this.bodies = root
      .createMutable(d.arrayOf(Body, this.maxBodies))
      .$name(`${label}Bodies`);

    const particles = this.particles;
    this.at = (i: number) => {
      'use gpu';
      return particles.$[i];
    };
  }

  /** Number of body records currently declared. */
  get bodyCount(): number {
    return this.declared;
  }

  /**
   * Declares a body and returns its index. `first`/`count` are filled in by the
   * extractor on the GPU; everything else comes from here.
   */
  addBody(desc: BodyDesc = {}): number {
    if (this.declared >= this.maxBodies) {
      throw new Error(`ParticleSet: body limit ${this.maxBodies} reached`);
    }
    const idx = this.declared++;
    this.bodies.writePartial([
      {
        idx,
        value: {
          com: d.vec3f(),
          first: 0,
          rot: d.vec4f(0, 0, 0, 1),
          restCom: d.vec3f(),
          count: 0,
          stiffness: desc.stiffness ?? 0.5,
          plasticity: desc.plasticity ?? 0,
          material: desc.material ?? 0,
          flags: desc.flags ?? 0,
        },
      },
    ]);
    return idx;
  }

  /**
   * Patches the CPU-owned fields of a body. Only the keys present are written, so this
   * never clobbers the `first`/`count`/`com`/`rot` the GPU maintains.
   */
  setBody(index: number, patch: BodyDesc): void {
    if (index >= this.declared) {
      throw new Error(`ParticleSet: no body ${index}`);
    }
    this.bodies.writePartial([{ idx: index, value: patch }]);
  }

  /** Workgroups of `size` needed to cover the whole capacity. */
  dispatchGroups(size: number): number {
    return Math.ceil(this.capacity / size);
  }

  /**
   * Starts tracking a body's centre of mass on the CPU. See {@link BodyTracker} for why
   * a game should not read `bodies` itself.
   *
   * ponytail: each tracker issues its own readback of the whole `bodies` array. With a
   * handful of bodies that is one small copy per frame; if a game tracks dozens, share
   * one read per frame across them.
   */
  track(index: number): BodyTracker {
    if (index >= this.declared) {
      throw new Error(`ParticleSet: no body ${index}`);
    }
    return new BodyTracker(this, index);
  }

  /** Shader-side helper: is this index inside the declared particle range? */
  readonly inRange = (i: number, count: number) => {
    'use gpu';
    return i < std.min(count, d.u32(this.capacity));
  };
}

/**
 * CPU-side view of one body's centre of mass.
 *
 * Particle state lives on the GPU, so this is a readback, and a readback resolves some
 * frames after it is issued rather than on the next line. Two consequences, both of
 * which a game would otherwise have to rediscover the hard way:
 *
 *  - Samples arrive at an irregular cadence *slower* than the frame rate, so they are a
 *    staircase, not a curve. Anything pinned straight to them - a follow camera above
 *    all - visibly shakes at the readback rate.
 *  - A difference quotient over "one frame" therefore reports a velocity several times
 *    too large, which quietly breaks every threshold built on it.
 *
 * So: {@link raw} is the newest sample, for logic that wants the truth. {@link vel} is
 * differenced over the time that actually elapsed. {@link pos} is an exponentially
 * filtered position advanced every frame, for anything a human looks at.
 */
export class BodyTracker {
  readonly set: ParticleSet;
  readonly index: number;
  /** Newest centre of mass from the GPU. A staircase; do not point a camera at it. */
  raw: [number, number, number] = [0, 0, 0];
  /** Smoothed position, advanced every {@link advance}. Point the camera at this. */
  pos: [number, number, number] = [0, 0, 0];
  /** Metres per second, differenced over the real interval between samples. */
  vel: [number, number, number] = [0, 0, 0];
  /** Newest fitted orientation from the GPU, xyzw unit quaternion. */
  rot: [number, number, number, number] = [0, 0, 0, 1];
  /**
   * Filter cutoff in 1/s. Higher follows tighter and lets more of the staircase
   * through; 14 costs about 70 ms of lag and removes it entirely.
   */
  rate = 14;

  private reading = false;
  private stamp = 0;

  constructor(set: ParticleSet, index: number) {
    this.set = set;
    this.index = index;
  }

  /** Teleports both the sample and the filter. Call when the body is repositioned. */
  reset(pos: readonly [number, number, number]): void {
    this.raw = [pos[0], pos[1], pos[2]];
    this.pos = [pos[0], pos[1], pos[2]];
    this.vel = [0, 0, 0];
    this.rot = [0, 0, 0, 1];
    this.stamp = 0;
  }

  /**
   * Requests a fresh sample and advances the filter by `dt` seconds. Call once per
   * frame; at most one readback is ever in flight, so calling it every frame is safe
   * whatever the latency turns out to be.
   */
  sync(dt: number): void {
    this.advance(dt);
    if (this.reading) {
      return;
    }
    this.reading = true;
    void this.set.bodies
      .read()
      .then((bodies) => {
        const b = bodies[this.index];
        if (!b || b.count === 0 || !Number.isFinite(b.com.x)) {
          return;
        }
        const now = performance.now();
        const gap = this.stamp > 0 ? Math.max(1e-3, (now - this.stamp) / 1000) : 0;
        this.stamp = now;
        const next: [number, number, number] = [b.com.x, b.com.y, b.com.z];
        if (gap > 0) {
          this.vel = [
            (next[0] - this.raw[0]) / gap,
            (next[1] - this.raw[1]) / gap,
            (next[2] - this.raw[2]) / gap,
          ];
        }
        this.raw = next;
        this.rot = [b.rot.x, b.rot.y, b.rot.z, b.rot.w];
      })
      .finally(() => {
        this.reading = false;
      });
  }

  /** Moves {@link pos} towards {@link raw}. Frame-rate independent by construction. */
  advance(dt: number): void {
    const k = 1 - Math.exp(-this.rate * Math.max(dt, 0));
    for (let i = 0; i < 3; i++) {
      this.pos[i]! += (this.raw[i]! - this.pos[i]!) * k;
    }
  }
}
