import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePass, TgpuComputePipeline, TgpuRoot, TgpuUniform } from 'typegpu';
import { extractRotation, quatConj, quatRotate } from '../math/gpu.ts';
import type { TracedField } from '../trace/field.ts';
import { BodyFlags, type ParticleSet } from './particles.ts';

/** Threads per body in the shape-matching reduction. */
const REDUCE = 256;
/** Threads per workgroup in the per-particle passes. */
const PARTICLE_GROUP = 64;

export const SolveParams = d.struct({
  gravity: d.vec3f,
  /** Substep length, seconds. */
  h: d.f32,
  /** Particle radius used against the world field. */
  radius: d.f32,
  /** Tangential motion removed on contact, `[0, 1]`. */
  friction: d.f32,
  /** Velocity retained per substep, `[0, 1]`. */
  damping: d.f32,
  /** Speed cap, world units per second. Keeps a bad frame from exploding the cloud. */
  maxSpeed: d.f32,
  /**
   * Extra acceleration applied to {@link SolveParams.pushBody} only - player control,
   * wind, a magnet. Uniform over the body, so it never fights shape matching.
   */
  push: d.vec3f,
  /** Body {@link SolveParams.push} applies to. Out-of-range disables it. */
  pushBody: d.u32,
  /**
   * One-shot velocity change for {@link SolveParams.pushBody}, applied by the `kick`
   * dispatch before the substep loop rather than as an acceleration.
   *
   * A jump has to be an impulse, not a force: fed through the BDF2 predictor as
   * `dv / dt` over one frame, only about a quarter of it survives into the velocity the
   * next frame reads, and the fraction depends on the substep count.
   */
  kick: d.vec3f,
  _pad: d.f32,
});
export type SolveParamsValue = d.InferInput<typeof SolveParams>;

export interface ClaySolverOptions {
  gravity?: readonly [number, number, number];
  /** Simulated seconds per {@link ClaySolver.step}. Fixed, so the sim is deterministic. */
  dt?: number;
  substeps?: number;
  radius?: number;
  friction?: number;
  damping?: number;
  maxSpeed?: number;
  /** Gauss-Newton iterations in the polar decomposition. Warm-started, so 2 is plenty. */
  rotationIterations?: number;
}

/**
 * Shape-matched particle bodies colliding against a signed distance field.
 * Claybook GDC'18 slides 49-53.
 *
 * Two things make this cheap enough to run on every particle of every body every
 * substep:
 *
 *  - **Collision is a field lookup.** No broadphase, no contact list, no pairwise
 *    anything: one `sample` says how deep the particle is and one `normal` says which
 *    way out. That is the whole reason Claybook's world is an SDF.
 *  - **Shape matching is two reductions.** A body has no constraint graph; its shape
 *    is restored by fitting one rigid transform to the whole cloud (Müller et al.
 *    2005) and pulling every particle a fraction of the way to its fitted position.
 *    Plasticity is then just letting the rest shape follow.
 *
 * The integrator is BDF2 for the predictor, which is what lets a 60 Hz step stay
 * stable at high stiffness.
 *
 * Nothing here knows about clay, brushes or Claybook - it is `ParticleSet` plus any
 * {@link TracedField} to collide with.
 */
export class ClaySolver {
  readonly root: TgpuRoot;
  readonly set: ParticleSet;
  readonly world: TracedField;
  readonly params: TgpuUniform<typeof SolveParams>;
  readonly substeps: number;
  /** Simulated seconds per {@link step}. */
  readonly dt: number;

  private readonly kick: TgpuComputePipeline;
  private readonly predict: TgpuComputePipeline;
  private readonly match: TgpuComputePipeline;
  private readonly apply: TgpuComputePipeline;
  private readonly finalize: TgpuComputePipeline;
  private readonly particleGroups: number;
  /** Whether {@link setForce} left an impulse for the next {@link step} to apply. */
  private kicking = false;

  constructor(set: ParticleSet, world: TracedField, options: ClaySolverOptions = {}) {
    const root = set.root;
    this.root = root;
    this.set = set;
    this.world = world;
    this.dt = options.dt ?? 1 / 60;
    this.substeps = options.substeps ?? 2;
    this.particleGroups = Math.ceil(set.capacity / PARTICLE_GROUP);
    const rotIters = options.rotationIterations ?? 2;

    const g = options.gravity ?? [0, -9.81, 0];
    const params = root.createUniform(SolveParams, {
      gravity: [g[0], g[1], g[2]],
      h: this.dt / this.substeps,
      radius: options.radius ?? 0.1,
      friction: options.friction ?? 0.35,
      damping: options.damping ?? 0.995,
      maxSpeed: options.maxSpeed ?? 40,
      push: [0, 0, 0],
      pushBody: 0xffffffff,
      kick: [0, 0, 0],
      _pad: 0,
    });
    this.params = params;

    const particles = set.particles;
    const bodies = set.bodies;
    const maxBodies = set.maxBodies;
    const capacity = set.capacity;

    /**
     * Owning body of particle `i`, or `maxBodies` when the slot is not live. A slot that
     * was never written reads `body == 0`, which is a legal index, so membership is
     * decided by the body's own `[first, first + count)` slice instead of by the field.
     */
    const ownerOf = (i: number) => {
      'use gpu';
      const b = particles.$[i].body;
      if (b >= d.u32(maxBodies)) {
        return d.u32(maxBodies);
      }
      const bd = bodies.$[b];
      if (i < bd.first || i >= bd.first + bd.count) {
        return d.u32(maxBodies);
      }
      return b;
    };

    this.kick = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [PARTICLE_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= d.u32(capacity)) {
          return;
        }
        const b = ownerOf(i);
        if (b !== params.$.pushBody || b >= d.u32(maxBodies)) {
          return;
        }
        if ((bodies.$[b].flags & d.u32(BodyFlags.kinematic)) !== d.u32(0)) {
          return;
        }
        const p = particles.$[i];
        const dv = params.$.kick;
        // Both velocity slots and the previous position, or the BDF2 predictor spends the
        // next substep undoing most of the kick: it extrapolates from `pos - prev` and
        // from the two-step velocity history, and all three have to agree.
        particles.$[i].vel = d.vec3f(p.vel + dv);
        particles.$[i].velPrev = d.vec3f(p.velPrev + dv);
        particles.$[i].prev = d.vec3f(p.prev - dv * params.$.h);
      }),
    });

    this.predict = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [PARTICLE_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= d.u32(capacity)) {
          return;
        }
        const b = ownerOf(i);
        if (b >= d.u32(maxBodies)) {
          return;
        }
        const flags = bodies.$[b].flags;
        if ((flags & d.u32(BodyFlags.kinematic)) !== d.u32(0)) {
          return;
        }
        const pr = params.$;
        const p = particles.$[i];
        let acc = std.select(pr.gravity, d.vec3f(), (flags & d.u32(BodyFlags.weightless)) !== d.u32(0));
        if (b === pr.pushBody) {
          acc = acc + pr.push;
        }
        // BDF2 predictor: the (4 x_n - x_{n-1}) / 3 + (2h/3) x' form. Second order and
        // strongly damping on the modes an explicit step would blow up on.
        const vPred = (p.vel * 4 - p.velPrev) * (1 / 3) + acc * (2 * pr.h / 3);
        let q = (p.pos * 4 - p.prev) * (1 / 3) + vPred * (2 * pr.h / 3);

        // World collision: one distance says how deep, one gradient says which way out.
        const dw = world.sample(q, 0).x;
        let prevOut = d.vec3f(p.pos);
        if (dw < pr.radius) {
          const n = world.normal(q);
          const push = pr.radius - dw;
          q = q + n * push;
          // A pushout is a correction, not motion. Velocity is extracted as
          // `(pos - prev) / h`, so leaving `prev` behind turns every resting contact into
          // a small upward kick - the body then trembles at the substep rate forever.
          // Sliding `prev` along the normal by the same amount is restitution 0: normal
          // velocity ends at zero, tangential velocity survives untouched.
          prevOut = p.pos + n * push;
          // Positional friction: undo part of this substep's tangential slide. Doing it
          // on positions rather than velocities keeps it stable at any stiffness,
          // because the velocity is derived from the positions afterwards.
          const dp = q - p.pos;
          q = q - (dp - n * std.dot(dp, n)) * pr.friction;
        }
        particles.$[i].prev = d.vec3f(prevOut);
        particles.$[i].pos = d.vec3f(q);
      }),
    });

    // 3 vec3f slots per thread: [sum pos | sum rest | -] in phase 1, the three columns
    // of the covariance matrix in phase 2. 12 KB, inside the 16 KB workgroup budget.
    const gsm = tgpu.workgroupVar(d.arrayOf(d.vec3f, REDUCE * 3));
    const reduceGsm = (li: number) => {
      'use gpu';
      for (let s = d.u32(REDUCE / 2); s > d.u32(0); s = s >>> 1) {
        if (li < s) {
          for (const c of std.range(0, 3)) {
            gsm.$[li * 3 + d.u32(c)] = d.vec3f(gsm.$[li * 3 + d.u32(c)] + gsm.$[(li + s) * 3 + d.u32(c)]);
          }
        }
        std.workgroupBarrier();
      }
    };

    this.match = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [REDUCE],
        in: { lid: d.builtin.localInvocationId, wid: d.builtin.workgroupId },
      })(({ lid, wid }) => {
        'use gpu';
        const b = wid.x;
        const li = lid.x;
        const bd = bodies.$[b];
        const first = bd.first;
        const count = bd.count;

        // --- phase 1: the two centroids ------------------------------------
        let sp = d.vec3f();
        let sr = d.vec3f();
        for (let t = li; t < count; t = t + d.u32(REDUCE)) {
          const p = particles.$[first + t];
          sp = sp + p.pos;
          sr = sr + p.rest;
        }
        gsm.$[li * 3 + 0] = d.vec3f(sp);
        gsm.$[li * 3 + 1] = d.vec3f(sr);
        gsm.$[li * 3 + 2] = d.vec3f();
        std.workgroupBarrier();
        reduceGsm(li);
        const inv = 1 / std.max(d.f32(count), 1);
        const com = gsm.$[0] * inv;
        // `restCom` is recomputed rather than remembered because plasticity moves
        // `rest`; a stale centroid would slowly translate the whole body.
        const restCom = gsm.$[1] * inv;
        std.workgroupBarrier();

        // --- phase 2: A = sum (p - com) (r - restCom)^T ---------------------
        let a0 = d.vec3f();
        let a1 = d.vec3f();
        let a2 = d.vec3f();
        for (let t = li; t < count; t = t + d.u32(REDUCE)) {
          const p = particles.$[first + t];
          const pr = p.pos - com;
          const rr = p.rest - restCom;
          a0 = a0 + pr * rr.x;
          a1 = a1 + pr * rr.y;
          a2 = a2 + pr * rr.z;
        }
        gsm.$[li * 3 + 0] = d.vec3f(a0);
        gsm.$[li * 3 + 1] = d.vec3f(a1);
        gsm.$[li * 3 + 2] = d.vec3f(a2);
        std.workgroupBarrier();
        reduceGsm(li);

        if (li === d.u32(0) && count > d.u32(0)) {
          const A = d.mat3x3f(gsm.$[0], gsm.$[1], gsm.$[2]);
          bodies.$[b].com = d.vec3f(com);
          bodies.$[b].restCom = d.vec3f(restCom);
          bodies.$[b].rot = d.vec4f(extractRotation(A, bd.rot, d.u32(rotIters)));
        }
      }),
    });

    this.apply = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [PARTICLE_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= d.u32(capacity)) {
          return;
        }
        const b = ownerOf(i);
        if (b >= d.u32(maxBodies)) {
          return;
        }
        const bd = bodies.$[b];
        if ((bd.flags & d.u32(BodyFlags.kinematic)) !== d.u32(0)) {
          return;
        }
        const pr = params.$;
        const p = particles.$[i];
        const target = bd.com + quatRotate(bd.rot, p.rest - bd.restCom);
        let q = std.mix(p.pos, target, std.clamp(bd.stiffness, 0, 1));
        // Shape matching pulls the cloud straight back into whatever the predictor just
        // pushed it out of, so the substep has to end with a projection or it never ends
        // resolved: the body settles where the match pull balances the penetration
        // recovery, which at stiffness 0.3 is most of a particle radius of permanent sink.
        const dw = world.sample(q, 0).x;
        if (dw < pr.radius) {
          const n = world.normal(q);
          const push = pr.radius - dw;
          q = q + n * push;
          particles.$[i].prev = d.vec3f(p.prev + n * push);
        }
        particles.$[i].pos = d.vec3f(q);
        if (bd.plasticity > 0) {
          // Let the rest shape follow the deformation. `plasticity` is per substep, so
          // 1 is putty and 0 is rigid - the same knob Claybook exposed per material.
          const rest = quatRotate(quatConj(bd.rot), q - bd.com) + bd.restCom;
          particles.$[i].rest = d.vec3f(std.mix(p.rest, rest, std.clamp(bd.plasticity, 0, 1)));
        }
      }),
    });

    this.finalize = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [PARTICLE_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= d.u32(capacity)) {
          return;
        }
        const b = ownerOf(i);
        if (b >= d.u32(maxBodies)) {
          return;
        }
        if ((bodies.$[b].flags & d.u32(BodyFlags.kinematic)) !== d.u32(0)) {
          return;
        }
        const pr = params.$;
        const p = particles.$[i];
        // ponytail: first-order velocity extraction, `(p_{n+1} - p_n) / h`, rather than
        // the matching BDF2 form `(3p_{n+1} - 4p_n + p_{n-1}) / 2h` - that one needs a
        // third position slot per particle. The cost is a little numerical damping.
        // Add a `prev2` field if clay ever looks over-damped.
        let v = (p.pos - p.prev) * (1 / std.max(pr.h, 1e-6)) * pr.damping;
        const speed = std.length(v);
        if (speed > pr.maxSpeed) {
          v = v * (pr.maxSpeed / speed);
        }
        particles.$[i].velPrev = d.vec3f(p.vel);
        particles.$[i].vel = d.vec3f(v);
      }),
    });
  }

  /**
   * Sets the uniform body force and one-shot velocity change for the next {@link step}.
   *
   * `accel` is world units per second squared and lasts the frame; `impulse` is a
   * velocity change and is applied exactly once. Only one body can be pushed at a time -
   * these are uniforms, so a second call replaces the first.
   */
  setForce(
    body: number,
    accel: readonly [number, number, number],
    impulse: readonly [number, number, number] = [0, 0, 0],
  ): void {
    this.params.writePartial({
      push: d.vec3f(accel[0], accel[1], accel[2]),
      kick: d.vec3f(impulse[0], impulse[1], impulse[2]),
      pushBody: body,
    });
    this.kicking = impulse[0] !== 0 || impulse[1] !== 0 || impulse[2] !== 0;
  }

  /**
   * Advances every body by {@link dt}, as `substeps` substeps of four dispatches each,
   * into a caller-owned compute pass.
   */
  step(pass: TgpuComputePass): void {
    const bodies = this.set.bodyCount;
    if (bodies === 0) {
      return;
    }
    if (this.kicking) {
      this.bind(this.kick, pass).dispatchWorkgroups(this.particleGroups);
      this.kicking = false;
    }
    for (let s = 0; s < this.substeps; s++) {
      this.bind(this.predict, pass).dispatchWorkgroups(this.particleGroups);
      this.bind(this.match, pass).dispatchWorkgroups(bodies);
      this.bind(this.apply, pass).dispatchWorkgroups(this.particleGroups);
      this.bind(this.finalize, pass).dispatchWorkgroups(this.particleGroups);
    }
  }

  private bind(pipeline: TgpuComputePipeline, pass: TgpuComputePass) {
    let p = pipeline.with(pass);
    for (const g of this.world.groups) {
      p = p.with(g);
    }
    return p;
  }
}
