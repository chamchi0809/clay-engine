import tgpu, { d, std } from 'typegpu';
import type {
  TgpuComputePass,
  TgpuComputePipeline,
  TgpuMutable,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
import { hash3 } from '../math/gpu.ts';
import { TILE } from '../field/volume.ts';
import { SplatField } from './splat.ts';
import type { TracedField } from '../trace/field.ts';

/**
 * One SPH particle. `nvel` is next substep's velocity, staged by the force pass and
 * consumed by the integrator: without it one pass would write `pos`/`vel` while its
 * neighbours are still reading them, and a neighbour pair seeing each other at
 * different times exchanges asymmetric forces, which injects energy every step.
 */
export const FluidParticle = d.struct({
  pos: d.vec3f,
  density: d.f32,
  vel: d.vec3f,
  pressure: d.f32,
  nvel: d.vec3f,
});

export const FluidParams = d.struct({
  gravity: d.vec3f,
  /** Substep length, seconds. */
  dt: d.f32,
  emitPos: d.vec3f,
  emitRadius: d.f32,
  emitVel: d.vec3f,
  /** Speed cap. */
  maxSpeed: d.f32,
  /** Pressure stiffness, `p = k (rho - rho0)`. */
  stiffness: d.f32,
  /** XSPH blend towards the neighbourhood mean velocity, `0..1`. */
  viscosity: d.f32,
  /** Velocity retained per substep. */
  damping: d.f32,
  /** Live particles, capped at capacity. */
  count: d.u32,
  /** Ring-buffer write cursor for this frame's emission. */
  emitBase: d.u32,
  emitCount: d.u32,
  emitSeed: d.u32,
  /** How deep into the world surface a particle must be to be reported as a contact. */
  contactDepth: d.f32,
});
export type FluidParamsValue = d.InferInput<typeof FluidParams>;

/** Fixed-size list of fluid/world contact points, for erosion. */
export const ContactList = d.struct({
  count: d.atomic(d.u32),
  points: d.arrayOf(d.vec4f, 32),
});

export interface FluidSimOptions {
  /** Particle capacity. Emission wraps, so a tap can run forever. */
  capacity?: number;
  /** Rest spacing between particles. Sets the mass and the rest density. */
  spacing?: number;
  /** SPH smoothing radius. Must be at least twice `spacing` for a usable neighbourhood. */
  smoothing?: number;
  /** Neighbours recorded per hash cell. */
  bucket?: number;
  /** Collision radius against the world field. */
  radius?: number;
  /** Radius of the sphere each particle contributes to the rendered surface. */
  surfaceRadius?: number;
  /** Material id written into the baked volume. */
  material?: number;
  /** Resolution of the baked fluid volume. Must suit `TILE * 2^(mipLevels-1)`. */
  bakeResolution?: number;
  bakeMipLevels?: number;
  /** Band half-width of the baked volume, in voxels. Directly sets the splat footprint. */
  bakeBand?: number;
  /** World-space region the bake covers. Should match the world volume. */
  origin?: readonly [number, number, number];
  worldSize?: number;
  substeps?: number;
  gravity?: readonly [number, number, number];
  stiffness?: number;
  /** XSPH velocity blend per substep, `0..1`. Claybook's clay-like water wants ~0.4. */
  viscosity?: number;
  damping?: number;
  maxSpeed?: number;
  label?: string;
}

/**
 * High-viscosity SPH fluid that collides against a signed distance field and renders
 * itself as one. Claybook GDC'18 slides 55-62.
 *
 * Three parts, and only the middle one is fluid-specific:
 *
 *  1. **Simulation.** Weakly compressible SPH over a fixed-capacity spatial hash:
 *     insert, density/pressure, forces, integrate, collide against the world field.
 *     Collision is again a single field lookup per particle - no contact generation.
 *     Viscosity is XSPH velocity blending rather than an explicit Laplacian, because
 *     only the former is stable at a game's timestep - see the force pass.
 *  2. **Bake.** Delegated wholesale to {@link SplatField}, which is not fluid code -
 *     it turns any particle cloud into a traceable volume, and a clay body uses the
 *     very same class.
 *  3. **Render.** {@link FluidSim.field} is a plain {@link TracedField}. `unionField`
 *     it with the world and the existing renderer draws, shadows and AOs the fluid with
 *     no new code at all.
 */
export class FluidSim {
  readonly root: TgpuRoot;
  readonly world: TracedField;
  readonly capacity: number;
  readonly params: TgpuUniform<typeof FluidParams>;
  readonly particles: TgpuMutable<d.WgslArray<typeof FluidParticle>>;
  readonly contacts: TgpuMutable<typeof ContactList>;
  /** The bake. `surface.field` is what the tracer and other colliders consume. */
  readonly surface: SplatField;
  readonly field: TracedField;
  readonly substeps: number;
  /** Rest spacing, as passed in. Sets the default emission radius. */
  readonly spacing: number;

  private readonly clearGrid: TgpuComputePipeline;
  private readonly insert: TgpuComputePipeline;
  private readonly density: TgpuComputePipeline;
  private readonly forces: TgpuComputePipeline;
  private readonly integrate: TgpuComputePipeline;
  private readonly emit: TgpuComputePipeline;
  private readonly particleGroups: number;
  private readonly gridGroups: number;

  private live = 0;
  private cursor = 0;
  private pendingEmit = 0;
  private seed = 1;
  private frozen: FluidParamsValue;

  constructor(root: TgpuRoot, world: TracedField, options: FluidSimOptions = {}) {
    const capacity = options.capacity ?? 8192;
    const spacing = options.spacing ?? 0.45;
    const h = options.smoothing ?? spacing * 2.2;
    const bucket = options.bucket ?? 64;
    const radius = options.radius ?? spacing * 0.5;
    const surfR = options.surfaceRadius ?? spacing * 1.1;
    const material = options.material ?? 0;
    const bakeRes = options.bakeResolution ?? 64;
    const bakeMips = options.bakeMipLevels ?? Math.max(1, Math.log2(bakeRes / TILE) | 0);
    const band = options.bakeBand ?? 2;
    const label = options.label ?? 'fluid';

    this.root = root;
    this.world = world;
    this.capacity = capacity;
    this.spacing = spacing;
    // 3, not 2: the pressure sound speed is `sqrt(stiffness)`, so the CFL limit is
    // `0.25 h / sqrt(k)` ~ 1/79 s at the defaults. 1/120 s leaves no margin for a
    // compressed pile; 1/180 does.
    this.substeps = options.substeps ?? 3;
    this.particleGroups = Math.ceil(capacity / 64);


    // --- rest density from the actual lattice, not a magic number -----------
    // A particle in an infinite simple-cubic lattice of `spacing` sees exactly this
    // much density, so `rho - rho0` is zero at rest by construction and the stiffness
    // knob only has to fight compression.
    const poly6 = 315 / (64 * Math.PI * h ** 9);
    const spikyGrad = -45 / (Math.PI * h ** 6);
    const mass = spacing ** 3;
    let restDensity = 0;
    const reach = Math.ceil(h / spacing);
    for (let z = -reach; z <= reach; z++) {
      for (let y = -reach; y <= reach; y++) {
        for (let x = -reach; x <= reach; x++) {
          const r2 = (x * x + y * y + z * z) * spacing * spacing;
          if (r2 < h * h) {
            restDensity += mass * poly6 * (h * h - r2) ** 3;
          }
        }
      }
    }

    const g = options.gravity ?? [0, -9.81, 0];
    this.frozen = {
      gravity: [g[0], g[1], g[2]],
      dt: 1 / 60 / this.substeps,
      emitPos: [0, 0, 0],
      emitRadius: spacing,
      emitVel: [0, 0, 0],
      // Capped below the sound speed: a particle may not cross more than half a
      // smoothing radius in one substep or the neighbour search misses the collision it
      // is about to have.
      maxSpeed: options.maxSpeed ?? 15,
      stiffness: options.stiffness ?? 220,
      viscosity: options.viscosity ?? 0.4,
      damping: options.damping ?? 0.995,
      count: 0,
      emitBase: 0,
      emitCount: 0,
      emitSeed: 1,
      contactDepth: radius * 1.5,
    };
    const params = root.createUniform(FluidParams, this.frozen);
    this.params = params;

    const particles = root
      .createMutable(d.arrayOf(FluidParticle, capacity))
      .$name(`${label}Particles`);
    this.particles = particles;
    this.contacts = root.createMutable(ContactList).$name(`${label}Contacts`);
    const contacts = this.contacts;

    // The renderable surface. Nothing below this line is aware of how it is built.
    this.surface = new SplatField(root, {
      capacity,
      positionAt: (i: number) => {
        'use gpu';
        return d.vec3f(particles.$[i].pos);
      },
      liveAt: (i: number) => {
        'use gpu';
        return i < params.$.count;
      },
    }, {
      radius: surfR,
      material,
      resolution: bakeRes,
      mipLevels: bakeMips,
      band,
      worldSize: options.worldSize ?? 24,
      origin: options.origin ?? [-12, -4, -12],
      label,
    });
    this.field = this.surface.field;

    // --- spatial hash ------------------------------------------------------
    // Cell size is exactly `h`, which is what makes a 3x3x3 scan complete.
    const gridRes = Math.ceil(this.surface.volume.worldSize / h) + 2;
    const gridCells = gridRes ** 3;
    // One cell of margin on *both* sides: a particle that leaves the volume must still
    // land in a real cell, or it is dropped from the hash and free-falls with no
    // pressure or viscosity while its neighbours still see it.
    const gridOrigin = this.surface.volume.origin.map((c) => c - h) as [number, number, number];
    this.gridGroups = Math.ceil(gridCells / 64);
    const counts = root
      .createMutable(d.arrayOf(d.atomic(d.u32), gridCells))
      .$name(`${label}CellCounts`);
    const items = root
      .createMutable(d.arrayOf(d.u32, gridCells * bucket))
      .$name(`${label}CellItems`);

    const cellOf = (p: d.v3f) => {
      'use gpu';
      const q = std.floor((p - d.vec3f(gridOrigin[0], gridOrigin[1], gridOrigin[2])) * (1 / h));
      return d.vec3i(q);
    };
    const cellIndex = (c: d.v3i) => {
      'use gpu';
      const u = d.vec3u(std.clamp(c, d.vec3i(0), d.vec3i(d.i32(gridRes - 1))));
      return u.x + d.u32(gridRes) * (u.y + d.u32(gridRes) * u.z);
    };
    const inGrid = (c: d.v3i) => {
      'use gpu';
      return std.all(std.ge(c, d.vec3i(0))) && std.all(std.lt(c, d.vec3i(d.i32(gridRes))));
    };

    this.emit = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.emitCount) {
          return;
        }
        const slot = (params.$.emitBase + i) % d.u32(capacity);
        const r = hash3(params.$.emitSeed * d.u32(9781) + i);
        // Emitting on a jittered sphere rather than a lattice: a lattice makes the
        // first density pass see a perfect rest state and the jet comes out as a rod.
        const dir = std.normalize(r * 2 - d.vec3f(1) + d.vec3f(1e-4));
        particles.$[slot] = FluidParticle({
          pos: params.$.emitPos + dir * (params.$.emitRadius * std.pow(r.x, 1 / 3)),
          density: 0,
          vel: params.$.emitVel,
          pressure: 0,
          nvel: params.$.emitVel,
        });
      }),
    });

    this.clearGrid = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x < d.u32(gridCells)) {
          std.atomicStore(counts.$[gid.x], 0);
        }
        if (gid.x === d.u32(0)) {
          std.atomicStore(contacts.$.count, 0);
        }
      }),
    });

    this.insert = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.count) {
          return;
        }
        const c = cellOf(particles.$[i].pos);
        if (!inGrid(c)) {
          return;
        }
        const cell = cellIndex(c);
        const slot = std.atomicAdd(counts.$[cell], 1);
        // ponytail: fixed bucket, overflow drops the neighbour link. A dropped link
        // only weakens one pair's pressure term, so the fluid stays stable; raise
        // `bucket` if a compressed pile visibly interpenetrates.
        if (slot < d.u32(bucket)) {
          items.$[cell * d.u32(bucket) + slot] = i;
        }
      }),
    });

    this.density = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.count) {
          return;
        }
        const pi = particles.$[i].pos;
        const base = cellOf(pi);
        let rho = d.f32(0);
        for (const dz of std.range(-1, 2)) {
          for (const dy of std.range(-1, 2)) {
            for (const dx of std.range(-1, 2)) {
              const c = base + d.vec3i(dx, dy, dz);
              if (!inGrid(c)) {
                continue;
              }
              const cell = cellIndex(c);
              const n = std.min(std.atomicLoad(counts.$[cell]), d.u32(bucket));
              for (let k = d.u32(0); k < n; k++) {
                const j = items.$[cell * d.u32(bucket) + k];
                const r2 = std.dot(pi - particles.$[j].pos, pi - particles.$[j].pos);
                if (r2 < h * h) {
                  rho = rho + mass * poly6 * std.pow(h * h - r2, 3);
                }
              }
            }
          }
        }
        particles.$[i].density = std.max(rho, restDensity * 0.1);
        particles.$[i].pressure = std.max(params.$.stiffness * (rho - restDensity), 0);
      }),
    });

    this.forces = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.count) {
          return;
        }
        const me = particles.$[i];
        const base = cellOf(me.pos);
        let fPress = d.vec3f();
        // XSPH velocity smoothing, seeded with the particle's own kernel weight so a
        // lone particle keeps its velocity. This is a *convex combination* of the
        // neighbourhood velocities, so it can never add energy no matter how large the
        // blend or the timestep.
        //
        // It replaces an explicit viscosity Laplacian, which cannot survive here: that
        // term's stability limit is `dt * nu * sum_j m/rho_j * viscLap * (h - r) < 2`,
        // and at Claybook-grade viscosity with a 1/120 s substep the sum is ~700, i.e.
        // six times over. That is what made the tap detonate and spray particles across
        // the whole volume instead of pooling.
        let vSum = me.vel * d.f32(h ** 6);
        let wSum = d.f32(h ** 6);
        for (const dz of std.range(-1, 2)) {
          for (const dy of std.range(-1, 2)) {
            for (const dx of std.range(-1, 2)) {
              const c = base + d.vec3i(dx, dy, dz);
              if (!inGrid(c)) {
                continue;
              }
              const cell = cellIndex(c);
              const n = std.min(std.atomicLoad(counts.$[cell]), d.u32(bucket));
              for (let k = d.u32(0); k < n; k++) {
                const j = items.$[cell * d.u32(bucket) + k];
                if (j === i) {
                  continue;
                }
                const other = particles.$[j];
                const dv = me.pos - other.pos;
                const r2 = std.dot(dv, dv);
                if (r2 >= h * h) {
                  continue;
                }
                const inv = 1 / other.density;
                const w = std.pow(h * h - r2, 3);
                vSum = vSum + other.vel * w;
                wSum = wSum + w;
                const r = std.sqrt(r2);
                if (r > 1e-6) {
                  fPress = fPress
                    - dv * (1 / r)
                      * (mass * (me.pressure + other.pressure) * 0.5 * inv * spikyGrad
                        * ((h - r) * (h - r)));
                }
              }
            }
          }
        }
        const acc = fPress * (1 / me.density) + params.$.gravity;
        let v = std.mix(me.vel, vSum * (1 / wSum), std.clamp(params.$.viscosity, 0, 1));
        v = (v + acc * params.$.dt) * params.$.damping;
        const speed = std.length(v);
        if (speed > params.$.maxSpeed) {
          v = v * (params.$.maxSpeed / speed);
        }
        particles.$[i].nvel = d.vec3f(v);
      }),
    });

    // --- integrate: advance, collide, contain -------------------------------
    // Separate from the force pass so nothing writes `pos`/`vel` while a neighbour is
    // still reading them, and so containment runs exactly once per substep.
    const [lox, loy, loz] = this.surface.volume.origin;
    const wsz = this.surface.volume.worldSize;
    const wallLo = d.vec3f(lox + radius, loy + radius, loz + radius);
    const wallHi = d.vec3f(lox + wsz - radius, loy + wsz - radius, loz + wsz - radius);

    this.integrate = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.count) {
          return;
        }
        // Copies, not references: a storage reference cannot be reassigned.
        let v = d.vec3f(particles.$[i].nvel);
        let p = particles.$[i].pos + v * params.$.dt;

        // World collision: push out along the gradient, kill the inward velocity.
        const dw = world.sample(p, 0).x;
        if (dw < radius) {
          const nrm = world.normal(p);
          p = p + nrm * (radius - dw);
          v = v - nrm * std.min(std.dot(v, nrm), 0);
          if (dw < params.$.contactDepth) {
            // A short, unordered sample of where the fluid is grinding on the world.
            // Read back asynchronously and turned into `cut` edits by the game - that
            // is Claybook's fluid erosion, with the policy left to the caller.
            const slot = std.atomicAdd(contacts.$.count, 1);
            if (slot < d.u32(32)) {
              contacts.$.points[slot] = d.vec4f(p, std.length(v));
            }
          }
        }

        // The baked volume is the only place this fluid exists to the renderer and the
        // hash grid is only as wide as that volume, so a particle outside it is
        // invisible, force-free and permanent. Walls, not a kill list: a level's water
        // budget stays constant and the ring buffer keeps its meaning.
        const cl = std.clamp(p, wallLo, wallHi);
        // Whichever components the clamp moved were outside, so they lose their velocity
        // and the particle slides along the wall instead of hammering it.
        const inside = std.eq(cl, p);
        particles.$[i].vel = std.select(d.vec3f(), v, inside);
        particles.$[i].pos = d.vec3f(cl);
      }),
    });

  }

  /** Live particle count. */
  get particleCount(): number {
    return this.live;
  }

  /**
   * Queues `n` particles for the next {@link step}, from a jittered ball whose radius
   * defaults to whatever holds `n` of them at rest spacing.
   *
   * A tap has a throughput limit and it is easy to blow past by two orders of
   * magnitude: a nozzle of radius `r` emptying at speed `v` carries away only
   * `pi r^2 v / spacing^3` particles per second (~80/s at the demo's numbers).
   * Emit faster and each batch lands inside the previous one, local density goes to
   * several times rest, and the pressure term answers with a firehose. Throttle the
   * caller, do not stiffen the solver.
   */
  spawn(
    n: number,
    pos: readonly [number, number, number],
    vel: readonly [number, number, number],
    spread = this.spacing * Math.cbrt(Math.max(n, 1)) * 0.62,
  ): void {
    this.pendingEmit = Math.min(n, this.capacity);
    this.frozen = {
      ...this.frozen,
      emitPos: [pos[0], pos[1], pos[2]],
      emitVel: [vel[0], vel[1], vel[2]],
      emitRadius: spread,
      emitSeed: this.seed++,
    };
  }

  /** Live tuning of anything in {@link FluidParams} except the emission cursor. */
  configure(patch: Partial<Omit<FluidParamsValue, 'count' | 'emitBase' | 'emitCount' | 'emitSeed'>>): void {
    this.frozen = { ...this.frozen, ...patch };
  }

  /**
   * One frame: emit, simulate `substeps` substeps, bake every mip level. All into a
   * caller-owned compute pass.
   */
  step(pass: TgpuComputePass): void {
    const emitCount = this.pendingEmit;
    this.pendingEmit = 0;
    const emitBase = this.cursor;
    if (emitCount > 0) {
      this.cursor = (emitBase + emitCount) % this.capacity;
      // The ring only ever grows the live prefix; once it wraps, everything is live.
      this.live = Math.min(this.capacity, Math.max(this.live, emitBase + emitCount));
    }
    this.params.write({ ...this.frozen, count: this.live, emitBase, emitCount });

    if (emitCount > 0) {
      this.bind(this.emit, pass).dispatchWorkgroups(Math.ceil(emitCount / 64));
    }
    for (let s = 0; s < this.substeps && this.live > 0; s++) {
      this.clearGrid.with(pass).dispatchWorkgroups(this.gridGroups);
      this.bind(this.insert, pass).dispatchWorkgroups(this.particleGroups);
      this.bind(this.density, pass).dispatchWorkgroups(this.particleGroups);
      this.bind(this.forces, pass).dispatchWorkgroups(this.particleGroups);
      this.bind(this.integrate, pass).dispatchWorkgroups(this.particleGroups);
    }
    this.surface.bake(pass);
  }

  /**
   * Where the fluid is currently grinding against the world, as `[x, y, z, speed]`.
   * A GPU readback, so `await` it across frames rather than per frame - erosion is
   * cumulative and does not care about a frame of latency.
   */
  async readContacts(): Promise<[number, number, number, number][]> {
    const v = await this.contacts.read();
    const n = Math.min(v.count as unknown as number, 32);
    return v.points.slice(0, n).map((p) => [p.x, p.y, p.z, p.w]);
  }

  destroy(): void {
    this.surface.destroy();
  }

  private bind(pipeline: TgpuComputePipeline, pass: TgpuComputePass) {
    let p = pipeline.with(pass);
    for (const g of this.world.groups) {
      p = p.with(g);
    }
    return p;
  }
}
