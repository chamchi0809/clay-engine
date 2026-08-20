import tgpu, { d, std } from 'typegpu';
import type {
  TgpuComputePass,
  TgpuComputePipeline,
  TgpuMutable,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
import type { TracedField } from '../trace/field.ts';
import { Particle, SimCounters, type ParticleSet } from './particles.ts';

/**
 * The box the extractor scans, as a uniform: a body can move, grow or be re-extracted
 * from a morphing field every frame without touching a pipeline.
 */
export const ExtractRegion = d.struct({
  origin: d.vec3f,
  /** World size of one lattice cell. */
  cell: d.f32,
  /**
   * Velocity every emitted particle starts with. Re-extracting a moving body - which
   * is what shape morphing is - otherwise drops it dead in mid-air.
   */
  vel: d.vec3f,
  _pad: d.f32,
});
export type ExtractRegionValue = d.InferInput<typeof ExtractRegion>;

/** Threads in the single-workgroup finish pass. */
const REDUCE_THREADS = 256;
/** Cells per workgroup axis in the two grid passes. */
const CELL_GROUP = 4;

export interface SurfaceExtractorOptions {
  /** Lattice cells per axis. Cost is O(res^3); particle count is O(res^2). */
  resolution?: number;
  /** Body index in the set that the emitted particles belong to. */
  body: number;
  /** First particle index of this body's slice. The caller owns the partitioning. */
  base: number;
  /** Particles this extractor may emit into its slice. */
  capacity: number;
  label?: string;
}

/**
 * Turns the zero level set of any {@link TracedField} into particles plus a triangle
 * list, by naive surface nets on a uniform lattice.
 *
 * This is the join Claybook needs twice (GDC'18 slides 45-48): a clay body's particles
 * *are* its mesh vertices, so one extraction feeds both the solver and the rasteriser,
 * and re-extracting from a blended field every frame is how shape morphing works. It
 * knows nothing about clay: the input is a field and a box.
 *
 * Dual vertex = average of the sign-change points on the 12 edges of a cell, plus one
 * Lipschitz snap onto the surface. Quads come from the lattice *edges*: an edge whose
 * two ends straddle the surface is shared by exactly four cells, and those four dual
 * vertices form one quad, wound by the sign at the edge's low end.
 *
 * ponytail: 8 corner samples per cell, so every corner is sampled 8 times. A workgroup
 * cache would cut that to ~1.2x - do it if extraction ever shows up in a profile.
 */
export class SurfaceExtractor {
  readonly root: TgpuRoot;
  readonly set: ParticleSet;
  readonly field: TracedField;
  readonly resolution: number;
  readonly body: number;
  readonly base: number;
  readonly capacity: number;
  readonly quadCapacity: number;
  readonly region: TgpuUniform<typeof ExtractRegion>;
  /** Particle indices, three per triangle. Read by {@link ParticleMesh}. */
  readonly indices: TgpuMutable<d.WgslArray<d.U32>>;
  /** `vertexCount, instanceCount, firstVertex, firstInstance` for a `drawIndirect`. */
  readonly drawArgsBuffer: GPUBuffer;

  private readonly pipelines: TgpuComputePipeline[];
  private readonly groups: number;

  constructor(set: ParticleSet, field: TracedField, options: SurfaceExtractorOptions) {
    const root = set.root;
    const res = options.resolution ?? 32;
    const { body, base, capacity } = options;
    if (base + capacity > set.capacity) {
      throw new Error(`SurfaceExtractor: slice [${base}, ${base + capacity}) exceeds set capacity`);
    }
    // A closed surface-nets mesh has about as many quads as vertices (V - E + F = 2
    // with E ~ 2V); the margin covers open boundaries clipped by the region.
    const quadCapacity = Math.ceil(capacity * 1.5);
    const label = options.label ?? `extract${body}`;

    this.root = root;
    this.set = set;
    this.field = field;
    this.resolution = res;
    this.body = body;
    this.base = base;
    this.capacity = capacity;
    this.quadCapacity = quadCapacity;
    this.groups = Math.ceil(res / CELL_GROUP);

    const region = root.createUniform(ExtractRegion, {
      origin: [0, 0, 0],
      cell: 0.1,
      vel: [0, 0, 0],
      _pad: 0,
    });
    this.region = region;
    const counters = root.createMutable(SimCounters).$name(`${label}Counters`);
    // +1 biased, so 0 means "this cell has no dual vertex" and no clear pass is needed.
    const idGrid = root.createMutable(d.arrayOf(d.u32, res * res * res)).$name(`${label}Ids`);
    const indices = root
      .createMutable(d.arrayOf(d.u32, quadCapacity * 6))
      .$name(`${label}Indices`);
    this.indices = indices;

    // INDIRECT is not expressible through `createMutable`, so the buffer is made by
    // hand and wrapped - same trick as `TileGrid`.
    this.drawArgsBuffer = root.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      label: `${label}DrawArgs`,
    });
    const drawArgs = root.createMutable(d.arrayOf(d.u32, 4), this.drawArgsBuffer);

    const particles = set.particles;
    const bodies = set.bodies;

    const dist = (p: d.v3f) => {
      'use gpu';
      return field.sample(p, 0).x;
    };
    /** `(crossingPoint * weight, weight)`, weight 1 when the ends straddle the surface. */
    const crossing = (pa: d.v3f, da: number, pb: d.v3f, db: number) => {
      'use gpu';
      if (da * db >= 0) {
        return d.vec4f();
      }
      const t = std.clamp(da / (da - db), 0, 1);
      return d.vec4f(std.mix(pa, pb, t), 1);
    };
    const flat = (c: d.v3u) => {
      'use gpu';
      return c.x + d.u32(res) * (c.y + d.u32(res) * c.z);
    };
    /** Biased id of the dual vertex in cell `c`, or 0. Caller guarantees `c` is in range. */
    const cellId = (c: d.v3u) => {
      'use gpu';
      return idGrid.$[flat(c)];
    };

    const resetPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        std.atomicStore(counters.$.particles, 0);
        std.atomicStore(counters.$.quads, 0);
      }),
    });

    const cellPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [CELL_GROUP, CELL_GROUP, CELL_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (std.any(std.ge(gid, d.vec3u(d.u32(res))))) {
          return;
        }
        const idx = flat(gid);
        const c = region.$.cell;
        const p0 = region.$.origin + d.vec3f(gid) * c;
        const px = p0 + d.vec3f(c, 0, 0);
        const py = p0 + d.vec3f(0, c, 0);
        const pz = p0 + d.vec3f(0, 0, c);
        const pxy = p0 + d.vec3f(c, c, 0);
        const pxz = p0 + d.vec3f(c, 0, c);
        const pyz = p0 + d.vec3f(0, c, c);
        const pxyz = p0 + d.vec3f(c, c, c);
        const d0 = dist(p0);
        const dx = dist(px);
        const dy = dist(py);
        const dz = dist(pz);
        const dxy = dist(pxy);
        const dxz = dist(pxz);
        const dyz = dist(pyz);
        const dxyz = dist(pxyz);

        const acc = crossing(p0, d0, px, dx)
          + crossing(py, dy, pxy, dxy)
          + crossing(pz, dz, pxz, dxz)
          + crossing(pyz, dyz, pxyz, dxyz)
          + crossing(p0, d0, py, dy)
          + crossing(px, dx, pxy, dxy)
          + crossing(pz, dz, pyz, dyz)
          + crossing(pxz, dxz, pxyz, dxyz)
          + crossing(p0, d0, pz, dz)
          + crossing(px, dx, pxz, dxz)
          + crossing(py, dy, pyz, dyz)
          + crossing(pxy, dxy, pxyz, dxyz);

        if (acc.w < 0.5) {
          idGrid.$[idx] = 0;
          return;
        }
        const v = acc.xyz * (1 / acc.w);
        // The average of the edge crossings sits inside the cell but not exactly on the
        // surface; one Lipschitz step lands on it and costs two samples.
        const n = field.normal(v);
        const p = v - n * dist(v);

        const slot = std.atomicAdd(counters.$.particles, 1);
        if (slot >= d.u32(capacity)) {
          idGrid.$[idx] = 0;
          return;
        }
        const id = d.u32(base) + slot;
        particles.$[id] = Particle({
          pos: p,
          body: d.u32(body),
          prev: p,
          material: field.field(p, 0).y,
          vel: region.$.vel,
          _padA: 0,
          velPrev: region.$.vel,
          _padB: 0,
          rest: p,
          _padC: 0,
          restNormal: n,
          _padD: 0,
        });
        idGrid.$[idx] = id + 1;
      }),
    });

    const emit = (a: number, b: number, cc: number, dd: number, flip: boolean) => {
      'use gpu';
      if (a === 0 || b === 0 || cc === 0 || dd === 0) {
        return;
      }
      const q = std.atomicAdd(counters.$.quads, 1);
      if (q >= d.u32(quadCapacity)) {
        return;
      }
      const o = q * 6;
      const i0 = a - 1;
      const i1 = std.select(b, dd, flip) - 1;
      const i2 = cc - 1;
      const i3 = std.select(dd, b, flip) - 1;
      indices.$[o + 0] = i0;
      indices.$[o + 1] = i1;
      indices.$[o + 2] = i2;
      indices.$[o + 3] = i0;
      indices.$[o + 4] = i2;
      indices.$[o + 5] = i3;
    };

    const quadPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [CELL_GROUP, CELL_GROUP, CELL_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (std.any(std.ge(gid, d.vec3u(d.u32(res))))) {
          return;
        }
        // This thread owns lattice point `gid` and the three edges leaving it towards
        // +x/+y/+z. Every edge of the lattice is owned exactly once.
        const c = region.$.cell;
        const l = region.$.origin + d.vec3f(gid) * c;
        const dL = dist(l);
        const inside = dL < 0;
        const i = gid.x;
        const j = gid.y;
        const k = gid.z;
        const hi = d.u32(res) - 1;

        // The four cells sharing an edge, listed counter-clockwise seen from the edge's
        // +direction, so the default winding faces the outside whenever the low end is
        // the inside one.
        if (j >= 1 && k >= 1 && j <= hi && k <= hi && dist(l + d.vec3f(c, 0, 0)) * dL < 0) {
          emit(
            cellId(d.vec3u(i, j - 1, k - 1)),
            cellId(d.vec3u(i, j, k - 1)),
            cellId(d.vec3u(i, j, k)),
            cellId(d.vec3u(i, j - 1, k)),
            !inside,
          );
        }
        if (i >= 1 && k >= 1 && i <= hi && k <= hi && dist(l + d.vec3f(0, c, 0)) * dL < 0) {
          emit(
            cellId(d.vec3u(i - 1, j, k - 1)),
            cellId(d.vec3u(i - 1, j, k)),
            cellId(d.vec3u(i, j, k)),
            cellId(d.vec3u(i, j, k - 1)),
            !inside,
          );
        }
        if (i >= 1 && j >= 1 && i <= hi && j <= hi && dist(l + d.vec3f(0, 0, c)) * dL < 0) {
          emit(
            cellId(d.vec3u(i - 1, j - 1, k)),
            cellId(d.vec3u(i, j - 1, k)),
            cellId(d.vec3u(i, j, k)),
            cellId(d.vec3u(i - 1, j, k)),
            !inside,
          );
        }
      }),
    });

    const gsm = tgpu.workgroupVar(d.arrayOf(d.vec3f, REDUCE_THREADS));
    const finishPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [REDUCE_THREADS],
        in: { lid: d.builtin.localInvocationId },
      })(({ lid }) => {
        'use gpu';
        const li = lid.x;
        const n = std.min(std.atomicLoad(counters.$.particles), d.u32(capacity));
        let sum = d.vec3f();
        for (let t = li; t < n; t = t + d.u32(REDUCE_THREADS)) {
          sum = sum + particles.$[d.u32(base) + t].rest;
        }
        gsm.$[li] = d.vec3f(sum);
        std.workgroupBarrier();
        for (let s = d.u32(REDUCE_THREADS / 2); s > d.u32(0); s = s >>> 1) {
          if (li < s) {
            gsm.$[li] = d.vec3f(gsm.$[li] + gsm.$[li + s]);
          }
          std.workgroupBarrier();
        }
        if (li === d.u32(0)) {
          const com = gsm.$[0] * (1 / std.max(d.f32(n), 1));
          bodies.$[body].first = d.u32(base);
          bodies.$[body].count = n;
          bodies.$[body].restCom = d.vec3f(com);
          bodies.$[body].com = d.vec3f(com);
          bodies.$[body].rot = d.vec4f(0, 0, 0, 1);
          const q = std.min(std.atomicLoad(counters.$.quads), d.u32(quadCapacity));
          drawArgs.$[0] = q * 6;
          drawArgs.$[1] = 1;
          drawArgs.$[2] = 0;
          drawArgs.$[3] = 0;
        }
      }),
    });

    this.pipelines = [resetPipeline, cellPipeline, quadPipeline, finishPipeline];
  }

  /** Moves the scanned box. `cell` is the lattice spacing, so `res * cell` is its size. */
  setRegion(
    origin: readonly [number, number, number],
    size: number,
    vel: readonly [number, number, number] = [0, 0, 0],
  ): void {
    this.region.write({
      origin: [origin[0], origin[1], origin[2]],
      cell: size / this.resolution,
      vel: [vel[0], vel[1], vel[2]],
      _pad: 0,
    });
  }

  /**
   * Re-extracts the whole body. Four dispatches into a caller-owned pass; they are
   * ordered, and WebGPU makes each one's writes visible to the next.
   */
  extract(pass: TgpuComputePass): void {
    const [reset, cells, quads, finish] = this.pipelines;
    this.bind(reset!, pass).dispatchWorkgroups(1);
    this.bind(cells!, pass).dispatchWorkgroups(this.groups, this.groups, this.groups);
    this.bind(quads!, pass).dispatchWorkgroups(this.groups, this.groups, this.groups);
    this.bind(finish!, pass).dispatchWorkgroups(1);
  }

  private bind(pipeline: TgpuComputePipeline, pass: TgpuComputePass) {
    let p = pipeline.with(pass);
    for (const g of this.field.groups) {
      p = p.with(g);
    }
    return p;
  }
}
