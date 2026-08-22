import tgpu, { d, std } from 'typegpu';
import type { TgpuComputePipeline, TgpuReadonly, TgpuRoot, TgpuUniform } from 'typegpu';
import { atlasWriteLayout, type BrushAtlas } from './atlas.ts';
import type { NormalizedMesh } from '../shape/mesh.ts';

/** One triangle in the bake box's `[-1, 1]^3` space. */
export const Tri = d.struct({
  a: d.vec3f,
  b: d.vec3f,
  c: d.vec3f,
});

export const BakeParams = d.struct({
  triCount: d.u32,
  /** Where this slot starts along the atlas's Z axis, in voxels. */
  slotZ: d.u32,
  /** {@link NormalizedMesh.shell}: half the surface thickness, in box units. */
  shell: d.f32,
});

/** Voxels per workgroup axis. 4^3 = 64 threads, one voxel each. */
const BAKE_GROUP = 4;

const dot2 = tgpu.fn([d.vec3f], d.f32)((v) => {
  'use gpu';
  return std.dot(v, v);
});

/**
 * Unsigned distance from a point to a triangle (Quilez).
 *
 * The three edge tests decide whether the foot of the perpendicular lands inside the
 * triangle. If it does, the answer is the plane distance; if not, it is the nearest of the
 * three segment distances. A degenerate triangle has a zero normal, all three signs come
 * out zero, and it falls into the segment branch - which is the right answer for a sliver.
 */
const udTriangle = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.vec3f], d.f32)((p, a, b, c) => {
  'use gpu';
  const ba = b - a;
  const pa = p - a;
  const cb = c - b;
  const pb = p - b;
  const ac = a - c;
  const pc = p - c;
  const nor = std.cross(ba, ac);
  const inside = std.sign(std.dot(std.cross(ba, nor), pa))
    + std.sign(std.dot(std.cross(cb, nor), pb))
    + std.sign(std.dot(std.cross(ac, nor), pc));
  if (inside < 2) {
    return std.sqrt(
      std.min(
        std.min(
          dot2(ba * std.clamp(std.dot(ba, pa) / dot2(ba), 0, 1) - pa),
          dot2(cb * std.clamp(std.dot(cb, pb) / dot2(cb), 0, 1) - pb),
        ),
        dot2(ac * std.clamp(std.dot(ac, pc) / dot2(ac), 0, 1) - pc),
      ),
    );
  }
  const plane = std.dot(nor, pa);
  return std.sqrt(plane * plane / std.max(dot2(nor), 1e-20));
});

/**
 * Signed solid angle a triangle subtends at `p` (Van Oosterom & Strackee 1983).
 *
 * Summed over a mesh and divided by `4*pi` this is the *generalised winding number*
 * (Jacobson et al. 2013), and it is why the baker does not care whether a mesh is
 * watertight. Ray parity - counting crossings along a line - needs a closed surface and
 * gives a plainly wrong answer for the open, self-intersecting, duplicated-face geometry
 * that real art assets are made of. The winding number degrades gracefully instead: near a
 * hole it slides continuously between inside and outside rather than flipping a whole
 * scanline.
 */
const solidAngle = tgpu.fn([d.vec3f, d.vec3f, d.vec3f, d.vec3f], d.f32)((p, a, b, c) => {
  'use gpu';
  const qa = a - p;
  const qb = b - p;
  const qc = c - p;
  const la = std.length(qa);
  const lb = std.length(qb);
  const lc = std.length(qc);
  const num = std.dot(qa, std.cross(qb, qc));
  const den = la * lb * lc
    + std.dot(qa, qb) * lc
    + std.dot(qb, qc) * la
    + std.dot(qc, qa) * lb;
  // `p` sitting exactly on a vertex zeroes both, and `atan2(0, 0)` is undefined. The voxel
  // centre is offset by half a voxel, so this is a measure-zero case, but a NaN here would
  // poison the whole slot.
  if (std.abs(num) + std.abs(den) < 1e-20) {
    return 0;
  }
  return 2 * std.atan2(num, den);
});

export interface MeshBakerOptions {
  /**
   * Triangle capacity. The buffer is allocated once and reused by every bake, at 48 bytes
   * a triangle, so the default costs 3 MiB and covers anything worth baking into a 48^3
   * slot. A mesh past this throws rather than being silently decimated.
   */
  maxTriangles?: number;
}

/**
 * Bakes a triangle mesh into a {@link BrushAtlas} slot, on the GPU.
 *
 * Brute force: one thread per voxel, every triangle tested. That is `res^3 * tris` work -
 * about half a billion triangle tests for a 5k-triangle mesh at 48^3 - which sounds
 * ruinous and takes a few milliseconds, because every thread in a workgroup reads the same
 * triangle on the same cycle and the whole loop runs out of cache. It is a load-time cost
 * paid once per asset.
 *
 * ponytail: the way past brute force is a narrow band - only voxels near a triangle's
 * bounding box need an exact distance, and the rest can be swept (Bridson's
 * `makelevelset3`) - but that trades the winding number's robustness for a sweep that
 * assumes a closed surface, and needs a BVH to be worth it. Revisit if bake time ever
 * shows up in a load screen.
 */
export class MeshBaker {
  readonly root: TgpuRoot;
  readonly atlas: BrushAtlas;
  readonly maxTriangles: number;

  private readonly tris: TgpuReadonly<d.WgslArray<typeof Tri>>;
  private readonly params: TgpuUniform<typeof BakeParams>;
  private readonly pipeline: TgpuComputePipeline;

  constructor(root: TgpuRoot, atlas: BrushAtlas, options: MeshBakerOptions = {}) {
    this.root = root;
    this.atlas = atlas;
    this.maxTriangles = options.maxTriangles ?? 65536;

    const tris = root
      .createReadonly(d.arrayOf(Tri, this.maxTriangles))
      .$name('bakeTriangles');
    const params = root.createUniform(BakeParams, { triCount: 0, slotZ: 0, shell: 0 });
    this.tris = tris;
    this.params = params;

    // A JS constant, not a uniform: the resolution is fixed when the atlas is created, and
    // baking it in lets the voxel-centre arithmetic fold at compile time.
    const res = atlas.resolution;

    this.pipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [BAKE_GROUP, BAKE_GROUP, BAKE_GROUP],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x >= d.u32(res) || gid.y >= d.u32(res) || gid.z >= d.u32(res)) {
          return;
        }
        // Voxel centres, so the outermost sample sits half a voxel inside the wall -
        // exactly where the atlas's half-texel inset reads it back from.
        const q = (d.vec3f(gid) + d.vec3f(0.5)) * (2 / res) - d.vec3f(1);
        let dist = d.f32(1e9);
        let omega = d.f32(0);
        for (let i = d.u32(0); i < params.$.triCount; i++) {
          const t = tris.$[i];
          dist = std.min(dist, udTriangle(q, t.a, t.b, t.c));
          omega = omega + solidAngle(q, t.a, t.b, t.c);
        }
        // Winding number over 0.5, i.e. solid angle over 2*pi. Taken absolute so a mesh
        // wound the other way round - an exporter's choice, invisible until it bakes as
        // nothing at all - comes out solid instead of empty.
        const inside = std.abs(omega) > 2 * Math.PI;
        // The solid, unioned with a slab of half-thickness `shell` either side of the
        // surface. At `shell = 0` the union is a no-op and this is the plain signed
        // distance. Above it, an open surface - which encloses nothing, so `inside` is
        // false everywhere - comes out as exactly the slab, and a mesh that is *nearly*
        // closed keeps the interior the winding number did find instead of hollowing out
        // around its own holes.
        const solid = std.select(dist, -dist, inside);
        std.textureStore(
          atlasWriteLayout.$.out,
          d.vec3u(gid.x, gid.y, gid.z + params.$.slotZ),
          d.vec4f(std.min(solid, dist - params.$.shell), 0, 0, 1),
        );
      }),
    });
  }

  /**
   * Bakes `mesh` into `slot` and resolves once the GPU is done with it.
   *
   * Awaited rather than queued into the frame's compute pass because a bake is a load-time
   * operation an order of magnitude longer than a frame, and a caller that has awaited it
   * knows the slot is readable - a brush pointing at a slot that has not been written yet
   * samples whatever the texture happened to be cleared to.
   */
  async bake(slot: number, mesh: NormalizedMesh): Promise<void> {
    if (slot < 0 || slot >= this.atlas.slots) {
      throw new Error(`MeshBaker: slot ${slot} is outside the atlas's 0..${this.atlas.slots - 1}`);
    }
    if (mesh.triangleCount > this.maxTriangles) {
      throw new Error(
        `MeshBaker: ${mesh.triangleCount} triangles exceeds the baker's capacity of `
          + `${this.maxTriangles}. Decimate the mesh, or raise \`maxTriangles\`.`,
      );
    }
    const flat = mesh.triangles;
    const packed: d.Infer<typeof Tri>[] = [];
    for (let t = 0; t < mesh.triangleCount; t++) {
      const o = t * 9;
      packed.push({
        a: d.vec3f(flat[o]!, flat[o + 1]!, flat[o + 2]!),
        b: d.vec3f(flat[o + 3]!, flat[o + 4]!, flat[o + 5]!),
        c: d.vec3f(flat[o + 6]!, flat[o + 7]!, flat[o + 8]!),
      });
    }
    this.tris.write(packed);
    this.params.write({
      triCount: mesh.triangleCount,
      slotZ: slot * this.atlas.resolution,
      shell: mesh.shell,
    });

    const groups = Math.ceil(this.atlas.resolution / BAKE_GROUP);
    this.pipeline.with(this.atlas.writeGroup).dispatchWorkgroups(groups, groups, groups);
    await this.root.device.queue.onSubmittedWorkDone();
  }
}
