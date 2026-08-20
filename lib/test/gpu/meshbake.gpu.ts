import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { BrushSet } from '../../src/field/brush.ts';
import { BrushAtlas } from '../../src/field/atlas.ts';
import { MeshBaker } from '../../src/field/meshbake.ts';
import { SdfBuilder } from '../../src/field/builder.ts';
import { SdfVolume } from '../../src/field/volume.ts';
import { volumeField, type TracedField } from '../../src/trace/field.ts';
import { normalizeMesh, parseObj, type BakedMesh, type MeshData } from '../../src/shape/mesh.ts';
import { compileShape, sdf, shapeBounds } from '../../src/shape/sdf.ts';
import { evalWith, near } from './evalbrush.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * The mesh baker, end to end: triangles in, brush distances out.
 *
 * Every assertion goes through `BrushSet.evalBrush` on a `volume` brush rather than reading
 * the texture back, because that is the only path the engine ever uses. It covers the whole
 * chain at once - normalisation, the winding-number sign, the atlas's slot addressing and
 * half-texel inset, and the brush's multiply-back-by-half-extent - and a mistake anywhere in
 * it lands on the same number.
 *
 * The reference is a box's closed form. A bake is a sampled field, so it cannot be exact;
 * the points are picked on-axis, away from the diagonals where the box distance has a kink,
 * because there the field is linear and trilinear reconstruction is exact up to f16.
 */
let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

/** A cube spanning [-1, 1]^3, wound outward. */
const CUBE_POSITIONS = [
  -1, -1, -1, /* 0 */ 1, -1, -1, /* 1 */ 1, 1, -1, /* 2 */ -1, 1, -1, /* 3 */
  -1, -1, 1, /* 4 */ 1, -1, 1, /* 5 */ 1, 1, 1, /* 6 */ -1, 1, 1, /* 7 */
];
const CUBE_FACES: Record<string, [number, number, number, number, number, number]> = {
  negZ: [0, 3, 2, 0, 2, 1],
  posZ: [4, 5, 6, 4, 6, 7],
  negY: [0, 1, 5, 0, 5, 4],
  posY: [3, 7, 6, 3, 6, 2],
  negX: [0, 4, 7, 0, 7, 3],
  posX: [1, 2, 6, 1, 6, 5],
};

function cube(opts: { skip?: string; flip?: boolean } = {}): MeshData {
  const indices: number[] = [];
  for (const [face, tris] of Object.entries(CUBE_FACES)) {
    if (face === opts.skip) {
      continue;
    }
    for (let t = 0; t < 2; t++) {
      const [a, b, c] = [tris[t * 3]!, tris[t * 3 + 1]!, tris[t * 3 + 2]!];
      indices.push(...(opts.flip ? [a, c, b] : [a, b, c]));
    }
  }
  return { positions: new Float32Array(CUBE_POSITIONS), indices: new Uint32Array(indices) };
}

/** Closed form for a box of half-extent `h` centred on the origin. */
function sdBox(p: readonly [number, number, number], h: number): number {
  const q = [Math.abs(p[0]) - h, Math.abs(p[1]) - h, Math.abs(p[2]) - h];
  const outside = Math.hypot(Math.max(q[0]!, 0), Math.max(q[1]!, 0), Math.max(q[2]!, 0));
  return outside + Math.min(Math.max(q[0]!, q[1]!, q[2]!), 0);
}

/** An atlas, a baker and an evaluator over a set that can see the atlas. */
function rig(options: { resolution?: number; slots?: number } = {}) {
  const atlas = new BrushAtlas(root!, { resolution: 48, slots: 4, ...options });
  return { atlas, baker: new MeshBaker(root!, atlas), evalAt: evalWith(root!, new BrushSet({ atlas })) };
}

// One voxel of the bake box is 2/48 normalised, and the widest bake box here is half-extent
// 4, so a voxel is a third of a world unit at worst. Where the field is linear, trilinear
// reconstruction is exact; this leaves room for f16 and for the linear filter's own rounding.
const EPS = 0.02;

test('a baked cube reproduces the box distance inside, outside and on the surface', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const { atlas, baker, evalAt } = rig();
  // fit 0.5 puts the cube in the middle half of the bake box, so `half` is 2 and the cube
  // comes out at world half-extent 1 - with a whole unit of genuine exterior field around it
  // to measure in, instead of the 0.1 the default fit leaves.
  const mesh = normalizeMesh(cube(), 0.5);
  assert.equal(mesh.triangleCount, 12);
  near(mesh.half, 2, 'the widest axis reaches 1, so a 0.5 fit needs a half-extent of 2', 1e-6);
  assert.deepEqual(mesh.center, [0, 0, 0]);

  const slot = atlas.allocate();
  await baker.bake(slot, mesh);

  const points: [number, number, number][] = [
    [0.3, 0.1, 0.05],
    [1.0, 0.1, 0.05],
    [1.5, 0.1, 0.05],
    [0.05, -1.7, 0.1],
    [0.1, 0.05, 1.9],
  ];
  const got = await evalAt({ kind: 'volume', slot, size: mesh.half }, points);
  for (let i = 0; i < points.length; i++) {
    near(got[i]!, sdBox(points[i]!, 1), `baked cube at ${points[i]!.join(',')}`, EPS);
  }
  assert.ok(got[0]! < 0, 'the interior has to come out negative, not just close');
  atlas.destroy();
});

test('the bake is scale-free: the same mesh at two fits is the same world shape', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // This is the property that lets one bake serve any size: the slot stores distance divided
  // by the bake box's half-extent, and the brush multiplies it back. Drop either half of that
  // and a shape's field stops being 1-Lipschitz the moment it is not baked at fit 1.
  const { atlas, baker, evalAt } = rig();
  const coarse = normalizeMesh(cube(), 0.25);
  const fine = normalizeMesh(cube(), 0.5);
  near(coarse.half, 4, 'a quarter fit doubles the bake box again', 1e-6);

  const slotCoarse = atlas.allocate();
  const slotFine = atlas.allocate();
  await baker.bake(slotCoarse, coarse);
  await baker.bake(slotFine, fine);

  const points: [number, number, number][] = [[1.5, 0.1, 0.05], [0.3, 0.1, 0.05]];
  const a = await evalAt({ kind: 'volume', slot: slotCoarse, size: coarse.half }, points);
  const b = await evalAt({ kind: 'volume', slot: slotFine, size: fine.half }, points);
  for (let i = 0; i < points.length; i++) {
    // The coarse bake spends half as many voxels on the shape, so it gets twice the slack.
    near(a[i]!, sdBox(points[i]!, 1), `fit 0.25 at ${points[i]!.join(',')}`, EPS * 2);
    near(b[i]!, sdBox(points[i]!, 1), `fit 0.5 at ${points[i]!.join(',')}`, EPS);
  }
  atlas.destroy();
});

test('the winding number signs a reverse-wound and a holed mesh', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const { atlas, baker, evalAt } = rig();
  const flipped = atlas.allocate();
  const holed = atlas.allocate();
  // Winding order is an exporter's choice and nothing in the file announces it. Ray parity
  // does not care, but a signed solid angle does - hence the `abs` in the baker, without
  // which half the meshes in the world would bake as empty space.
  await baker.bake(flipped, normalizeMesh(cube({ flip: true }), 0.5));
  // A missing face is what makes ray parity useless on real assets: every ray through the
  // hole reports the whole column as outside. The generalised winding number only loses the
  // solid angle the hole subtends, so the far side of the shape is still solidly inside.
  await baker.bake(holed, normalizeMesh(cube({ skip: 'posY' }), 0.5));

  const inside: [number, number, number] = [0.05, -0.7, 0.1];
  const outside: [number, number, number] = [0.05, -1.6, 0.1];

  const flip = await evalAt({ kind: 'volume', slot: flipped, size: 2 }, [inside, outside]);
  near(flip[0]!, sdBox(inside, 1), 'a reverse-wound cube bakes solid, not inverted', EPS);
  near(flip[1]!, sdBox(outside, 1), 'and its exterior is still exterior', EPS);

  const hole = await evalAt({ kind: 'volume', slot: holed, size: 2 }, [inside, outside]);
  assert.ok(
    hole[0]! < 0,
    `a point away from the hole must still read inside, got ${hole[0]}`,
  );
  near(hole[1]!, sdBox(outside, 1), 'and the exterior near the intact face is unaffected', EPS);
  atlas.destroy();
});

test('slots do not bleed into each other', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // Slots are stacked along Z in one texture, so the outermost row of voxels in a slot sits
  // one texel from the next slot's innermost. A linear fetch there without the half-texel
  // inset blends the two shapes, and the tell is a Z wall that reads as almost-surface.
  const { atlas, baker, evalAt } = rig({ slots: 3 });
  const mesh = normalizeMesh(cube(), 0.5);
  // The middle slot, so there is an unwritten neighbour on both sides. An unwritten slot is
  // zero, i.e. "surface everywhere", which is the worst thing that could bleed in.
  await baker.bake(1, mesh);

  const got = await evalAt({ kind: 'volume', slot: 1, size: mesh.half }, [
    [0.1, 0.05, 1.98],
    [0.1, 0.05, -1.98],
  ]);
  for (const v of got) {
    assert.ok(
      v > 0.8,
      `at the slot's Z wall the field is nearly a unit outside the cube, got ${v} - a value `
        + 'near zero means the sample blended into the neighbouring slot',
    );
  }
  atlas.destroy();
});

test('the baker refuses a slot it does not have and a mesh it cannot hold', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const { atlas, baker } = rig({ slots: 2 });
  const mesh = normalizeMesh(cube(), 0.5);
  await assert.rejects(() => baker.bake(2, mesh), /slot 2 is outside/);
  await assert.rejects(() => baker.bake(-1, mesh), /slot -1 is outside/);
  atlas.allocate();
  atlas.allocate();
  assert.throws(() => atlas.allocate(), /all 2 slots are taken/);

  const tiny = new MeshBaker(root!, atlas, { maxTriangles: 4 });
  await assert.rejects(() => tiny.bake(0, mesh), /12 triangles exceeds/);
  atlas.destroy();
});

/** World distance at each point, out of a traced field's own sampler at mip 0. */
async function sampleField(
  field: TracedField,
  points: readonly [number, number, number][],
): Promise<number[]> {
  const r = root!;
  const n = points.length;
  const pts = r.createReadonly(d.arrayOf(d.vec3f, n));
  pts.write(points.map(([x, y, z]) => d.vec3f(x, y, z)));
  const out = r.createMutable(d.arrayOf(d.f32, n));
  const sample = field.sample;
  let pipeline = r.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= d.u32(n)) {
        return;
      }
      out.$[gid.x] = sample(pts.$[gid.x], 0).x;
    }),
  });
  for (const g of field.groups) {
    pipeline = pipeline.with(g);
  }
  pipeline.dispatchWorkgroups(Math.ceil(n / 64));
  return [...(await out.buffer.read())];
}

test('a baked mesh bakes into a world volume as an ordinary brush', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // The end of the chain, and the part the unit tests above cannot reach: the builder's
  // per-mip pipelines have their own bind groups, and the atlas has to fit alongside them.
  // If the group indices collided, or if `rebuild` forgot to bind the atlas, the fold would
  // sample an unbound texture and the whole volume would come out empty.
  const r = root!;
  const { atlas, baker } = rig();
  const set = new BrushSet({ atlas });
  const norm = normalizeMesh(cube(), 0.5);
  const slot = atlas.allocate();
  await baker.bake(slot, norm);
  const baked: BakedMesh = {
    slot,
    half: norm.half,
    center: norm.center,
    triangleCount: norm.triangleCount,
  };

  // The shape goes through the real authoring path: `sdf.mesh(...).at(...)`, compiled by
  // the same `compileShape`/`compile` pair a level uses.
  const shape = sdf.mesh(baked).at([1, 0, 0]);
  const bounds = shapeBounds(shape, set);
  assert.deepEqual(bounds.center, [1, 0, 0], 'the brush is centred where it was placed');
  near(bounds.radius, 2 * Math.sqrt(3), 'the bake box diagonal bounds any surface in it', 1e-6);

  const volume = new SdfVolume(r, { resolution: 64, worldSize: 8, origin: [-4, -4, -4] });
  const builder = new SdfBuilder(volume, { brushSet: set });
  builder.setBrushes(set.compile(compileShape(shape, () => 0)));

  const encoder = r['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  builder.rebuild(pass);
  pass.end();
  encoder.submit();

  // The stored band is 4 voxels of 0.125, so only |d| < 0.5 survives unsaturated.
  const points: [number, number, number][] = [
    [2.3, 0.05, 0.03],
    [1.7, 0.05, 0.03],
    [1.0, 0.05, 1.25],
    [1.0, -1.3, 0.02],
  ];
  const got = await sampleField(volumeField(volume), points);
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const local: [number, number, number] = [p[0] - 1, p[1], p[2]];
    // A voxel is 0.125 here and the bake under it has its own sampling error, so this is
    // about half a voxel of slack - enough to catch a wrong scale, offset or sign.
    near(got[i]!, sdBox(local, 1), `world volume at ${p.join(',')}`, 0.06);
  }
  volume.destroy();
  atlas.destroy();
});

// --- CPU-only. Here rather than in `test/*.test.mjs` because those files are hand-written
// JS mirrors of GPU logic, and these two are the real functions - which only a bundle that
// runs the TypeScript through can import.

test('normalizeMesh pivots on the bounding box, not the modelling origin', () => {
  // A mesh authored far from its origin - a rock exported with the rest of the level still
  // around it - has to end up centred, or rotating the brush swings it round the old origin.
  const shifted = cube();
  const positions = Float32Array.from(shifted.positions, (v, i) => v + [10, -4, 0.5][i % 3]!);
  const mesh = normalizeMesh({ positions, indices: shifted.indices }, 0.5);
  assert.deepEqual(mesh.center, [10, -4, 0.5]);
  near(mesh.half, 2, 'the extent is unchanged by the offset', 1e-6);
  for (const v of mesh.triangles) {
    near(Math.abs(v), 0.5, 'every corner lands on the fit box', 1e-6);
  }
});

test('normalizeMesh keeps the aspect ratio and rejects degenerate input', () => {
  // The widest axis sets the scale for every axis, so a slab bakes as a slab. Scaling each
  // axis to fill the box independently would be the obvious thing and would silently
  // un-flatten every flat asset in a game.
  const flat = normalizeMesh({
    positions: new Float32Array([-4, -1, 0, 4, -1, 0, 0, 1, 0]),
  }, 1);
  assert.equal(flat.triangleCount, 1);
  near(flat.half, 4, 'the widest half-extent wins', 1e-6);
  near(flat.triangles[0]!, -1, 'the wide axis fills the box', 1e-6);
  near(flat.triangles[1]!, -0.25, 'the narrow one keeps its proportion', 1e-6);

  assert.throws(() => normalizeMesh({ positions: new Float32Array([0, 0, 0]) }), /at least one/);
  assert.throws(
    () => normalizeMesh({ positions: new Float32Array(9) }),
    /no extent/,
    'a mesh collapsed to a point would divide the scale by zero',
  );
});

test('parseObj reads positions and faces and ignores everything else', () => {
  const mesh = parseObj(`
# a quad and a triangle
v 0 0 0
vn 0 1 0
v 1 0 0
vt 0 0
v 1 1 0
v 0 1 0
usemtl none
f 1//1 2//1 3//1 4//1
f -4/1/1 -3/1/1 -2/1/1
`);
  assert.deepEqual(Array.from(mesh.positions), [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
  // The quad is fanned from its first corner; the negative-index face counts back from the
  // newest vertex, so it names the same three points as `1 2 3`.
  assert.deepEqual(Array.from(mesh.indices!), [0, 1, 2, 0, 2, 3, 0, 1, 2]);
  assert.throws(() => parseObj('# nothing here\n'), /no `v`\/`f` lines/);
});
