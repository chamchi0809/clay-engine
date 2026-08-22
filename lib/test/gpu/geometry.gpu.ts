import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { BrushSet, defaultBrushSet } from '../../src/field/brush.ts';
import { BrushAtlas } from '../../src/field/atlas.ts';
import { MeshBaker } from '../../src/field/meshbake.ts';
import { geometry, signedArea, triangulateShape } from '../../src/shape/geometry.ts';
import type { Shape2D } from '../../src/shape/geometry.ts';
import { triangleArea } from '../../src/shape/polygon.ts';
import { normalizeMesh, type MeshData } from '../../src/shape/mesh.ts';
import { evalWith, near } from './evalbrush.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * three.js's geometry catalogue.
 *
 * Two things can go wrong with a generator and neither one throws. It can produce a mesh of
 * the wrong *size* - a `radius` read as a diameter, a `height` that turns out to be a half
 * height - which nobody notices until an asset lands next to an analytic brush that took the
 * same number and came out twice as big. Or it can produce a mesh that is not closed, or is
 * wound inward, which a rasteriser draws quite happily and the baker turns into a hollow
 * shape or an inside-out one.
 *
 * So the tests here are about dimensions and about topology, and mostly run on the CPU: a
 * generator is arithmetic over arrays and needs no driver. The GPU is only asked the one
 * question it alone can answer, which is whether a generated mesh bakes into the field its
 * closed form predicts.
 */
let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

// --- reading a MeshData back ----------------------------------------------

const triangleCount = (mesh: MeshData): number =>
  Math.floor((mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3);

const corner = (mesh: MeshData, t: number, k: number): number =>
  mesh.indices ? mesh.indices[t * 3 + k]! : t * 3 + k;

const vertex = (mesh: MeshData, v: number): [number, number, number] =>
  [mesh.positions[v * 3]!, mesh.positions[v * 3 + 1]!, mesh.positions[v * 3 + 2]!];

const bounds = (mesh: MeshData) => {
  const lo: [number, number, number] = [Infinity, Infinity, Infinity];
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a]!, mesh.positions[i + a]!);
      hi[a] = Math.max(hi[a]!, mesh.positions[i + a]!);
    }
  }
  return { lo, hi };
};

/** Smallest and largest distance from the origin over every vertex. */
const radii = (mesh: MeshData) => {
  let min = Infinity;
  let max = 0;
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const r = Math.hypot(mesh.positions[i]!, mesh.positions[i + 1]!, mesh.positions[i + 2]!);
    min = Math.min(min, r);
    max = Math.max(max, r);
  }
  return { min, max };
};

/**
 * Signed volume, by the divergence theorem: a sixth of the sum of `a . (b x c)`.
 *
 * Positive means the triangles are wound counter-clockwise seen from outside, which is the
 * winding the baker's generalised winding number needs to call the interior "inside". A mesh
 * that is otherwise perfect and wound the other way bakes as a solid universe with a
 * hole in it, and nothing before the bake complains.
 */
function volume(mesh: MeshData): number {
  let sum = 0;
  for (let t = 0; t < triangleCount(mesh); t++) {
    const a = vertex(mesh, corner(mesh, t, 0));
    const b = vertex(mesh, corner(mesh, t, 1));
    const c = vertex(mesh, corner(mesh, t, 2));
    sum += a[0] * (b[1] * c[2] - b[2] * c[1])
      + a[1] * (b[2] * c[0] - b[0] * c[2])
      + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return sum / 6;
}

/**
 * Directed edges that have no opposite, after welding vertices by position.
 *
 * Welding is the point: none of these generators share a vertex across a seam or a face -
 * three.js does not either, because its vertices carry UVs - so index-level manifoldness is
 * never true and never the question. What matters is whether the *surface* closes, and the
 * pole of a sphere or the seam of a torus closes by two vertices landing on the same point.
 *
 * Triangles that collapse under welding are dropped rather than counted: a sphere's pole row
 * is a fan of them by construction, in three.js as here.
 */
function openEdges(mesh: MeshData): number {
  const grid = 1e6;
  const ids = new Map<string, number>();
  const weld = (v: number): number => {
    const p = vertex(mesh, v);
    // Rounded, not truncated, and stringified so that `-0` and `0` are the same key: a seam
    // closes at `cos(2*pi)`, which is a rounding error away from `cos(0)` rather than equal
    // to it, and a pole sits at `-0` on the axes it collapses along.
    const k = `${Math.round(p[0] * grid) / grid},${Math.round(p[1] * grid) / grid},${Math.round(p[2] * grid) / grid}`;
    const found = ids.get(k);
    if (found !== undefined) {
      return found;
    }
    ids.set(k, v);
    return v;
  };

  const half = new Map<string, number>();
  for (let t = 0; t < triangleCount(mesh); t++) {
    const w = [0, 1, 2].map((k) => weld(corner(mesh, t, k)));
    if (w[0] === w[1] || w[1] === w[2] || w[2] === w[0]) {
      continue;
    }
    for (let e = 0; e < 3; e++) {
      const a = w[e]!;
      const b = w[(e + 1) % 3]!;
      half.set(`${a}_${b}`, (half.get(`${a}_${b}`) ?? 0) + 1);
    }
  }
  let unmatched = 0;
  for (const [k, count] of half) {
    const [a, b] = k.split('_');
    unmatched += Math.abs(count - (half.get(`${b}_${a}`) ?? 0));
  }
  return unmatched;
}

// --- outlines used by `shape` and `extrude` -------------------------------

/** A unit square with a square hole a quarter its area, wound however it felt like. */
const HOLED: Shape2D = {
  contour: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
  holes: [[[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]],
};

// --- the catalogue --------------------------------------------------------

/**
 * One call to every generator the module exports, with parameters that exercise the
 * interesting branch of each: a hole, a bevel, a closed sweep, a subdivision.
 */
const CATALOGUE: Record<string, () => MeshData> = {
  plane: () => geometry.plane({ width: 3, height: 2, widthSegments: 2, heightSegments: 3 }),
  circle: () => geometry.circle({ radius: 2, segments: 24 }),
  ring: () => geometry.ring({ innerRadius: 1, outerRadius: 2, phiSegments: 2 }),
  shape: () => geometry.shape({ shapes: HOLED }),
  box: () => geometry.box({ width: 2, height: 4, depth: 6, widthSegments: 2 }),
  sphere: () => geometry.sphere({ radius: 2, widthSegments: 12, heightSegments: 8 }),
  cylinder: () => geometry.cylinder({ radiusTop: 1, radiusBottom: 2, height: 3 }),
  cone: () => geometry.cone({ radius: 1, height: 2 }),
  capsule: () => geometry.capsule({ radius: 0.5, height: 2, capSegments: 8, radialSegments: 24 }),
  lathe: () => geometry.lathe({ points: [[0, -1], [1, -0.5], [0.6, 0.5], [0, 1]], segments: 16 }),
  torus: () => geometry.torus({ radius: 2, tube: 0.5 }),
  torusKnot: () => geometry.torusKnot({ radius: 1, tube: 0.2, p: 2, q: 3 }),
  tube: () => geometry.tube({
    path: (t) => [Math.cos(t * Math.PI * 2), Math.sin(t * Math.PI * 2) * 0.5, Math.sin(t * Math.PI * 4) * 0.3],
    radius: 0.2,
    closed: true,
  }),
  // A bipyramid over the equatorial square, apexes at 0 and 1. Wound apex-last around the
  // top and apex-first around the bottom, which is what puts both fans' normals outward -
  // `polyhedron` passes the winding straight through, and the baker's winding number is the
  // one consumer that cares which way round it is.
  polyhedron: () => geometry.polyhedron({
    vertices: [0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1],
    indices: [0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 2, 5, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 2],
    radius: 2,
    detail: 1,
  }),
  tetrahedron: () => geometry.tetrahedron({ radius: 1.5 }),
  octahedron: () => geometry.octahedron({ radius: 1.5 }),
  icosahedron: () => geometry.icosahedron({ radius: 1.5 }),
  dodecahedron: () => geometry.dodecahedron({ radius: 1.5 }),
  extrude: () => geometry.extrude({ shapes: HOLED, depth: 1, bevelEnabled: false }),
  edges: () => geometry.edges({ geometry: geometry.box({ widthSegments: 2 }) }),
  wireframe: () => geometry.wireframe({ geometry: geometry.box({ widthSegments: 2 }) }),
};

test('every generator produces a well-formed mesh', () => {
  for (const [name, make] of Object.entries(CATALOGUE)) {
    const mesh = make();
    const count = triangleCount(mesh);
    assert.ok(count > 0, `${name}: no triangles`);
    assert.equal(mesh.positions.length % 3, 0, `${name}: positions are not xyz triples`);
    for (let i = 0; i < mesh.positions.length; i++) {
      assert.ok(Number.isFinite(mesh.positions[i]!), `${name}: position ${i} is not finite`);
    }
    // An out-of-range index reads as `undefined` in JS and as garbage memory on the GPU, so
    // it is the one error here that could actually corrupt a bake rather than just look wrong.
    const vertices = mesh.positions.length / 3;
    for (let t = 0; t < count; t++) {
      for (let k = 0; k < 3; k++) {
        const v = corner(mesh, t, k);
        assert.ok(v >= 0 && v < vertices, `${name}: index ${v} is outside ${vertices} vertices`);
      }
    }
    // `Builder.tri` drops the degenerate ones, so a generator that produced nothing but
    // degenerate triangles would have thrown by now - but a merely lopsided one would not.
    assert.ok(
      count >= 4,
      `${name}: ${count} triangles is too few for any of these shapes`,
    );
  }
});

test('the generators take three.js parameters at three.js scale', () => {
  // Every assertion here is a number three.js would produce for the same options. The failure
  // this catches is the quiet one: a radius used as a diameter, or a height used as a half
  // height, which makes an asset that loads and spawns and is simply the wrong size.
  const box = bounds(CATALOGUE.box!());
  assert.deepEqual(box.lo, [-1, -2, -3], 'width, height and depth are full extents');
  assert.deepEqual(box.hi, [1, 2, 3]);

  const sphere = radii(CATALOGUE.sphere!());
  near(sphere.min, 2, 'every vertex of a sphere is at the radius', 1e-6);
  near(sphere.max, 2, 'and none further', 1e-6);

  // The platonics take a *circumradius*, which is what `polyhedron` pushes every vertex to -
  // including the ones subdivision invents, which is why a detailed one is a geodesic sphere.
  for (const name of ['tetrahedron', 'octahedron', 'icosahedron', 'dodecahedron', 'polyhedron']) {
    const r = radii(CATALOGUE[name]!());
    near(r.min, name === 'polyhedron' ? 2 : 1.5, `${name}: vertices sit on the circumradius`, 1e-6);
    near(r.max, name === 'polyhedron' ? 2 : 1.5, `${name}: and none escape it`, 1e-6);
  }

  // three.js's capsule `height` is the *cylindrical* middle, so the thing is taller than it.
  const capsule = bounds(CATALOGUE.capsule!());
  near(capsule.hi[1]!, 1.5, 'half the middle plus a radius', 1e-6);
  near(capsule.lo[1]!, -1.5, 'and the same below', 1e-6);
  near(Math.max(capsule.hi[0]!, capsule.hi[2]!), 0.5, 'the barrel is one radius wide', 1e-6);

  // A torus lies in XY with its hole along Z - three.js's orientation, and *not* the one
  // `sdf.torus` uses. The engine's own primitives stand on Y; these follow three.js instead.
  const torus = bounds(CATALOGUE.torus!());
  near(torus.hi[0]!, 2.5, 'outer radius is radius + tube', 1e-6);
  near(torus.hi[2]!, 0.5, 'and the hole runs along Z, so the tube is all the depth there is', 1e-6);

  const ring = radii(CATALOGUE.ring!());
  near(ring.min, 1, 'the inner radius is a genuine hole, not a decoration', 1e-6);
  near(ring.max, 2, 'and the outer one is where it stops', 1e-6);

  const cylinder = bounds(CATALOGUE.cylinder!());
  near(cylinder.hi[1]!, 1.5, 'height is the full height, centred on the origin', 1e-6);
  near(cylinder.hi[0]!, 2, 'and the wider of the two radii sets the width', 1e-6);

  // three.js's cone is a cylinder with a zero top radius, apex up.
  const cone = bounds(CATALOGUE.cone!());
  near(cone.hi[1]!, 1, 'the apex is half the height up', 1e-6);
  near(cone.hi[0]!, 1, 'and the base is one radius across', 1e-6);
});

test('the closed generators close, and face outwards', () => {
  // Volume in the same units the parameters were given in, so a wrong-by-a-factor generator
  // shows up as a wrong volume rather than merely a positive one.
  const expected: Record<string, number> = {
    box: 2 * 4 * 6,
    sphere: (4 / 3) * Math.PI * 8,
    // A truncated cone: pi*h/3 * (R^2 + R*r + r^2).
    cylinder: (Math.PI * 3) / 3 * (4 + 2 + 1),
    cone: (Math.PI * 1 * 2) / 3,
    // A cylinder of radius 0.5 and height 2, plus a whole sphere of radius 0.5.
    capsule: Math.PI * 0.25 * 2 + (4 / 3) * Math.PI * 0.125,
    torus: 2 * Math.PI * Math.PI * 2 * 0.25,
    // Circumradius r: a tetrahedron has edge r*sqrt(8/3), an octahedron edge r*sqrt(2), and
    // both have volume `k * edge^3` for a constant of their own.
    tetrahedron: (Math.SQRT2 / 12) * (1.5 * Math.sqrt(8 / 3)) ** 3,
    octahedron: (Math.SQRT2 / 3) * (1.5 * Math.SQRT2) ** 3,
    // The 2x2 square with a 1x1 hole, extruded one unit: area 4 - 1 = 3.
    extrude: 3,
  };

  for (const name of Object.keys(expected)) {
    const mesh = CATALOGUE[name]!();
    assert.equal(openEdges(mesh), 0, `${name}: the surface does not close`);
    // A faceted mesh under-shoots the smooth solid it approximates, never over-shoots, so the
    // tolerance is one-sided and generous - the assertion is about the order of magnitude and
    // the sign, which is where a doubled radius or a flipped winding lands.
    const got = volume(mesh);
    assert.ok(got > 0, `${name}: volume ${got} - the mesh is wound inside out`);
    assert.ok(
      got > expected[name]! * 0.85 && got < expected[name]! * 1.001,
      `${name}: volume ${got}, expected about ${expected[name]}`,
    );
  }

  // The knot and the closed tube have no volume worth writing down, but they still have to
  // close. The tube is the interesting one: its last ring is a separate ring of vertices that
  // only lands on the first if the swept frame comes back to where it started, which is what
  // the closing rotation - and the wrapped finite difference underneath it - are for.
  for (const name of ['torusKnot', 'tube', 'icosahedron', 'dodecahedron', 'polyhedron', 'lathe']) {
    const mesh = CATALOGUE[name]!();
    assert.equal(openEdges(mesh), 0, `${name}: the surface does not close`);
    assert.ok(volume(mesh) > 0, `${name}: wound inside out`);
  }

  // And the flat ones do not close, by definition - every edge of a disc's rim is a border.
  // This is the property the shell bake exists to work around, so it is worth pinning.
  for (const name of ['plane', 'circle', 'ring', 'shape']) {
    assert.ok(openEdges(CATALOGUE[name]!()) > 0, `${name}: a surface cannot be watertight`);
  }
});

test('triangulateShape cuts an outline with a hole into the right amount of triangle', () => {
  const tri = triangulateShape(HOLED);
  assert.equal(tri.rings.length, 2, 'the contour and its one hole');
  assert.equal(tri.rings[0], 0, 'the contour comes first');
  assert.equal(tri.rings[1], 4, 'and the hole starts after its four points');
  // 2x2 square minus a 1x1 hole. Ear clipping cannot lose or duplicate area without either
  // dropping a triangle or overlapping two, so the area is the whole correctness check.
  near(triangleArea(tri), 4 - 1, 'the triangles cover the outline and not the hole', 1e-9);

  // Winding is normalised on the way in, so an outline that arrives clockwise - which is
  // most of what an SVG or a font hands over - triangulates the same.
  const flipped = triangulateShape({
    contour: [...HOLED.contour].reverse(),
    holes: [[...HOLED.holes![0]!].reverse()],
  });
  near(triangleArea(flipped), 4 - 1, 'a reversed outline is the same outline', 1e-9);
  assert.ok(signedArea(HOLED.contour) > 0, 'the fixture itself is counter-clockwise');
  assert.ok(signedArea([...HOLED.contour].reverse()) < 0, 'and its reverse is not');
});

test('the flat generators refuse to bake as solids and accept a thickness', () => {
  for (const name of ['plane', 'circle', 'ring', 'shape']) {
    const mesh = CATALOGUE[name]!();
    // The message is the deliverable here: a flat mesh bakes to an empty slot, which is an
    // object that loads without error and cannot be seen. Failing loudly with the fix in the
    // text is the whole reason `normalizeMesh` looks at the bounding box at all.
    assert.throws(() => normalizeMesh(mesh), /thickness/, `${name}: should refuse a solid bake`);
    const shelled = normalizeMesh(mesh, 0.9, 0.2);
    assert.ok(shelled.shell > 0, `${name}: a thickness has to reach the brush`);
    near(shelled.shell * shelled.half, 0.1, `${name}: half the thickness, in world units`, 1e-6);
  }
  // A solid is unaffected: `thickness` is a union, so it can only add.
  assert.ok(normalizeMesh(CATALOGUE.box!()).shell === 0, 'no thickness asked for, no shell');
});

test('edges keeps only the creases, wireframe keeps every edge', () => {
  // Each rod is a square prism: six quads, twelve triangles. A cube subdivided two ways per
  // axis has twelve creases split in two, so twenty-four rods - and every other edge in it,
  // the grid lines and the quad diagonals, is flat and must not show.
  const box = geometry.box({ widthSegments: 2, heightSegments: 2, depthSegments: 2 });
  const creases = geometry.edges({ geometry: box });
  assert.equal(triangleCount(creases), 24 * 12, 'a subdivided cube still has twelve edges');

  // The wireframe keeps everything: 48 triangles over a closed surface share 72 edges, so
  // three times the rods - the grid lines and each quad's diagonal on top of the creases.
  const all = geometry.wireframe({ geometry: box });
  assert.equal(triangleCount(all), 72 * 12, 'a wireframe draws the grid and the diagonals too');

  // Raising the threshold past ninety degrees leaves nothing to draw, and an empty mesh is
  // not a mesh - better to say so than to hand back something that bakes to nothing.
  assert.throws(
    () => geometry.edges({ geometry: box, thresholdAngle: 120 }),
    /no edges past the threshold/,
  );

  // The rods are solid, which is the point of translating a line list this way at all.
  assert.equal(openEdges(creases), 0, 'a rod cage is closed');
  assert.ok(volume(creases) > 0, 'and wound outward, so it bakes as material');
});

// --- the one question that needs a driver ---------------------------------

/** Closed form for a box of half-extents `h` centred on the origin. */
function sdBox(p: readonly [number, number, number], h: readonly [number, number, number]): number {
  const q = [Math.abs(p[0]) - h[0]!, Math.abs(p[1]) - h[1]!, Math.abs(p[2]) - h[2]!];
  const outside = Math.hypot(Math.max(q[0]!, 0), Math.max(q[1]!, 0), Math.max(q[2]!, 0));
  return outside + Math.min(Math.max(q[0]!, q[1]!, q[2]!), 0);
}

function rig(options: { resolution?: number; slots?: number } = {}) {
  const atlas = new BrushAtlas(root!, { resolution: 64, slots: 2, ...options });
  return { atlas, baker: new MeshBaker(root!, atlas), evalAt: evalWith(root!, new BrushSet({ atlas })) };
}

test('the platonic meshes and the platonic brushes are the same solid', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // The claim `geometry.ts` and the README both make is that `sdf.icosahedron(r)` and
  // `geometry.icosahedron({ radius: r })` are interchangeable - same size, same orientation -
  // so a game can start with the analytic brush and swap in a bake, or stand the two side by
  // side. Nothing enforces that except this: every vertex and every face of the generated
  // mesh is fed to the analytic brush, which has to answer zero.
  //
  // It is not a hypothetical. The dodecahedron and the icosahedron were both written with the
  // *mirror* face-normal family - a perfectly good solid, a tenth of a turn off three.js's -
  // and nothing looked wrong until the mesh and the brush were put in the same scene.
  const evalAt = evalWith(root!, defaultBrushSet);
  const R = 1.5;
  for (const name of ['tetrahedron', 'octahedron', 'icosahedron', 'dodecahedron'] as const) {
    const mesh = geometry[name]({ radius: R });
    const points: [number, number, number][] = [];
    for (let v = 0; v < mesh.positions.length / 3; v++) {
      points.push(vertex(mesh, v));
    }
    // Face centroids as well as vertices, because a mirrored solid still has *some* vertex
    // on every one of its own planes - it is the faces in between that end up in the wrong
    // place. A centroid is strictly inside its face, so its own plane is the nearest one.
    const faces = triangleCount(mesh);
    for (let t2 = 0; t2 < faces; t2++) {
      const c = [0, 1, 2].map((k) => vertex(mesh, corner(mesh, t2, k)));
      points.push([
        (c[0]![0] + c[1]![0] + c[2]![0]) / 3,
        (c[0]![1] + c[1]![1] + c[2]![1]) / 3,
        (c[0]![2] + c[1]![2] + c[2]![2]) / 3,
      ]);
    }
    const surface = points.length;
    // And the same points pulled a tenth of the way in, which have to read as inside: a
    // solid that merely *touched* all those points at zero would pass everything above.
    for (let i = 0; i < surface; i++) {
      const p = points[i]!;
      points.push([p[0] * 0.9, p[1] * 0.9, p[2] * 0.9]);
    }

    const got = await evalAt({ kind: name, size: R }, points);
    for (let i = 0; i < surface; i++) {
      near(got[i]!, 0, `${name}: mesh point ${points[i]!.join(',')} is off the brush's surface`);
    }
    for (let i = surface; i < points.length; i++) {
      assert.ok(got[i]! < -1e-3, `${name}: a point inside the mesh reads as ${got[i]}`);
    }
  }
});

test('a generated box bakes into the field its closed form predicts', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // The generators are only worth anything if what comes out the far end of the bake is the
  // shape that went in, so one of them is checked against a closed form the engine already
  // has. The box is the one where any mistake - a face wound inward, a missing face, an
  // extent halved - changes the answer somewhere on these five points.
  const { atlas, baker, evalAt } = rig();
  const mesh = normalizeMesh(geometry.box({ width: 2, height: 2, depth: 2 }), 0.5);
  assert.equal(mesh.triangleCount, 12, 'a box is twelve triangles, as in three.js');
  near(mesh.half, 2, 'the widest half-extent is 1, so a 0.5 fit needs a box of 2', 1e-6);

  const slot = atlas.allocate();
  await baker.bake(slot, mesh);
  const points: [number, number, number][] = [
    [0.3, 0.1, 0.05], [1.0, 0.1, 0.05], [1.5, 0.1, 0.05], [0.05, -1.7, 0.1], [0.1, 0.05, 1.9],
  ];
  const got = await evalAt({ kind: 'volume', slot, size: mesh.half }, points);
  for (let i = 0; i < points.length; i++) {
    near(got[i]!, sdBox(points[i]!, [1, 1, 1]), `baked box at ${points[i]!.join(',')}`, 0.02);
  }
  assert.ok(got[0]! < 0, 'the interior is negative, so the winding came out right');
  atlas.destroy();
});

test('a thickness turns a flat generator into a slab', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // The bake is `min(solid, dist - shell)`, and a plane encloses nothing, so `solid` is just
  // the unsigned distance and the whole field is the surface offset by `shell`. That makes
  // the expected value the distance to the *quad*, minus a quarter - not the distance to the
  // slab-shaped box the quad's silhouette suggests, which is what makes the last point here
  // worth having: it is off the side, where the two disagree.
  const { atlas, baker, evalAt } = rig();
  const mesh = normalizeMesh(geometry.plane({ width: 2, height: 2 }), 0.5, 0.5);
  near(mesh.half, 2.5, 'the bake box grows by half the thickness before the fit is applied', 1e-6);
  near(mesh.shell * mesh.half, 0.25, 'half the thickness, in world units', 1e-6);

  const slot = atlas.allocate();
  await baker.bake(slot, mesh);
  // The plane lies in XY facing +Z, so the slab is `|z| <= 0.25` over the unit square. The
  // last two points are off the side of the quad, where the shell wraps its rim: the distance
  // there is to the *edge*, not to the slab-shaped box the silhouette suggests.
  const probes: [[number, number, number], number, number][] = [
    // A voxel is 2 * 2.5 / 64 world units, and the field has a kink at the surface itself,
    // where a linear filter can only cut the corner off. The trough therefore reads about
    // half a voxel shallow - inherent to a sampled field, not a bake error, and the reason a
    // shell wants to be a few voxels thick.
    [[0.05, 0.03, 0], -0.25, 0.05],
    [[0.02, 0.05, 0.125], -0.125, 0.02],
    [[0.02, 0.05, 0.75], 0.5, 0.02],
    [[0.02, 0.05, 1.25], 1.0, 0.02],
    [[1.6, 0.05, 0], 0.35, 0.03],
  ];
  const got = await evalAt(
    { kind: 'volume', slot, size: mesh.half },
    probes.map(([p]) => p),
  );
  for (let i = 0; i < probes.length; i++) {
    const [p, want, eps] = probes[i]!;
    near(got[i]!, want, `shelled plane at ${p.join(',')}`, eps);
  }
  assert.ok(got[0]! < 0, 'the middle of the slab is inside it, which is the entire point');
  atlas.destroy();
});
