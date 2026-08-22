import {
  Game,
  geometry,
  orbit,
  sdf,
  type BakedMesh,
  type MeshData,
  type Shape,
  type Vec2,
} from '@clay/engine';
import type { DemoStart } from '../shell.ts';

/**
 * Every geometry the engine can make, all of them at once, each next to the exact
 * primitive it overlaps with.
 *
 * The point of the pairing is that it is checkable by eye. On the left of each pair is a
 * triangle mesh from `geometry.*` - three.js's catalogue, same constructor names and same
 * parameters - put through `game.loadMesh`, which votes every voxel of a 64^3 box against
 * 80-odd thousand triangles and stores the result as a normalised distance. On the right,
 * where there is one, is `sdf.*`: the same shape as a closed-form distance function,
 * evaluated exactly, wherever the ray happens to be. If the bake is right the two
 * silhouettes agree and only the corners differ, because a corner is where a 64^3 grid
 * runs out.
 *
 * Which is also the advice: reach for the right-hand one when it exists. It costs no atlas
 * slot, no bake, and no resolution. The left-hand route is for the fifteen shapes here
 * that have no closed form worth writing - a torus knot, a lathed profile, an extruded
 * outline, a subdivided polyhedron - and for a three.js scene being ported across as it
 * stands.
 *
 * Four of these enclose no volume at all. A plane, a disc, an annulus and a triangulated
 * outline are surfaces, and a surface bakes to an empty field; they are loaded with a
 * `thickness` instead, which offsets the surface into a shell. (`edges` and `wireframe`
 * look like they belong on that list and do not: they hand back square rods, because a
 * line has no volume either and a rod is what a line has to become to be baked at all.)
 *
 * It is one field. Every shape on the board - forty-odd brushes and six shelves - is one
 * `sdf.union` baked into a single volume once at load, and nothing about the scene changes
 * after that. Which is the thing worth noticing about a distance field: the cost of a
 * shape is where it *is*, not how many there are. The builder culls brushes into 8^3
 * tiles, so a voxel over the torus knot folds the torus knot and its shelf and nothing
 * else, and forty shapes trace at the price of the two that used to be here.
 *
 * `#pick=torusKnot` orbits that one pair close up instead of the whole board, and
 * `#spin=off` stops the drift.
 */

// --- outlines the flat geometries are built from --------------------------

const ring = (n: number, radius: number, phase = 0): Vec2[] =>
  Array.from({ length: n }, (_unused, i): Vec2 => {
    const a = (i / n) * Math.PI * 2 + phase;
    return [Math.cos(a) * radius, Math.sin(a) * radius];
  });

/**
 * How thick a thing with no thickness is made.
 *
 * `SHELL` is what the flat generators are baked with and `ROD` is the square section
 * `edges` and `wireframe` turn a line segment into, and both are set by the *volume* rather
 * than by taste. The board is 21 units across 256 voxels, so a voxel is 0.082, and a slab
 * an eyelash over one voxel thick lands between the samples: it traces as speckle, and the
 * hole cut through the ring's twin frays. Two voxels is the point at which trilinear
 * reconstruction has something to reconstruct, so these are both a shade over two.
 */
const SHELL = 0.18;
const ROD = 0.18;

/** A five-pointed star, for `extrude`. */
const STAR: Vec2[] = Array.from({ length: 10 }, (_unused, i): Vec2 => {
  const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
  const r = i % 2 === 0 ? 0.95 : 0.45;
  return [Math.cos(a) * r, Math.sin(a) * r];
});

/** A vase profile for `lathe`, on the axis at both ends so the sweep closes into a solid. */
const VASE: Vec2[] = [
  [0, -0.95], [0.6, -0.9], [0.66, -0.45], [0.34, 0.15], [0.46, 0.62], [0.4, 0.95], [0, 0.98],
];

interface Entry {
  /** Matches the `geometry.*` function, and what `#pick=` takes. */
  name: string;
  /** The three.js constructor this mirrors. */
  three: string;
  /** The arguments below, in prose, since the HUD cannot show code. */
  params: string;
  /** Triangles to bake. Every entry has one; that is the point of the catalogue. */
  mesh: MeshData;
  /** Shell thickness, for the entries that enclose no volume. */
  thickness?: number;
  /** The closed-form primitive covering the same shape, where one exists. */
  analytic?: Shape;
  /** How `sdf.*` was called, and anything surprising about the pairing. */
  note?: string;
}

/**
 * Built at module load, not per switch: these are a few thousand triangles each and pure
 * arithmetic, so there is nothing to defer and a lot to be said for failing at import time
 * if one of them is broken.
 */
const CATALOGUE: readonly Entry[] = [
  {
    name: 'plane',
    three: 'PlaneGeometry',
    params: 'width 1.8, height 1.8, 4x4 segments',
    mesh: geometry.plane({ width: 1.8, height: 1.8, widthSegments: 4, heightSegments: 4 }),
    thickness: SHELL,
    analytic: sdf.box([0.9, 0.9, SHELL / 2]),
    note: `flat, so it is baked as a ${SHELL} shell - which is a slab, and sdf.box is the slab`,
  },
  {
    name: 'circle',
    three: 'CircleGeometry',
    params: 'radius 0.95, 48 segments',
    mesh: geometry.circle({ radius: 0.95, segments: 48 }),
    thickness: SHELL,
    analytic: sdf.cylinder(0.95, SHELL / 2).rotate([Math.PI / 2, 0, 0]),
    note: 'a disc in XY; the cylinder is rotated onto its side to face the same way',
  },
  {
    name: 'ring',
    three: 'RingGeometry',
    params: 'inner 0.45, outer 0.95, 48 segments',
    mesh: geometry.ring({ innerRadius: 0.45, outerRadius: 0.95, thetaSegments: 48 }),
    thickness: SHELL,
    // A hole is not a primitive, so the twin is two of them: the inner one is deliberately
    // the taller, since a cutter flush with the surface it cuts leaves a film behind.
    analytic: sdf
      .cut(sdf.cylinder(0.95, SHELL / 2), sdf.cylinder(0.45, SHELL))
      .rotate([Math.PI / 2, 0, 0]),
    note: 'no primitive has a hole in it, so the twin is sdf.cut of two cylinders',
  },
  {
    name: 'shape',
    three: 'ShapeGeometry',
    params: 'a hexagon with a triangular hole, ear-clipped',
    mesh: geometry.shape({ shapes: { contour: ring(6, 0.95, Math.PI / 6), holes: [ring(3, 0.44)] } }),
    thickness: SHELL,
    note: 'the winding of the hole is not asked for - the triangulator orients both rings',
  },
  {
    name: 'extrude',
    three: 'ExtrudeGeometry',
    params: 'a five-pointed star, depth 0.5, bevelled 0.12 x 0.1 over 3 segments',
    mesh: geometry.extrude({
      shapes: { contour: STAR },
      depth: 0.5,
      bevelThickness: 0.12,
      bevelSize: 0.1,
      bevelSegments: 3,
    }),
    note: 'the bevel contracts the outline along its angle bisectors, so the points stay points',
  },
  {
    name: 'box',
    three: 'BoxGeometry',
    params: 'width 1.8, height 1.4, depth 1.2',
    mesh: geometry.box({ width: 1.8, height: 1.4, depth: 1.2 }),
    analytic: sdf.box([0.9, 0.7, 0.6]),
    note: 'sdf.box takes half-extents, three.js takes full ones',
  },
  {
    name: 'sphere',
    three: 'SphereGeometry',
    params: 'radius 1, 32x16 segments',
    mesh: geometry.sphere({ radius: 1 }),
    analytic: sdf.sphere(1),
    note: 'the honest look at what a 64^3 bake costs: the mesh is faceted, the twin is not',
  },
  {
    name: 'cylinder',
    three: 'CylinderGeometry',
    params: 'radius 0.8, height 1.8',
    mesh: geometry.cylinder({ radiusTop: 0.8, radiusBottom: 0.8, height: 1.8 }),
    analytic: sdf.cylinder(0.8, 0.9),
  },
  {
    name: 'cappedCone',
    three: 'CylinderGeometry, with the two radii different',
    params: 'bottom 0.9, top 0.35, height 1.8',
    mesh: geometry.cylinder({ radiusBottom: 0.9, radiusTop: 0.35, height: 1.8 }),
    analytic: sdf.cappedCone(0.9, 0.35, 0.9),
    note: 'three.js has no CappedConeGeometry; a cylinder with unequal radii is one',
  },
  {
    name: 'cone',
    three: 'ConeGeometry',
    params: 'radius 0.9, height 1.8',
    mesh: geometry.cone({ radius: 0.9, height: 1.8 }),
    analytic: sdf.cone(0.9, 0.9),
    note: 'apex up, base at -halfHeight, matching three.js',
  },
  {
    name: 'capsule',
    three: 'CapsuleGeometry',
    params: 'radius 0.5, height 1',
    mesh: geometry.capsule({ radius: 0.5, height: 1 }),
    analytic: sdf.capsule(0.5, 0.5),
    note: 'height is the cylindrical middle only, so both of these are 2 units tall',
  },
  {
    name: 'lathe',
    three: 'LatheGeometry',
    params: 'a 7-point vase profile, 48 segments',
    mesh: geometry.lathe({ points: VASE, segments: 48 }),
    note: 'the profile touches the axis at both ends, which is what closes the sweep into a solid',
  },
  {
    name: 'torus',
    three: 'TorusGeometry',
    params: 'radius 0.75, tube 0.3',
    mesh: geometry.torus({ radius: 0.75, tube: 0.3 }),
    analytic: sdf.torus(0.75, 0.3).rotate([Math.PI / 2, 0, 0]),
    note: "three.js's torus lies in XY, sdf.torus in XZ - the twin is turned to agree",
  },
  {
    name: 'torusKnot',
    three: 'TorusKnotGeometry',
    params: 'radius 0.75, tube 0.24, p 2, q 3',
    mesh: geometry.torusKnot({ radius: 0.75, tube: 0.24, p: 2, q: 3 }),
    note: 'no closed form worth writing - this is the shape a bake exists for',
  },
  {
    name: 'tube',
    three: 'TubeGeometry',
    params: 'a closed loop with a threefold wobble, radius 0.16',
    mesh: geometry.tube({
      path: (t) => {
        const a = t * Math.PI * 2;
        return [Math.cos(a) * 0.85, Math.sin(a * 3) * 0.3, Math.sin(a) * 0.85];
      },
      radius: 0.16,
      tubularSegments: 128,
      radialSegments: 10,
      closed: true,
    }),
    note: 'closed, so the frame is smoothed across the seam and the surface has an inside',
  },
  {
    name: 'polyhedron',
    three: 'PolyhedronGeometry',
    params: 'a triangular bipyramid, radius 1, detail 0',
    mesh: geometry.polyhedron({
      vertices: [0, 1, 0, 0, -1, 0, 1, 0, 0, -0.5, 0, 0.866, -0.5, 0, -0.866],
      indices: [0, 2, 3, 0, 3, 4, 0, 4, 2, 1, 3, 2, 1, 4, 3, 1, 2, 4],
      radius: 1,
    }),
    note: 'raising detail subdivides every face and pushes the new vertices out to radius',
  },
  {
    name: 'tetrahedron',
    three: 'TetrahedronGeometry',
    params: 'radius 1',
    mesh: geometry.tetrahedron({ radius: 1 }),
    analytic: sdf.tetrahedron(1),
    note: 'radius is the circumradius for both, and both put a vertex at (1, 1, 1)/sqrt(3)',
  },
  {
    name: 'octahedron',
    three: 'OctahedronGeometry',
    params: 'radius 1',
    mesh: geometry.octahedron({ radius: 1 }),
    analytic: sdf.octahedron(1),
  },
  {
    name: 'icosahedron',
    three: 'IcosahedronGeometry',
    params: 'radius 1',
    mesh: geometry.icosahedron({ radius: 1 }),
    analytic: sdf.icosahedron(1),
  },
  {
    name: 'dodecahedron',
    three: 'DodecahedronGeometry',
    params: 'radius 1',
    mesh: geometry.dodecahedron({ radius: 1 }),
    analytic: sdf.dodecahedron(1),
    note: 'the dual of the icosahedron, so the two share an inradius of 0.7947 at radius 1',
  },
  {
    name: 'edges',
    three: 'EdgesGeometry',
    params: `the sharp edges of a 1.6 box, as ${ROD} rods`,
    mesh: geometry.edges({ geometry: geometry.box({ width: 1.6, height: 1.6, depth: 1.6 }), thickness: ROD }),
    analytic: sdf.boxFrame([0.8, 0.8, 0.8], ROD / 2),
    note: 'three.js draws lines; a line has no volume, so these are square rods instead',
  },
  {
    name: 'wireframe',
    three: 'WireframeGeometry',
    params: `every edge of an icosahedron, as ${ROD} rods`,
    mesh: geometry.wireframe({ geometry: geometry.icosahedron({ radius: 1 }), thickness: ROD }),
    note: 'edges keeps only the sharp ones; wireframe keeps all of them, flat ones included',
  },
];

// --- the board -------------------------------------------------------------

/**
 * Four pairs across, however many rows that comes to.
 *
 * The two spacings are not the same number and that is the whole of the layout: a pair is
 * separated by less than two pairs are, so the eye groups the mesh with its twin rather
 * than with the shape in the next column along. Everything in the catalogue is about two
 * units across, so 2.5 within a pair and 5.0 between them is 0.4 of a gap against 0.6.
 */
const COLS = 4;
const PITCH_X = 5.0;
const PITCH_Z = 3.0;
const ROWS = Math.ceil(CATALOGUE.length / COLS);
/** Half the separation inside a pair. */
const OFFSET = 1.25;
/** Where a shape's centre sits, which is one shape-radius above its shelf. */
const HEIGHT = 0.35;

/**
 * The default camera looks down `+z`, and looking down `+z` puts `+x` on the *left* of the
 * screen. Reading order is a claim about the screen - the HUD says the baked mesh is the
 * left of each pair - so the layout is mirrored here rather than everywhere it is used.
 */
const MIRROR = -1;

/** Cell centre, in reading order: left to right, then away from the camera. */
const cell = (i: number): [number, number] => [
  MIRROR * ((i % COLS) - (COLS - 1) / 2) * PITCH_X,
  (Math.floor(i / COLS) - (ROWS - 1) / 2) * PITCH_Z,
];

export const start: DemoStart = async ({ canvas, status, hud, flags }) => {
  status('creating device…');
  const game = await Game.create({
    canvas,
    materials: {
      clay: { albedo: [0.78, 0.44, 0.36], roughness: 0.7, emissive: [0, 0, 0], metallic: 0 },
      brass: { albedo: [0.86, 0.66, 0.3], roughness: 0.28, emissive: [0.05, 0.03, 0], metallic: 0.85 },
      stone: { albedo: [0.62, 0.61, 0.58], roughness: 0.85, emissive: [0, 0, 0], metallic: 0 },
    },
    // One slot per entry, at 64 voxels rather than the default 48: the whole demo is a
    // side-by-side against an exact field, and at 48 the argument is about the atlas
    // rather than about the bake. 24 slots of 64^3 is about 50 MB of texture.
    meshes: { resolution: 64, slots: CATALOGUE.length + 2 },
    // Sized to the board and no larger. A volume is a cube whatever the scene is, so the
    // Y range here is mostly air - the price of one field over a wide, flat layout, and
    // cheaper than the alternative of one volume per shape, which would bind two dozen
    // 3D textures into a single trace and blow past what a shader stage can hold.
    //
    // The board is centred in it with the better part of a unit to spare on every side,
    // which is not slack: the field just inside the wall has to stay positive or a ray has
    // no exterior to approach the surface through, and the shelf that ran too close to the
    // near wall came out as a band of speckle rather than as a shelf.
    bounds: { size: 21, origin: [-10.5, -2.4, -10.5] },
  });

  // Baked up front, all of them: the board is assembled once and never changes, so every
  // slot has to be written before the union that refers to them is handed to the builder.
  const baked: BakedMesh[] = [];
  for (const [i, entry] of CATALOGUE.entries()) {
    status(`baking ${i + 1}/${CATALOGUE.length} — ${entry.name}…`);
    baked.push(await game.loadMesh(entry.mesh, { thickness: entry.thickness }));
  }

  game.spawn.sun();
  const camera = game.spawn.camera({ fov: Math.PI / 3 });

  // One shelf per row rather than one plinth per cell. Both look fine; a shelf is six
  // brushes instead of twenty-four, and at the coarsest mip a tile is eleven units wide
  // and would otherwise pull most of the board through the 64-brush per-tile cull.
  const shelves = Array.from({ length: ROWS }, (_unused, row) =>
    sdf
      .roundBox([COLS * PITCH_X / 2 - 0.4, 0.42, PITCH_Z / 2 - 0.18], 0.2)
      .at([0, -1.1, (row - (ROWS - 1) / 2) * PITCH_Z])
      .material('stone'));

  status('baking the board…');
  const level = game.spawn.solid({
    shape: sdf.union(
      ...shelves,
      ...CATALOGUE.flatMap((entry, i) => {
        const [x, z] = cell(i);
        // Centred in its cell when it is on its own, so a shape with no twin does not sit
        // off to one side looking like half of the pair is missing.
        const dx = entry.analytic ? -MIRROR * OFFSET : 0;
        return [
          sdf.mesh(baked[i]!).at([x + dx, HEIGHT, z]).material('clay'),
          entry.analytic?.at([x + MIRROR * OFFSET, HEIGHT, z]).material('brass') ?? sdf.none(),
        ];
      }),
    ),
    // 21 units at 256 voxels is a 0.082 voxel, which is finer than the Claybook demo's by
    // a factor of two and a half. It has to be: the whole board is one field, so the
    // volume is what every silhouette on screen is resampled through, and at anything
    // coarser it would be the volume blurring the comparison rather than the bake.
    resolution: 256,
  });

  Object.assign(globalThis, { game, level, geometry, sdf, CATALOGUE, baked });

  // --- the read-out --------------------------------------------------------

  // A legend by row, because the shapes cannot label themselves: there is no text in a
  // distance field, and reading order on the board is reading order here.
  const legend: [string, string][] = Array.from({ length: ROWS }, (_unused, row) => [
    `row ${row + 1}`,
    CATALOGUE.slice(row * COLS, row * COLS + COLS).map((e) => e.name).join(' · '),
  ]);
  const shells = CATALOGUE.filter((e) => e.thickness).map((e) => e.name);
  const bakeOnly = CATALOGUE.filter((e) => !e.analytic).length;
  const vertices = CATALOGUE.reduce((n, e) => n + e.mesh.positions.length / 3, 0);

  hud([
    ['board', `${CATALOGUE.length} geometries, all at once`],
    ['left · clay', `geometry.* baked into 64³ slots — ${Math.round(vertices / 100) / 10}k vertices`],
    ['right · brass', `sdf.* evaluated exactly — ${CATALOGUE.length - bakeOnly} of them have one`],
    ['bake only', `${bakeOnly} shapes with no closed form worth writing`],
    ['shells', `${shells.join(', ')} enclose no volume, so they bake as thin slabs`],
    ...legend,
    ['keys', 'drag to orbit · scroll to zoom'],
  ]);

  // --- input ---------------------------------------------------------------

  // `#pick=torusKnot` is the one thing the carousel was good for: getting close enough to
  // one pair to see where the bake and the closed form part company.
  const picked = CATALOGUE.findIndex((e) => e.name === flags.get('pick'));
  const [px, pz] = picked < 0 ? [0, 0] : cell(picked);
  const target: readonly [number, number, number] = [px, HEIGHT, pz];

  let yaw = 0;
  // Steeply enough down that the six rows stay six rows. From a low angle the far half of
  // the board foreshortens into a band and the catalogue stops reading as a grid.
  let pitch = picked < 0 ? 0.88 : 0.45;
  let distance = picked < 0 ? 18 : 6;
  let dragging = false;
  // The drift is a convenience, not the demo. The moment somebody turns the camera
  // themselves they have an angle they want, and spinning away from it is rude.
  let manual = flags.get('spin') === 'off';

  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    manual = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => {
    dragging = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
      return;
    }
    yaw -= e.movementX * 0.006;
    pitch = Math.min(1.45, Math.max(-0.2, pitch - e.movementY * 0.005));
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    distance = Math.min(40, Math.max(3.5, distance * Math.exp(e.deltaY * 0.0012)));
  }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  let frames = 0;
  let lastFrames = 0;
  let lastReport = 0;
  game.start(() => {
    frames++;
    // A sway rather than a spin. The board is six rows wide and only reads as rows from
    // roughly the front; a camera that walks all the way round spends most of its time
    // showing the catalogue edge-on, which is a worse default than not moving at all.
    if (!manual) {
      yaw = Math.sin(game.time * 0.18) * 0.4;
    }
    Object.assign(camera, orbit(target, { yaw, pitch, distance, lookHeight: 0 }));

    const now = performance.now();
    if (now - lastReport > 400) {
      const fps = Math.round(((frames - lastFrames) * 1000) / (now - lastReport));
      status(`${canvas.width}x${canvas.height} · ${fps} fps · `
        + `${CATALOGUE.length} geometries in one 256³ field`);
      lastReport = now;
      lastFrames = frames;
    }
  });
};
