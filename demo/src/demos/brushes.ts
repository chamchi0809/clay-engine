import { Game, orbit, sdf } from '@clay/engine';
import type { DemoStart } from '../shell.ts';
import { hexPrism, rock, TETRA_OBJ } from './brush-shapes.ts';

/**
 * Two ways to get a shape the engine does not ship.
 *
 * The columns are a *custom primitive*: a distance function written in the game's own
 * source, handed to `Game.create({ brushes })`, and compiled into the brush fold. The rock
 * and the tetrahedron are *baked meshes*: triangles turned into a signed distance field on
 * the GPU at load time and stored in an atlas slot.
 *
 * The two arrive by very different routes and are indistinguishable afterwards. Both are
 * brushes - placed with `.at()`, spun with `.rotate()`, resized with `.scale()`, given a
 * material, restricted with `.only()`, and (the last thing on screen here) used as the
 * cutter in a `cut`.
 *
 * A game that only ever places spheres and boxes needs none of this, and there is a whole
 * separate demo of that. This is the escape hatch, and it is deliberately the only demo
 * that writes a line of GPU code.
 */
export const start: DemoStart = async ({ canvas, status, hud }) => {
  status('creating device…');
  const game = await Game.create({
    canvas,
    materials: {
      clay: { albedo: [0.78, 0.44, 0.36], roughness: 0.75, emissive: [0, 0, 0], metallic: 0 },
      stone: { albedo: [0.66, 0.64, 0.6], roughness: 0.85, emissive: [0, 0, 0], metallic: 0 },
      brass: { albedo: [0.86, 0.66, 0.3], roughness: 0.3, emissive: [0.05, 0.03, 0], metallic: 0.85 },
    },
    // A custom kind cannot be added later: the fold is compiled into every pipeline that
    // bakes or edits a field, and a shader that exists cannot grow a branch.
    brushes: { hexPrism },
    // Same deal, and the reason this is opt-in rather than always there: the atlas is one
    // fixed-size 3D texture bound into those same pipelines. `{}` would do - these are the
    // defaults spelled out. Eight slots at 48 voxels is about 3.5 MB.
    meshes: { resolution: 48, slots: 8 },
    bounds: { size: 24, origin: [-12, -4, -12] },
  });

  status('baking meshes…');
  // 80 triangles against 48^3 voxels, brute force, on the GPU. Both bakes together are a
  // few milliseconds - the await is here because the field has to exist before a shape
  // that references its slot can be built, not because it is slow.
  const stone = await game.loadMesh(rock());
  const spike = await game.loadMesh(TETRA_OBJ);

  game.spawn.sun();
  const camera = game.spawn.camera({ fov: Math.PI / 3 });

  // Six columns in a ring. `sdf.custom` passes `size` and `radius` straight through to the
  // distance function, so what they mean is whatever `hexPrism` says they mean: here a
  // flat-to-flat radius, a half-height, and an edge round.
  const columns = sdf.union(
    ...Array.from({ length: 6 }, (_unused, i) => {
      const angle = (i / 6) * Math.PI * 2;
      const height = 1.1 + 0.55 * (i % 3);
      return sdf
        .custom('hexPrism', { size: [0.55, height, 0], radius: 0.08 })
        .at([Math.sin(angle) * 6.2, height - 0.9, Math.cos(angle) * 6.2])
        .material('brass');
    }),
  );

  const plate = sdf.roundBox([8.4, 1, 8.4], 0.4).at([0, -1.7, 0]).material('stone');
  const slab = sdf.roundBox([5.6, 0.45, 5.6], 0.3).at([0, -0.35, 0]).material('clay');

  // `sdf.mesh` is the only line that knows a mesh was involved. From here it is a brush.
  const boulder = sdf.mesh(stone).scale(1.9).rotate([0.2, 0.7, 0.1]).at([-2.4, 1.4, 1.6])
    .material('stone');
  const obelisk = sdf.mesh(spike).scale(1.6).at([2.8, 1.1, -1.2]).material('brass');

  const level = game.spawn.solid({
    // The bite: the tetrahedron again, this time as the thing being subtracted, pressed
    // into the clay slab and told to leave the stone plate under it alone. A cutter is a
    // brush with `op: 'cut'` and nothing else different, which is why a baked mesh gets to
    // be one for free.
    shape: sdf.cut(
      sdf.union(plate, slab, columns, boulder, obelisk),
      sdf.mesh(spike).scale(2.6).rotate([2.6, 0.5, 0]).at([1.4, 0.5, 3.1]).only('clay'),
    ),
    resolution: 128,
  });

  // Console handles, unconditionally: this page exists to be poked at. Try
  // `level.cut(sdf.mesh(stone).scale(0.6).at([0, 0, 0]).only('clay'))`.
  Object.assign(globalThis, { game, level, stone, spike, sdf });

  // Nothing here is interactive, so the panel is a legend rather than a key list: which
  // thing on screen came by which of the two routes.
  hud([
    ['columns', 'custom primitive — hexPrism, compiled into the brush fold'],
    ['boulder', `baked mesh — ${stone.triangleCount} generated triangles`],
    ['obelisk', `baked mesh — ${spike.triangleCount} triangles from an .obj`],
    ['bite', 'the same baked mesh as a cutter, restricted to the clay'],
  ]);

  let frames = 0;
  let lastReport = 0;
  let lastFrames = 0;
  game.start(() => {
    frames++;
    // Slow orbit. Nothing in the scene moves, so this is also the honest test of whether
    // the bakes are stable: a wobbling boulder means the field is wrong, not the camera.
    Object.assign(
      camera,
      orbit([0, 0.6, 0], {
        yaw: game.time * 0.16,
        pitch: 0.34 + Math.sin(game.time * 0.11) * 0.12,
        distance: 15,
      }),
    );

    const now = performance.now();
    if (now - lastReport > 400) {
      const fps = Math.round(((frames - lastFrames) * 1000) / (now - lastReport));
      status(`${canvas.width}x${canvas.height} · ${fps} fps · `
        + `1 custom primitive · 2 baked meshes (${stone.triangleCount} + ${spike.triangleCount} tris)`);
      lastReport = now;
      lastFrames = frames;
    }
  });
};
