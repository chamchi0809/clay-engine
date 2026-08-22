import { Game, orbit, sdf } from '@clay/engine';
import type { DemoStart } from '../shell.ts';
import { pages, shapeNames, shapes, type ShapeName } from './levels.ts';

const BOUNDS = { size: 24, origin: [-12, -4, -12] as [number, number, number] };

/**
 * The Claybook clone: roll a ball of clay through three levels, denting everything it
 * touches, with water that pools where it lands and erodes what it runs over.
 *
 * `#debug=ao`, `#alpha=1`, `#aoSteps=4`, `#page=2` - freeze a lighting term, disable
 * temporal accumulation, retune a knob, or boot straight into a page. Diagnosing a
 * raymarcher without these is guesswork.
 *
 * `#opacity=1` makes the water solid again, which is the A/B every transparency artefact
 * wants: if it is still there with solid water, transparency is not what is wrong.
 */
export const start: DemoStart = async ({ canvas, status, hud, flags }) => {
  const num = (k: string) => (flags.has(k) ? Number(flags.get(k)) : undefined);
  const game = await Game.create({
    canvas,
    materials: {
      clay: { albedo: [0.78, 0.44, 0.36], roughness: 0.75, emissive: [0, 0, 0], metallic: 0 },
      stone: { albedo: [0.66, 0.64, 0.6], roughness: 0.85, emissive: [0, 0, 0], metallic: 0 },
      gold: { albedo: [0.92, 0.74, 0.28], roughness: 0.25, emissive: [0.12, 0.08, 0.01], metallic: 0.9 },
      // The one see-through material in the game. `opacity` is what puts the liquid in
      // the transparency layer at all; `absorption` is why a puddle is clear at its
      // feathered edge and deep blue in the middle of the pool, since it is absorbed per
      // unit travelled rather than applied as a flat tint.
      water: {
        albedo: [0.24, 0.5, 0.78],
        // 0.22, not the 0.06 real water has: the surface is a field of small spheres, so
        // at mirror roughness a large fraction of pixels sit near the sun's specular peak
        // and the pool reads as white haze rather than as water. Spreading the highlight
        // is the cheap fix; the expensive one is a smoother surface bake.
        roughness: 0.22,
        metallic: 0.2,
        opacity: num('opacity') ?? 0.2,
        ior: 1.33,
        absorption: num('absorption') ?? 1.4,
      },
    },
    bounds: BOUNDS,
    shading: {
      debug: (flags.get('debug') ?? 'off') as 'off',
      taaAlpha: num('alpha'),
      aoSteps: num('aoSteps'),
      aoDistance: num('aoDistance'),
      aoAperture: num('aoAperture'),
      aoMinMip: num('aoMinMip'),
      shadowSteps: num('shadowSteps'),
      shadowMinMip: num('shadowMinMip'),
      filterRadius: num('filterRadius'),
      taaClamp: num('taaClamp'),
    },
    transparency: {
      // `#tdebug=thickness|refraction|transmitted|surface|normal|material`.
      debug: (flags.get('tdebug') ?? 'off') as 'off',
      // The deepest the water ever pools on a page. Both what absorption assumes when
      // there is no floor behind the liquid to measure against - a drop falling past the
      // edge of the plate - and the cap on what is measured, without which the falling
      // column reads as metres of water wherever a far wall happens to be behind it.
      thickness: num('thickness') ?? 1.2,
    },
  });

  game.spawn.sun();
  const camera = game.spawn.camera({ fov: Math.PI / 3 });
  const level = game.spawn.solid({ shape: pages[0]!.shape, resolution: 128 });
  const ball = game.spawn.softBody({
    shape: shapes.ball.material('clay'),
    position: pages[0]!.spawn,
    // 0.55 is where an impact is legible without the ball sagging under its own weight;
    // 0.85 looks rigid. The plasticity is what makes it read as clay rather than rubber:
    // a third of every dent stays, per second of contact.
    stiffness: 0.55,
    plasticity: 0.35,
    // The rod is the longest thing the ball ever becomes (0.95 + 0.48 of reach), and
    // the extraction box is fixed at spawn, so it has to be sized for the rod here.
    reach: 1.5,
  });
  const water = game.spawn.fluid({ material: 'water', bakeResolution: 128 });

  // Console handles. A raymarcher is impossible to debug from the outside: the only way
  // to tell a bad field from a bad shader from a bad solver is to poke at the live
  // objects (`game.probe(a, b)` reads the collider field back along a line). Dev only -
  // a shipped page has no business exposing its internals on `window`.
  if (import.meta.env.DEV) {
    Object.assign(globalThis, { game, level, ball, water, sdf });
  }

  // --- input --------------------------------------------------------------
  const down = new Set<string>();
  let yaw = 2.3;
  let pitch = 0.42;
  let dragging = false;
  addEventListener('keydown', (e) => {
    if (e.repeat) {
      return;
    }
    down.add(e.code);
    if (e.code === 'Digit1' || e.code === 'Digit2' || e.code === 'Digit3') {
      morph(shapeNames[Number(e.code.slice(5)) - 1]!);
    } else if (e.code === 'KeyR') {
      load(page);
    } else if (e.code === 'KeyN') {
      load((page + 1) % pages.length);
    } else if (e.code === 'KeyF') {
      tapOn = !tapOn;
    }
    if (e.code === 'Space') {
      e.preventDefault();
    }
  });
  addEventListener('keyup', (e) => down.delete(e.code));
  addEventListener('blur', () => down.clear());
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => {
    dragging = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
      return;
    }
    yaw -= e.movementX * 0.005;
    pitch = Math.min(1.25, Math.max(-0.15, pitch - e.movementY * 0.004));
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // --- level state --------------------------------------------------------
  let page = 0;
  let shape: ShapeName = 'ball';
  let tapOn = false;
  let cleared = 0;
  let jumpAt = -1;
  /** Where the last trail imprint was stamped. */
  let trail: [number, number, number] = [0, -1e3, 0];
  let frames = 0;
  let lastFrames = 0;
  let lastReport = 0;

  /**
   * The ball's own shape, where it is and how it is turned. A sphere would do for the
   * ball and be wrong for everything else: the rod has to leave a slot, not a hole, or
   * morphing stops meaning anything.
   *
   * `turn` before `at`, because the fitted rotation is about the centre of mass and
   * `at` is what puts that centre in the world.
   */
  const imprint = () =>
    shapes[shape].turn(ball.rotation).at([...ball.position]).smooth(0.25).only('clay');

  const morph = (next: ShapeName) => {
    if (next === shape) {
      return;
    }
    shape = next;
    ball.morph(shapes[next].material('clay'));
  };

  const load = (n: number) => {
    page = n;
    level.shape = pages[n]!.shape;
    shape = 'ball';
    ball.morph(shapes.ball.material('clay'));
    ball.setPosition(pages[n]!.spawn);
    tapOn = false;
  };
  load(Math.min(pages.length - 1, Math.max(0, num('page') ?? 0)));

  game.start((_dt) => {
    const p = pages[page]!;
    frames++;
    const eye = ball.smoothPosition;

    // --- camera-relative driving ------------------------------------------
    const fx = Math.sin(yaw);
    const fz = Math.cos(yaw);
    let mx = 0;
    let mz = 0;
    if (down.has('KeyW')) {
      mx += fx;
      mz += fz;
    }
    if (down.has('KeyS')) {
      mx -= fx;
      mz -= fz;
    }
    if (down.has('KeyA')) {
      mx += fz;
      mz -= fx;
    }
    if (down.has('KeyD')) {
      mx -= fz;
      mz += fx;
    }
    const len = Math.hypot(mx, mz);
    if (len > 0) {
      ball.addForce([(mx / len) * 30, 0, (mz / len) * 30]);
    }
    // The engine does not know what a jump is, and shouldn't: this is game design.
    // ponytail: "grounded" is inferred from the readback velocity plus a cooldown,
    // because the solver does not report contacts per body. A contact count on the body
    // would make this exact; a cooldown makes it good enough to platform with.
    const grounded = Math.abs(ball.velocity[1]) < 3.2 && frames - jumpAt > 18;
    if (down.has('Space') && grounded) {
      ball.addForce([0, 7.5, 0], 'velocity');
      jumpAt = frames;
    }

    // --- water ------------------------------------------------------------
    // 8 drops every 4 frames = 120/s, which is what a 0.6-radius nozzle running at
    // 5 m/s can actually carry away at this spacing. Emitting every frame buries each
    // batch inside the last one and the pressure term turns the tap into a firehose.
    if (tapOn && frames % 4 === 0) {
      water.emit(8, p.tap, [0, -5, 0], 0.6);
    }

    // --- the trail --------------------------------------------------------
    // The clay keeps the shape of whatever rolled over it. Only the clay: `.only('clay')`
    // is why the ball dents the blobs and the dam but stops dead on the stone plate
    // under them, which is the whole reason the mechanic is playable rather than a way
    // to burrow out of the level.
    //
    // Distance-gated, not frame-gated: standing still must not keep carving the same
    // spot deeper, or the ball sinks where it is parked instead of leaving a groove.
    if (grounded && flags.get('trail') !== 'off') {
      const [tx, ty, tz] = ball.position;
      if (Math.hypot(tx - trail[0], ty - trail[1], tz - trail[2]) > ball.radius * 0.35) {
        trail = [tx, ty, tz];
        level.cut(imprint());
      }
    }

    // --- runtime edits: digging and erosion -------------------------------
    if (down.has('KeyQ') && frames % 4 === 0) {
      level.cut(imprint().scale(1.15));
      trail = [...ball.position];
    }
    if (down.has('KeyE') && frames % 4 === 0) {
      level.add(sdf.sphere(ball.radius * 0.9).at([...ball.position]).material('clay').smooth(0.4));
    }
    if (tapOn && frames % 6 === 0) {
      void erode();
    }

    // --- goal / respawn ---------------------------------------------------
    const at = ball.position;
    const dg = Math.hypot(at[0] - p.goal[0], at[1] - p.goal[1], at[2] - p.goal[2]);
    if (dg < p.goalRadius) {
      cleared++;
      load((page + 1) % pages.length);
    } else if (at[1] < BOUNDS.origin[1] - 1) {
      ball.setPosition(p.spawn);
    }

    // --- camera -----------------------------------------------------------
    Object.assign(camera, orbit(eye, { yaw, pitch, distance: 8.5, height: 1.2, lookHeight: 0.4 }));

    const now = performance.now();
    if (now - lastReport > 400) {
      const fps = Math.round(((frames - lastFrames) * 1000) / (now - lastReport));
      status(`${canvas.width}x${canvas.height} · ${fps} fps · ${water.count} drops`);
      hud([
        ['page', `${page + 1}/${pages.length} — ${p.name}`],
        ['goal', p.hint],
        ['shape', `${shape}${ball.morphing ? ' …' : ''}`],
        ['water', tapOn ? 'flowing (F)' : 'off (F)'],
        ['cleared', String(cleared)],
        ['keys', 'WASD roll · Space hop · 1/2/3 morph · Q dig · E add · F water · R reset · N next · drag to orbit'],
      ]);
      lastReport = now;
      lastFrames = frames;
    }
  });

  /**
   * Fluid erosion: the sim reports where water is grinding on the terrain, and each such
   * point becomes a small subtractive edit. Only fast water cuts, so a standing pool
   * does not eat the floor it is sitting in.
   */
  const erode = async () => {
    // Only the fastest few cut, and only a handful per tick: an edit rebuilds every tile
    // it touches, so an unbounded fan-out of tiny cuts is a frame spike.
    const hits = (await water.contacts())
      .filter((c) => c.speed > 2.2)
      .sort((a, b) => b.speed - a.speed)
      .slice(0, 6);
    for (const c of hits) {
      level.cut(sdf.sphere(0.3).at(c.position).smooth(0.25).only('clay'));
    }
  };
};
