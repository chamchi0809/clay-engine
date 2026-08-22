import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { SdfBuilder } from '../../src/field/builder.ts';
import { SdfVolume } from '../../src/field/volume.ts';
import { defaultBrushSet } from '../../src/field/brush.ts';
import { compileShape, sdf } from '../../src/shape/sdf.ts';
import { SplatField, type ParticleCloud } from '../../src/sim/splat.ts';
import { unionField, volumeField, type TracedField } from '../../src/trace/field.ts';
import { makeTracer, SweepHit } from '../../src/trace/march.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * The soft-body-as-occluder pairing, end to end: a brush-built world unioned with the
 * splatted shell of a body, at the sizes the clay demo actually uses.
 *
 * The sizes are the point. A body's bake box is sized for the longest shape it can morph
 * into, so most of it is empty air around the body, and every voxel of that air reads
 * *saturated* - `band * voxel`, a number far smaller than the emptiness it stands for.
 * A union that lets that number decide the step, or the hit, draws the box on the floor
 * instead of the body.
 */
let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

/** Top of the plate. */
const FLOOR = -1;
const BALL_R = 0.95;
/** Centre of the body, resting on the floor. */
const BALL: readonly [number, number, number] = [0, FLOOR + BALL_R, 0];
/** `reach * 2 + spacing * 2` for the demo's ball: `reach` 1.5, spacing `radius / 8.5`. */
const BOX = 1.5 * 2 + (BALL_R / 8.5) * 2;
/** `SoftBody`'s own `box / resolution * 1.15`. */
const SPLAT_R = (BOX / Math.ceil(BOX / (BALL_R / 8.5))) * 1.15;
/** The demo's world: 24 units at 128 voxels. */
const WORLD = 24;
const WORLD_RES = 128;

function world(r: TgpuRoot): SdfVolume {
  const volume = new SdfVolume(r, {
    resolution: WORLD_RES,
    worldSize: WORLD,
    origin: [-12, -4, -12],
    label: 'occluder-world',
  });
  const builder = new SdfBuilder(volume, { brushSet: defaultBrushSet });
  builder.setBrushes(
    defaultBrushSet.compile(compileShape(sdf.box([6, 1, 6]).at([0, FLOOR - 1, 0]), () => 0)),
  );
  const encoder = r['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  builder.rebuild(pass);
  pass.end();
  encoder.submit();
  return volume;
}

/**
 * The body: a shell of particles on the ball's surface, which is what surface nets hands
 * the splat. Not a solid ball - the inside of a body is empty in its own bake too, and
 * that emptiness is saturated exactly like the air around it.
 */
function body(r: TgpuRoot, oriented = true): SplatField {
  const spacing = BALL_R / 8.5;
  const n = Math.ceil((4 * Math.PI * BALL_R * BALL_R) / (spacing * spacing * 0.87));
  const points: d.v4f[] = [];
  // Fibonacci sphere: even coverage without poles, so the shell has no seam of holes.
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * i) / (n - 1);
    const rad = Math.sqrt(Math.max(0, 1 - y * y));
    const a = golden * i;
    points.push(
      d.vec4f(
        BALL[0] + Math.cos(a) * rad * BALL_R,
        BALL[1] + y * BALL_R,
        BALL[2] + Math.sin(a) * rad * BALL_R,
        1,
      ),
    );
  }
  const pos = r.createReadonly(d.arrayOf(d.vec4f, n), points);
  const centre = d.vec3f(BALL[0], BALL[1], BALL[2]);
  const cloud: ParticleCloud = {
    capacity: n,
    positionAt: (i: number) => {
      'use gpu';
      return pos.$[i].xyz;
    },
    liveAt: (_i: number) => {
      'use gpu';
      return true;
    },
    // Exact here, because the shell is a sphere. A real body's normals come from the
    // field gradient at extraction and go stale as it deforms; what this pins down is
    // where the bake puts the surface when the normals are right.
    ...(oriented
      ? {
        normalAt: (i: number) => {
          'use gpu';
          return std.normalize(pos.$[i].xyz - centre);
        },
      }
      : {}),
  };
  const splat = new SplatField(r, cloud, {
    radius: SPLAT_R,
    resolution: 32,
    band: 4,
    worldSize: BOX,
    label: 'occluder-body',
  });
  splat.setCenter(BALL);
  const encoder = r['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  splat.bake(pass);
  pass.end();
  encoder.submit();
  return splat;
}

interface Probe {
  ro: readonly [number, number, number];
  rd: readonly [number, number, number];
  tMax: number;
  aperture?: number;
}

/** Runs a batch of shadow/AO-shaped sweeps and reads the hit records back. */
async function sweep(
  r: TgpuRoot,
  field: TracedField,
  probes: readonly Probe[],
  maxSteps: number,
): Promise<{ t: number; hit: number; steps: number }[]> {
  const tracer = makeTracer(field, { maxSteps });
  const n = probes.length;
  const origins = r.createReadonly(d.arrayOf(d.vec4f, n));
  // `w` carries the cone aperture, and the direction buffer's `w` carries `tMax`.
  const dirs = r.createReadonly(d.arrayOf(d.vec4f, n));
  origins.write(probes.map((s) => d.vec4f(s.ro[0], s.ro[1], s.ro[2], s.aperture ?? 0)));
  dirs.write(probes.map((s) => d.vec4f(s.rd[0], s.rd[1], s.rd[2], s.tMax)));
  const out = r.createMutable(d.arrayOf(SweepHit, n));

  let pipeline = r.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= d.u32(n)) {
        return;
      }
      const o = origins.$[gid.x];
      const v = dirs.$[gid.x];
      out.$[gid.x] = tracer.sweep(o.xyz, std.normalize(v.xyz), d.f32(0), v.w, d.f32(0), o.w);
    }),
  });
  for (const g of field.groups) {
    pipeline = pipeline.with(g);
  }
  pipeline.dispatchWorkgroups(Math.ceil(n / 64));
  return [...(await out.buffer.read())].map((h) => ({ t: h.t, hit: h.hit, steps: h.steps }));
}

/** A shading point on the floor, biased out along its normal exactly as the shader does. */
function onFloor(x: number, z: number): readonly [number, number, number] {
  return [x, FLOOR + WORLD / WORLD_RES, z];
}

/** The demo's sun, near enough: high and off to one side. */
const SUN = [0.45, 1, 0.3] as const;

/**
 * Radius of the baked shell along each direction, measured the way the renderer would:
 * fire inward from a point clear of the surface and see where the trace stops.
 */
async function shellRadii(
  r: TgpuRoot,
  splat: SplatField,
  dirs: readonly (readonly [number, number, number])[],
): Promise<number[]> {
  // 1.5 out is clear of the shell under either bake and still inside the box, so the
  // sweep starts in the volume rather than flying in through its bounds.
  const from = 1.5;
  const hits = await sweep(
    r,
    splat.field,
    dirs.map((v) => {
      const len = Math.hypot(v[0], v[1], v[2]);
      return {
        ro: [
          BALL[0] + (v[0] / len) * from,
          BALL[1] + (v[1] / len) * from,
          BALL[2] + (v[2] / len) * from,
        ] as const,
        rd: [-v[0], -v[1], -v[2]] as const,
        tMax: 3,
      };
    }),
    64,
  );
  return hits.map((h) => (h.hit === 1 ? from - h.t : Number.NaN));
}

test('normals put the baked shell on the particles instead of a radius outside them', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  // Not a tuning question - a body's particles *are* its mesh vertices, so a shell that
  // stands `SPLAT_R` proud of them is an occluder bigger than the thing being drawn. The
  // silhouette then sits inside its own fully-occluded zone, and near a contact, where AO
  // is on a cliff, a 13% radius error is not a 13% error in the picture: a clay ball
  // resting on the floor went black around the base while a traced sphere of the same
  // radius beside it did not.
  const dirs = [
    [1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0.6, 0.5, -0.62],
  ] as const;
  const [plain, facets] = await Promise.all([
    shellRadii(root, body(root, false), dirs),
    shellRadii(root, body(root, true), dirs),
  ]);

  // Half the splat radius separates the two cleanly, and the bake's own voxel is
  // BOX/32 = 0.10, so neither bound is asking for sub-voxel accuracy.
  const slack = SPLAT_R / 2;
  for (let i = 0; i < dirs.length; i++) {
    assert.ok(
      plain[i]! > BALL_R + slack,
      `spheres bulge past the particles along ${dirs[i]} (r=${plain[i]}, particles at ${BALL_R})`,
    );
    assert.ok(
      Math.abs(facets[i]! - BALL_R) < slack,
      `discs land on them along ${dirs[i]} (r=${facets[i]}, particles at ${BALL_R})`,
    );
  }
});

test('a body does not shadow the floor with its bake box', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  const field = unionField(volumeField(world(root)), body(root).field);
  const edge = BOX / 2 - 0.1;

  const [up, sun, corner, outside, under] = await sweep(
    root,
    field,
    [
      // Inside the bake box, clear of the ball, straight up: the first place the box
      // shows up as a hard-edged square on the floor.
      { ro: onFloor(1.35, 0), rd: [0, 1, 0], tMax: 24 },
      // The same point towards the sun, which is the ray the picture actually casts.
      { ro: onFloor(1.35, 0), rd: SUN, tMax: 24 },
      // The far corner of the box, where the body is furthest away.
      { ro: onFloor(edge, edge), rd: SUN, tMax: 24 },
      // Outside the box entirely - the control, and what the rest of the floor looks like.
      { ro: onFloor(edge + 0.6, edge + 0.6), rd: SUN, tMax: 24 },
      // And one really under the ball, to prove the body casts anything at all.
      { ro: onFloor(0.3, 0), rd: [0, 1, 0], tMax: 24 },
    ],
    64,
  );

  assert.equal(outside!.hit, 0, 'floor clear of the box is lit');
  assert.equal(under!.hit, 1, 'floor under the body is shadowed');
  assert.equal(up!.hit, 0, `floor beside the body is lit (t=${up!.t}, steps=${up!.steps})`);
  assert.equal(sun!.hit, 0, `and lit towards the sun too (t=${sun!.t}, steps=${sun!.steps})`);
  assert.equal(corner!.hit, 0, `as is the corner of the box (t=${corner!.t}, steps=${corner!.steps})`);
});

test('the wall of a bake box is not a surface from either side of it', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  const field = unionField(volumeField(world(root)), body(root).field);
  const half = BOX / 2;
  const wall = BALL[0] + half;

  // Straddling the wall in both directions and at both signs of `dBox`. Outside it the
  // body answers with the distance to its own box, which here is a millimetre - and a
  // millimetre is only safe as long as it stays paired with the saturation flag that
  // came with it. On the floor the world's own reading is *not* saturated, so this is
  // the exact pair that used to print the box on the ground.
  const probes = [-0.02, 0.02, 0.1].flatMap((offset) => [
    { ro: onFloor(wall + offset, 0), rd: [0, 1, 0] as const, tMax: 24 },
    { ro: onFloor(wall + offset, 0), rd: SUN, tMax: 24 },
  ]);
  const hits = await sweep(root, field, probes, 64);

  for (let i = 0; i < hits.length; i++) {
    assert.equal(
      hits[i]!.hit,
      0,
      `probe ${i} at the box wall is lit (t=${hits[i]!.t}, steps=${hits[i]!.steps})`,
    );
  }
});

test('a body does not occlude the floor with its bake box', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  const field = unionField(volumeField(world(root)), body(root).field);
  const edge = BOX / 2 - 0.1;

  // The AO cone as `makeShading` casts it: narrow, 2.5 units long, 32 steps.
  const [inBox, corner, outside] = await sweep(
    root,
    field,
    [
      { ro: onFloor(1.35, 0), rd: [0, 1, 0], tMax: 2.5, aperture: 0.08 },
      { ro: onFloor(edge, edge), rd: [0, 1, 0], tMax: 2.5, aperture: 0.08 },
      { ro: onFloor(edge + 0.6, edge + 0.6), rd: [0, 1, 0], tMax: 2.5, aperture: 0.08 },
    ],
    32,
  );

  assert.equal(outside!.hit, 0, 'floor clear of the box is open');
  assert.equal(inBox!.hit, 0, `floor beside the body is open (t=${inBox!.t}, steps=${inBox!.steps})`);
  assert.equal(corner!.hit, 0, `and so is the corner of the box (t=${corner!.t}, steps=${corner!.steps})`);
});
