import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { SdfBuilder } from '../../src/field/builder.ts';
import { SdfVolume } from '../../src/field/volume.ts';
import { defaultBrushSet } from '../../src/field/brush.ts';
import { compileShape, sdf, type Shape } from '../../src/shape/sdf.ts';
import { unionField, volumeField, type TracedField } from '../../src/trace/field.ts';
import { makeTracer, SweepHit } from '../../src/trace/march.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * What the tracer *spends*, as opposed to where it lands.
 *
 * A wrong `t` shows up as a wrong picture and gets found by eye. A `t` that is right but
 * cost sixty iterations does not: it looks perfect until the scene is big enough that the
 * budget runs out, and then a whole region of the screen turns into sky at one camera
 * angle and comes back at another. So these assert on `steps`, which is the only place
 * that failure mode is visible before it is a hole.
 *
 * The volume is deliberately given the demos' mip depth, because the cost pinned down here
 * scales with it: at five levels the coarsest voxel is sixteen mip-0 voxels wide, and the
 * tracer discounts every step it takes there by 0.87 of one.
 */
let root: TgpuRoot | null = null;
let available = false;

const RESOLUTION = 128;
const WORLD = 16;
/** The volume spans `[-HALF, HALF]` in every axis. */
const HALF = WORLD / 2;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

/** Bakes one shape into a fresh volume. */
function bake(r: TgpuRoot, shape: Shape, options: ConstructorParameters<typeof SdfVolume>[1]): SdfVolume {
  const volume = new SdfVolume(r, options);
  const builder = new SdfBuilder(volume, { brushSet: defaultBrushSet });
  builder.setBrushes(defaultBrushSet.compile(compileShape(shape, () => 0)));

  const encoder = r['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  builder.rebuild(pass);
  pass.end();
  encoder.submit();
  return volume;
}

/** A volume holding one sphere of radius 3 at the origin, five mips deep. */
function scene(r: TgpuRoot): SdfVolume {
  return bake(r, sdf.sphere(3), {
    resolution: RESOLUTION,
    mipLevels: 5,
    worldSize: WORLD,
    origin: [-HALF, -HALF, -HALF],
    label: 'march-test',
  });
}

/**
 * A second, much smaller volume off to one side, at the depth a soft body's collider bake
 * actually gets: 32 voxels is two mip levels, against the scene's five.
 */
function pebble(r: TgpuRoot): SdfVolume {
  return bake(r, sdf.sphere(1).at([5, 0, 0]), {
    resolution: 32,
    worldSize: 4,
    origin: [3, -2, -2],
    label: 'march-test-pebble',
  });
}

interface Ray {
  ro: readonly [number, number, number];
  /** Normalised on the GPU, so these can be written as whole numbers. */
  rd: readonly [number, number, number];
  tMax: number;
  /** Swept sphere radius. Omitted is a plain ray. */
  radius?: number;
}

/** Runs a batch of rays through the real tracer and reads the hit records back. */
async function march(
  r: TgpuRoot,
  field: TracedField,
  rays: readonly Ray[],
  maxSteps: number,
): Promise<{ t: number; hit: number; steps: number }[]> {
  const tracer = makeTracer(field, { maxSteps });
  const n = rays.length;
  const origins = r.createReadonly(d.arrayOf(d.vec4f, n));
  // `w` carries `tMax`, so one buffer covers the rest of the ray.
  const dirs = r.createReadonly(d.arrayOf(d.vec4f, n));
  origins.write(rays.map((s) => d.vec4f(s.ro[0], s.ro[1], s.ro[2], s.radius ?? 0)));
  dirs.write(rays.map((s) => d.vec4f(s.rd[0], s.rd[1], s.rd[2], s.tMax)));
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
      out.$[gid.x] = tracer.sweep(o.xyz, std.normalize(v.xyz), d.f32(0), v.w, o.w, d.f32(0));
    }),
  });
  for (const g of field.groups) {
    pipeline = pipeline.with(g);
  }
  pipeline.dispatchWorkgroups(Math.ceil(n / 64));
  return [...(await out.buffer.read())].map((h) => ({ t: h.t, hit: h.hit, steps: h.steps }));
}

test('flying in from outside the volume is not paid for step by step', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  const [head, corner] = await march(
    root,
    volumeField(scene(root)),
    [
      // Straight at the sphere through the middle of a face.
      { ro: [0, 0, -14], rd: [0, 0, 1], tMax: 60 },
      // In through a corner at an angle, so the flight in is longer than the box is wide.
      { ro: [-12, 6, -12], rd: [12, -6, 12], tMax: 60 },
    ],
    192,
  );

  assert.equal(head!.hit, 1, 'the head-on ray finds the sphere');
  assert.ok(Math.abs(head!.t - 11) < 0.2, `and stops on its surface (t=${head!.t})`);
  // A hierarchical trace crosses fourteen units of nothing in a handful of steps. Anything
  // near fifty means it walked the last unit of the approach at the minimum step size,
  // which is what a saturated exterior sample used to force - and the pre-pass has only
  // sixty-four steps in total, so that is its whole budget spent before it is even inside.
  assert.ok(head!.steps < 20, `in a handful of steps (took ${head!.steps})`);

  assert.equal(corner!.hit, 1, 'the angled ray finds it too');
  assert.ok(corner!.steps < 20, `at the same price (took ${corner!.steps})`);
});

test('a ray starting inside `hitEps` of a surface is still allowed to leave it', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  // `hitEps` is half a mip-0 voxel, so 1/16 of a unit here. Start a twentieth of a unit
  // off the sphere - inside that threshold, but nowhere near touching.
  const near = 3 + 0.05;
  const [out, along, into, contact] = await march(
    root,
    volumeField(scene(root)),
    [
      // Straight away from the sphere. Nothing is in front of this ray at all.
      { ro: [near, 0, 0], rd: [1, 0, 0], tMax: 20 },
      // Away at about sixty degrees off the normal, where a cosine-weighted AO cone puts
      // most of its samples. Not tangentially: a ray grazing a sphere of this radius at
      // this clearance stays inside `hitEps` for a long stretch, and calling that a hit
      // is what `hitEps` is *for*.
      { ro: [near, 0, 0], rd: [1, 2, 0], tMax: 20 },
      // And back into it, which must still land on the surface where it always did.
      { ro: [near, 0, 0], rd: [-1, 0, 0], tMax: 20 },
      // A swept sphere already overlapping *is* a contact, and reporting it at t=0 is the
      // whole point of the query - so `radius` is not subject to the same ramp.
      { ro: [3.5, 0, 0], rd: [1, 0, 0], tMax: 20, radius: 0.75 },
    ],
    192,
  );

  // Believing `hitEps` at t=0 asks "is the origin near a surface", which has the same
  // answer for every direction - so a shading point beside a ball resting on the floor
  // reported *every* AO cone and shadow ray occluded, and the contact went hard black.
  assert.equal(out!.hit, 0, `leaving along the normal escapes (t=${out!.t})`);
  assert.equal(along!.hit, 0, `and so does leaving at an angle (t=${along!.t})`);
  assert.equal(into!.hit, 1, 'heading back in still hits');
  assert.ok(Math.abs(into!.t - 0.05) < 0.07, `on the surface it came from (t=${into!.t})`);
  assert.equal(contact!.hit, 1, 'a swept sphere already overlapping reports contact');
  assert.ok(contact!.t < 0.05, `at once (t=${contact!.t})`);
});

test('a ray that enters almost parallel to a face still crosses the volume', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  // Skims four tenths above the +y face and sinks into it at one unit in twenty, so it
  // spends eight units of flight within a coarse voxel of the wall before it is inside,
  // and clears the sphere comfortably once it is. Approaching a plane at a grazing angle
  // is the case sphere tracing is worst at, and a field that knows its own bounds should
  // not be marching towards them at all.
  const [graze] = await march(
    root,
    volumeField(scene(root)),
    [{ ro: [-9, HALF + 0.4, 6], rd: [20, -1, 0], tMax: 40 }],
    192,
  );

  assert.equal(graze!.hit, 0, 'it passes over the sphere and out the far side');
  assert.ok(
    graze!.t >= 40 - 1e-3,
    `reaching the end of the interval rather than stalling short of it (t=${graze!.t})`,
  );
  assert.ok(graze!.steps < 192, `without exhausting the step budget (took ${graze!.steps})`);
});

test('a shallow volume joins a deep one without dragging the trace down to it', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }
  // The pairing every soft body creates: a 32-voxel collider bake, two mips deep, unioned
  // with a scene four times its depth so that shadow and AO rays can see the body. Which
  // hierarchy the union adopts decides both halves of this test - take the shallower one
  // and the scene is traced entirely at mip 0, take the deeper one without clamping the
  // small volume's queries and its band is read as four times too wide and rays walk
  // straight through it.
  const field = unionField(volumeField(scene(root)), volumeField(pebble(root)));
  assert.equal(field.maxMip, 4, 'the union keeps the deeper hierarchy');

  const [atPebble, atSphere, past] = await march(
    root,
    field,
    [
      // Down the x axis at the small volume's sphere, which spans x in [4, 6].
      { ro: [12, 0, 0], rd: [-1, 0, 0], tMax: 30 },
      // The same line continued: past the small sphere is the big one, ending at x = 3.
      { ro: [12, 1.6, 0], rd: [-1, 0, 0], tMax: 30 },
      // Clear of both, the length of the scene.
      { ro: [12, 5, 0], rd: [-1, 0, 0], tMax: 30 },
    ],
    192,
  );

  assert.equal(atPebble!.hit, 1, 'the small volume is hit');
  assert.ok(
    Math.abs(atPebble!.t - 6) < 0.25,
    `on its surface rather than through it (t=${atPebble!.t}, expected 6)`,
  );

  // 1.6 above the axis misses the small sphere (radius 1) and meets the big one (radius 3)
  // at x = sqrt(9 - 2.56) = 2.538.
  assert.equal(atSphere!.hit, 1, 'and the scene behind it still is too');
  assert.ok(
    Math.abs(atSphere!.t - (12 - 2.538)) < 0.25,
    `at the right distance (t=${atSphere!.t}, expected 9.46)`,
  );

  assert.equal(past!.hit, 0, 'a ray clear of both escapes');
  // The point of keeping the deeper hierarchy: crossing 30 units of mostly-empty scene is
  // a handful of coarse steps. Capped at the small volume's two levels it would be dozens,
  // and a 64-step shadow ray would not reach the light.
  assert.ok(past!.steps < 25, `in coarse steps, not fine ones (took ${past!.steps})`);
});
