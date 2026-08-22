import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { SdfBuilder } from '../../src/field/builder.ts';
import { SdfVolume } from '../../src/field/volume.ts';
import { defaultBrushSet } from '../../src/field/brush.ts';
import { compileShape, sdf } from '../../src/shape/sdf.ts';
import { volumeField } from '../../src/trace/field.ts';
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

/** A volume holding one sphere of radius 3 at the origin, five mips deep. */
function scene(r: TgpuRoot): SdfVolume {
  const volume = new SdfVolume(r, {
    resolution: RESOLUTION,
    mipLevels: 5,
    worldSize: WORLD,
    origin: [-HALF, -HALF, -HALF],
    label: 'march-test',
  });
  const builder = new SdfBuilder(volume, { brushSet: defaultBrushSet });
  builder.setBrushes(defaultBrushSet.compile(compileShape(sdf.sphere(3), () => 0)));

  const encoder = r['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  builder.rebuild(pass);
  pass.end();
  encoder.submit();
  return volume;
}

interface Ray {
  ro: readonly [number, number, number];
  /** Normalised on the GPU, so these can be written as whole numbers. */
  rd: readonly [number, number, number];
  tMax: number;
}

/** Runs a batch of rays through the real tracer and reads the hit records back. */
async function march(
  r: TgpuRoot,
  volume: SdfVolume,
  rays: readonly Ray[],
  maxSteps: number,
): Promise<{ t: number; hit: number; steps: number }[]> {
  const field = volumeField(volume);
  const tracer = makeTracer(field, { maxSteps });
  const n = rays.length;
  const origins = r.createReadonly(d.arrayOf(d.vec4f, n));
  // `w` carries `tMax`, so one buffer covers the rest of the ray.
  const dirs = r.createReadonly(d.arrayOf(d.vec4f, n));
  origins.write(rays.map((s) => d.vec4f(s.ro[0], s.ro[1], s.ro[2], 0)));
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
      out.$[gid.x] = tracer.ray(o.xyz, std.normalize(v.xyz), d.f32(0), v.w);
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
    scene(root),
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
    scene(root),
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
