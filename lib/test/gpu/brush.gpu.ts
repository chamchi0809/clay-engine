import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { Brush, BrushSet, defaultBrushSet, type BrushDesc } from '../../src/field/brush.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * The brush fold, evaluated on the real driver.
 *
 * `field/brush.ts` is the one file where a wrong sign or a swapped axis is invisible: every
 * primitive still produces *a* surface, just not the one asked for, and the bake, the tracer
 * and the collider all agree on the wrong answer. So the primitives are checked against
 * distances worked out by hand, at points chosen to catch exactly those mistakes - inside,
 * outside, on the surface, and off-axis where a rotation would show.
 */
let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

/** Evaluates one brush at each point, on the GPU, and reads the distances back. */
async function evalAt(
  brush: BrushDesc,
  points: readonly [number, number, number][],
  set: BrushSet = defaultBrushSet,
): Promise<number[]> {
  const r = root!;
  const n = points.length;
  const evalBrush = set.evalBrush;
  const brushBuf = r.createUniform(Brush, set.make(brush));
  const pointsBuf = r.createReadonly(d.arrayOf(d.vec3f, n));
  pointsBuf.write(points.map(([x, y, z]) => d.vec3f(x, y, z)));
  const out = r.createMutable(d.arrayOf(d.f32, n));

  const pipeline = r.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= d.u32(n)) {
        return;
      }
      out.$[gid.x] = evalBrush(brushBuf.$, pointsBuf.$[gid.x]);
    }),
  });

  pipeline.dispatchWorkgroups(Math.ceil(n / 64));
  return [...(await out.buffer.read())];
}

const near = (actual: number, expected: number, msg: string, eps = 2e-3) =>
  assert.ok(
    Math.abs(actual - expected) < eps,
    `${msg}: expected ~${expected}, got ${actual}`,
  );

test('sphere is exact inside, outside and on the surface', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const got = await evalAt(
    { kind: 'sphere', size: 2, pos: [1, 0, 0] },
    [[1, 0, 0], [4, 0, 0], [3, 0, 0], [1, 0, 5]],
  );
  near(got[0]!, -2, 'centre is one radius inside');
  near(got[1]!, 1, 'three units out from a radius-2 sphere');
  near(got[2]!, 0, 'on the surface');
  near(got[3]!, 3, 'distance is radial, not axis-aligned');
});

test('a uniform scale scales the distance, not just the extent', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // A scaled SDF must stay a distance function: divide the query, multiply the result.
  // Dropping the multiply leaves a field that is 1-Lipschitz only at scale 1, and the
  // tracer overshoots through anything scaled up.
  const got = await evalAt(
    { kind: 'sphere', size: 1, scale: 3 },
    [[6, 0, 0], [0, 0, 0]],
  );
  near(got[0]!, 3, 'radius 3 after scaling, so 6 units out is 3 away');
  near(got[1]!, -3, 'centre is one scaled radius inside');
});

test('rotation applies about the brush origin, inverse-transforming the query', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  // A capsule runs along local Y. Rotated 90 degrees about Z it runs along world X, so a
  // point off the *world* X axis is now radial. A forward-rotated query (the classic sign
  // slip) would report the un-rotated distance and this test would read 1.5 instead of 0.5.
  const got = await evalAt(
    { kind: 'capsule', size: [0, 2, 0], radius: 0.5, euler: [0, 0, Math.PI / 2] },
    [[1, 0, 0], [0, 1, 0], [3, 0, 0]],
  );
  near(got[0]!, -0.5, 'inside the barrel, one unit along the new axis');
  near(got[1]!, 0.5, 'one unit off the new axis is half a radius out');
  near(got[2]!, 0.5, 'past the cap: 3 - 2 - 0.5');
});

test('plane and box agree with their closed forms', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const plane = await evalAt({ kind: 'plane', radius: 1.5 }, [[0, 3, 0], [9, 0, -9]]);
  near(plane[0]!, 1.5, 'a horizontal half-space measures along Y only');
  near(plane[1]!, -1.5, 'below the plane is inside, whatever X and Z are');

  const box = await evalAt({ kind: 'box', size: [1, 2, 3] }, [[2, 0, 0], [0, 0, 0], [1, 2, 3]]);
  near(box[0]!, 1, 'one unit off the near face');
  near(box[1]!, -1, 'centre is the nearest half-extent inside');
  near(box[2]!, 0, 'the corner is on the surface');
});

/**
 * A hex prism, the running example for a custom kind: `size.x` is the flat-to-flat radius,
 * `size.y` the half-height, and `radius` rounds the edges. Also the shape whose bound is
 * least obvious, which is the point - an author has to say what it reaches.
 */
const hexPrism = {
  sdf: (p: d.v3f, size: d.v3f, radius: number) => {
    'use gpu';
    const k = d.vec3f(-0.8660254, 0.5, 0.57735);
    const q = std.abs(p);
    const xy = q.xy - k.xy * (2 * std.min(std.dot(k.xy, q.xy), 0));
    const dxy = d.vec2f(
      std.length(xy - d.vec2f(std.clamp(xy.x, -k.z * size.x, k.z * size.x), size.x))
        * std.sign(xy.y - size.x),
      q.z - size.y,
    );
    return std.min(std.max(dxy.x, dxy.y), 0) + std.length(std.max(dxy, d.vec2f())) - radius;
  },
  bound: (size: readonly [number, number, number], radius: number) =>
    Math.hypot(size[0] / 0.8660254, size[1]) + radius,
};

test('a custom kind dispatches, and the builtins still do', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }
  const set = new BrushSet({ custom: { hexPrism } });
  assert.equal(set.kindId('sphere'), 0, 'registering a custom kind must not renumber builtins');
  assert.ok(set.kindId('hexPrism') >= 9, 'a custom kind gets an id past the builtins');

  // Flat-to-flat radius 1 along Y (the prism's axis is local Z), half-height 2.
  const hex = await evalAt(
    { kind: 'hexPrism', size: [1, 2, 0] },
    [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 4]],
    set,
  );
  near(hex[0]!, -1, 'the centre is one flat-to-flat radius inside');
  near(hex[1]!, 0, 'the middle of a flat face is on the surface');
  near(hex[2]!, 1, 'one unit out from that face');
  near(hex[3]!, 2, 'past the end cap: 4 - 2');

  // The same set must still evaluate a builtin, i.e. the chain falls through rather than
  // shadowing everything before it.
  const sphere = await evalAt({ kind: 'sphere', size: 2 }, [[5, 0, 0]], set);
  near(sphere[0]!, 3, 'a builtin still resolves in a set that has custom kinds');
});

test('a custom kind is unknown to the default set, and cannot shadow a builtin', () => {
  assert.throws(
    () => defaultBrushSet.make({ kind: 'hexPrism' }),
    /unknown primitive 'hexPrism'/,
    'an unregistered name has to fail loudly - it would otherwise bake as empty space',
  );
  assert.throws(
    () => new BrushSet({ custom: { sphere: hexPrism } }),
    /'sphere' is a builtin/,
    'silently overriding a builtin would change every existing shape in the game',
  );
});

test('a custom bound feeds the tile cull and the shape bounds alike', () => {
  const set = new BrushSet({ custom: { hexPrism } });
  // Corner-to-corner is the flat-to-flat radius over cos(30 degrees), which the closed form
  // above hides; getting it wrong under-reports the reach and the bake clips at tile seams.
  near(
    set.reachOf('hexPrism', [1, 2, 0], 0.1),
    Math.hypot(1 / 0.8660254, 2) + 0.1,
    'reachOf routes a custom kind to its own bound',
    1e-6,
  );
  near(set.reachOf('sphere', 2), 2, 'and still answers for a builtin', 1e-6);
  // `make` folds scale and the blend band in, because that is what the GPU compares against.
  const brush = set.make({ kind: 'hexPrism', size: [1, 2, 0], scale: 2, smooth: 0.05 });
  near(brush.bound, Math.hypot(1 / 0.8660254, 2) * 2 + 0.05, 'scaled and widened', 1e-6);
});
