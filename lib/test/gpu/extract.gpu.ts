import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { ExtractMotion, SurfaceExtractor } from '../../src/sim/extract.ts';
import { ParticleSet } from '../../src/sim/particles.ts';
import { analyticField, offsetField } from '../../src/trace/field.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

test('re-extraction keeps the live GPU centre and mean velocity', async (t) => {
  if (!available) {
    return t.skip('no WebGPU adapter');
  }

  const set = new ParticleSet(root!, { capacity: 1024, maxBodies: 1, label: 'extractTest' });
  const body = set.addBody();
  set.particles.writePartial([
    {
      idx: 0,
      value: { pos: [3, 5, 6], body, prev: [3, 5, 6], vel: [1, 2, 3], velPrev: [1, 2, 3] },
    },
    {
      idx: 1,
      value: { pos: [5, 5, 6], body, prev: [5, 5, 6], vel: [3, 4, 5], velPrev: [3, 4, 5] },
    },
  ]);
  set.bodies.writePartial([{
    idx: body,
    value: { first: 0, count: 2, com: [4, 5, 6], restCom: [4, 5, 6] },
  }]);

  const motion = root!.createMutable(ExtractMotion, {
    center: [99, 99, 99],
    _padA: 0,
    velocity: [0, 0, 0],
    _padB: 0,
  });
  const localSphere = analyticField((p: d.v3f) => {
    'use gpu';
    return std.length(p) - 0.75;
  });
  const field = offsetField(localSphere, () => {
    'use gpu';
    return d.vec3f(motion.$.center);
  });
  const extractor = new SurfaceExtractor(set, field, {
    resolution: 16,
    body,
    base: 0,
    capacity: 1000,
    motion,
    label: 'motionContinuity',
  });
  extractor.setRegion([-1.5, -1.5, -1.5], 3, [0, 0, 0], true);

  const encoder = root!['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  extractor.extract(pass);
  pass.end();
  encoder.submit();

  const gotMotion = await motion.read();
  assert.deepEqual(
    [gotMotion.center.x, gotMotion.center.y, gotMotion.center.z],
    [4, 5, 6],
    'the new field must be centred on the live cloud, not a CPU/readback position',
  );
  assert.deepEqual(
    [gotMotion.velocity.x, gotMotion.velocity.y, gotMotion.velocity.z],
    [2, 3, 4],
    'new particles must inherit the cloud mean velocity',
  );

  const bodies = await set.bodies.read();
  const particles = await set.particles.read();
  const count = bodies[body]!.count;
  assert.ok(count > 100, `expected an extracted sphere, got ${count} particles`);
  const centerSum = [0, 0, 0];
  for (let i = 0; i < count; i++) {
    const p = particles[i]!;
    centerSum[0] += p.pos.x;
    centerSum[1] += p.pos.y;
    centerSum[2] += p.pos.z;
    assert.deepEqual([p.vel.x, p.vel.y, p.vel.z], [2, 3, 4]);
  }
  for (let i = 0; i < 3; i++) {
    const center = centerSum[i]! / count;
    assert.ok(Math.abs(center - [4, 5, 6][i]!) < 0.03, `centre axis ${i} drifted to ${center}`);
  }
});
