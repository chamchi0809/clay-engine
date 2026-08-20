import { test } from 'node:test';
import assert from 'node:assert/strict';

/**
 * JS mirrors of `shape/sdf.ts` and `game/orbit.ts`. Both are pure arithmetic over plain
 * data, so the parts that are silently wrong rather than loudly wrong - a flattened
 * subtraction, a bounds radius that under-covers - are testable without a GPU.
 */

// --- sdf.cut flattening ---------------------------------------------------
// `a \ (b | c)` must become the fold [a add, b cut, c cut]: apply(add a) -> a,
// apply(cut b) -> a\b, apply(cut c) -> (a\b)\c == a\(b|c). The identity only holds
// while the removed side is a pure union.
test('cut flattens a union of holes into consecutive cut ops', () => {
  const fold = (ops) => ops.reduce((acc, [op, set]) =>
    op === 'add' ? new Set([...acc, ...set]) : new Set([...acc].filter((v) => !set.has(v))),
    new Set());
  const a = new Set([1, 2, 3, 4]);
  const b = new Set([2]);
  const c = new Set([3]);
  const flattened = fold([['add', a], ['cut', b], ['cut', c]]);
  const tree = new Set([...a].filter((v) => !b.has(v) && !c.has(v)));
  assert.deepEqual([...flattened], [...tree]);
});

test('cut of a cut is not expressible as a fold, so it must throw rather than flatten', () => {
  const fold = (ops) => ops.reduce((acc, [op, set]) =>
    op === 'add' ? new Set([...acc, ...set]) : new Set([...acc].filter((v) => !set.has(v))),
    new Set());
  // a \ (b \ c) == (a\b) | (a&c), whereas a naive op flip yields (a\b) | c. They part
  // company exactly when the inner cutter pokes outside the solid - `9` here - and then
  // the flattened version *adds* material where the tree removed none.
  const a = new Set([1, 2]);
  const b = new Set([2]);
  const c = new Set([2, 9]);
  const naive = fold([['add', a], ['cut', b], ['add', c]]);
  const tree = new Set([...a].filter((v) => !(b.has(v) && !c.has(v))));
  assert.deepEqual([...naive].sort(), [1, 2, 9]);
  assert.deepEqual([...tree].sort(), [1, 2]);
  assert.notDeepEqual([...naive].sort(), [...tree].sort());
});

// --- shapeBounds ----------------------------------------------------------
// The clay body derives its extraction box, collider radius and bake voxel from this,
// so under-covering clips the body flat where it touches the box boundary.
test('bounds cover every additive primitive and ignore subtractions', () => {
  const reach = { sphere: (s) => s.size };
  const nodes = [
    { kind: 'sphere', pos: [0, 0, 0], size: 1, op: 'add' },
    { kind: 'sphere', pos: [3, 0, 0], size: 0.5, op: 'add' },
    { kind: 'sphere', pos: [40, 0, 0], size: 9, op: 'cut' },
  ];
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (const n of nodes) {
    if (n.op === 'cut') continue;
    const r = reach[n.kind](n);
    lo = lo.map((v, i) => Math.min(v, n.pos[i] - r));
    hi = hi.map((v, i) => Math.max(v, n.pos[i] + r));
  }
  assert.deepEqual(lo, [-1, -1, -1]);
  assert.deepEqual(hi, [3.5, 1, 1]);
  const center = lo.map((v, i) => (v + hi[i]) / 2);
  const radius = Math.max(...hi.map((v, i) => v - center[i]));
  // Every additive primitive is inside the sphere (center, radius).
  for (const n of nodes.filter((x) => x.op !== 'cut')) {
    const d = Math.hypot(...n.pos.map((v, i) => v - center[i]));
    assert.ok(d + reach[n.kind](n) <= radius * Math.sqrt(3) + 1e-9);
  }
});

// --- orbit ----------------------------------------------------------------
const orbit = (around, o) => {
  const horiz = Math.cos(o.pitch) * o.distance;
  const vert = Math.sin(o.pitch) * o.distance;
  return {
    position: [
      around[0] - Math.sin(o.yaw) * horiz,
      around[1] + vert + (o.height ?? 0),
      around[2] - Math.cos(o.yaw) * horiz,
    ],
    target: [around[0], around[1] + (o.lookHeight ?? 0), around[2]],
  };
};

test('orbit keeps the requested distance and puts the eye behind the yaw direction', () => {
  const around = [1, 2, 3];
  for (const yaw of [0, 0.7, Math.PI, -2.1]) {
    const { position } = orbit(around, { yaw, pitch: 0, distance: 8.5 });
    assert.equal(position[1], around[1], 'pitch 0 stays level');
    assert.ok(Math.abs(Math.hypot(position[0] - around[0], position[2] - around[2]) - 8.5) < 1e-9);
    // yaw is the direction the camera looks along, so the eye sits opposite it.
    const fwd = [Math.sin(yaw), Math.cos(yaw)];
    const toTarget = [around[0] - position[0], around[2] - position[2]];
    assert.ok(fwd[0] * toTarget[0] + fwd[1] * toTarget[1] > 0);
  }
});

test('orbit height raises the eye without moving the look-at point', () => {
  const { position, target } = orbit([0, 0, 0], { yaw: 0.3, pitch: 0.4, distance: 6, height: 1.2, lookHeight: 0.4 });
  assert.ok(Math.abs(position[1] - (Math.sin(0.4) * 6 + 1.2)) < 1e-9);
  assert.deepEqual(target, [0, 0.4, 0]);
});

// --- material mask weight -------------------------------------------------
// Mirror of the `keep` weight in `applyBrush`. Silently-wrong failure modes here are an
// inverted select (mask -1 disabling the brush instead of enabling it) and a flipped
// smoothstep (carving everything *except* the target material), both of which look like
// a physics bug rather than a shader bug.
const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const keep = (material, mask) =>
  mask < 0 ? 1 : 1 - smoothstep(0.25, 0.75, Math.abs(material - mask));

test('an unmasked brush applies everywhere, a masked one only on its own material', () => {
  assert.equal(keep(0, -1), 1);
  assert.equal(keep(7, -1), 1);
  // Exactly on the target: full effect. One slot away: none.
  assert.equal(keep(2, 2), 1);
  assert.equal(keep(3, 2), 0);
  assert.equal(keep(1, 2), 0);
  // Halfway across a seam the two materials blend, so the carve does too.
  assert.equal(keep(2.5, 2), 0.5);
  // Monotone in the distance from the target, or the ramp has a hole in it.
  const w = [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1].map((dx) => keep(2 + dx, 2));
  for (let i = 1; i < w.length; i++) {
    assert.ok(w[i] <= w[i - 1], `keep must not rise with distance: ${w.join(', ')}`);
  }
});

// --- plasticity units -----------------------------------------------------
// Mirror of the per-second -> per-substep conversion in `SoftBody`. The failure mode this
// guards is silent and expensive to find: feed a per-second number straight into a
// per-substep decay and the rest shape is forgotten inside two frames, so a body resting
// under gravity collapses into a bowl and looks like a broken collider instead of a
// mistuned knob.
const perSubstep = (p, h) => 1 - (1 - Math.min(1, Math.max(0, p))) ** h;

test('plasticity converts per-second to per-substep and stays rate-independent', () => {
  const h = 1 / 180;
  assert.equal(perSubstep(0, h), 0);
  assert.equal(perSubstep(1, h), 1);
  // A third per second is a very small bite per substep - two orders below the number.
  assert.ok(perSubstep(0.35, h) < 0.005, String(perSubstep(0.35, h)));
  // Same clay at any substep rate: compounding one second of substeps must land on the
  // per-second figure however the second is diced.
  for (const substeps of [60, 180, 600]) {
    const kept = (1 - perSubstep(0.35, 1 / substeps)) ** substeps;
    assert.ok(Math.abs(kept - 0.65) < 1e-9, `${substeps}: ${kept}`);
  }
});

// --- quaternion composition -----------------------------------------------
// Mirror of `quatMul` + `eulerToQuat` in `field/brush.ts`. A brush transforms points by
// `quatRotate(conj(rot), p - pos)`, so a wrong multiplication order or a sign slip turns
// a stamped imprint into a mirrored one - the shape still looks plausible, it just faces
// the wrong way, which is exactly the kind of bug nobody spots in a screenshot.
const qmul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const qrot = (q, v) => {
  const [x, y, z, w] = q;
  const t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])];
  return [
    v[0] + w * t[0] + (y * t[2] - z * t[1]),
    v[1] + w * t[1] + (z * t[0] - x * t[2]),
    v[2] + w * t[2] + (x * t[1] - y * t[0]),
  ];
};
const axis = (a, deg) => {
  const h = (deg * Math.PI) / 360;
  return [a[0] * Math.sin(h), a[1] * Math.sin(h), a[2] * Math.sin(h), Math.cos(h)];
};

test('turn() composes on top of an authored rotate(), in that order', () => {
  const near = (got, want) =>
    got.every((v, i) => Math.abs(v - want[i]) < 1e-9) || assert.fail(`${got} != ${want}`);
  // Identity is a no-op on either side.
  near(qmul([0, 0, 0, 1], axis([0, 0, 1], 90)), axis([0, 0, 1], 90));
  near(qmul(axis([0, 0, 1], 90), [0, 0, 0, 1]), axis([0, 0, 1], 90));
  // A rod authored lying along X (rotate z 90: y -> -x) then turned 90 about Y
  // (x -> -z) ends along +Z, not back along Y: the runtime rotation is applied *after*
  // the authored one.
  const rod = axis([0, 0, 1], 90);
  const q = qmul(axis([0, 1, 0], 90), rod);
  near(qrot(q, [0, 1, 0]).map((v) => +v.toFixed(12)), [0, 0, 1]);
  // Order matters, so the mirrored composition must differ.
  const flipped = qmul(rod, axis([0, 1, 0], 90));
  assert.ok(q.some((v, i) => Math.abs(v - flipped[i]) > 1e-6));
});
