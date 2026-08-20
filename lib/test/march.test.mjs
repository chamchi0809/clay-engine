// Plain-JS mirror of the non-trivial tracing math in src/trace/march.ts.
// The shader version can only be checked on a GPU; this checks the arithmetic.
//   node --test lib/test
import { test } from 'node:test';
import assert from 'node:assert/strict';

// --- mirror of makeTracer().sweep over an analytic field ---------------------

/** `dist(p) -> signed distance`, `band` = saturation distance (mirrors TracedField). */
function sweep(dist, ro, rd, { tMin = 0, tMax = 100, radius = 0, aperture = 0, band = 8, eps = 0.01, maxSteps = 256 } = {}) {
  const invGrowth = 1 / (1 + aperture);
  let t = tMin;
  let prev = 0;
  let steps = 0;
  for (let i = 0; i < maxSteps; i++) {
    steps = i + 1;
    const p = ro.map((c, k) => c + rd[k] * t);
    const D = dist(p);
    const sat = Math.max(-1, Math.min(1, D / band));
    const target = radius + aperture * t;
    if (D < target + eps && sat < 0.999) {
      t += Math.max(D - target, 0);
      return { t, hit: true, steps, dist: D };
    }
    prev = D;
    t = Math.max((t + D - radius) * invGrowth, t + eps * 0.25);
    if (t > tMax) break;
  }
  return { t: Math.min(t, tMax), hit: false, steps, dist: 0 };
}

const sphereAt = (cx, cy, cz, r) => (p) =>
  Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz) - r;

test('ray terminates on the surface it should hit', () => {
  const f = sphereAt(0, 0, 10, 2);
  const h = sweep(f, [0, 0, 0], [0, 0, 1]);
  assert.ok(h.hit, 'must hit');
  // Sphere front face is at t = 8. The hit epsilon is 0.01, the tail extrapolation
  // is capped at 5x, so the worst case is a few hundredths short or long.
  assert.ok(Math.abs(h.t - 8) < 0.06, `t=${h.t}`);
});

test('a zero-radius ray never reports a hit without the epsilon', () => {
  // eps = 0 is the bug this guards: |D| shrinks geometrically and never reaches 0.
  const f = sphereAt(0, 0, 10, 2);
  assert.equal(sweep(f, [0, 0, 0], [0, 0, 1], { eps: 0 }).hit, false);
});

test('ray that misses reports no hit', () => {
  const f = sphereAt(0, 0, 10, 2);
  assert.equal(sweep(f, [0, 0, 0], [0, 1, 0]).hit, false);
});

test('sweep never tunnels through the surface', () => {
  // Every reported hit must be at or before the true entry point, for all three
  // sweep kinds. Overshooting is what puts the shading point *inside* geometry and
  // turns AO and shadows black.
  const f = sphereAt(0, 0, 10, 2);
  for (const aperture of [0, 0.01, 0.05, 0.2]) {
    for (const radius of [0, 0.25, 1]) {
      const h = sweep(f, [0, 0, 0], [0, 0, 1], { radius, aperture });
      assert.ok(h.hit, `radius=${radius} aperture=${aperture} must hit`);
      // Tangency: t + radius + aperture*t = 8  ->  t = (8 - radius)/(1 + aperture)
      const tangency = (8 - radius) / (1 + aperture);
      assert.ok(
        h.t <= tangency + 1e-6,
        `radius=${radius} aperture=${aperture}: t=${h.t} past tangency ${tangency}`,
      );
    }
  }
});

test('a saturated sample is never a hit (conservative outer bounds stay invisible)', () => {
  // `volumeField` returns the distance to the volume box, flagged saturated, for
  // points outside it. Without the saturation guard the tracer stops on the box.
  const boxSkin = () => 0.0001; // pretends to be right at a bound, always saturated
  const h = sweep(boxSkin, [0, 0, 0], [0, 0, 1], { band: 0.0001 });
  assert.equal(h.hit, false);
});

test('cone step is the tangency point, not slide 27 printed factor', () => {
  // Advance from t with free distance D under aperture a: the swept cone touches at
  // t' = (t + D)/(1 + a). Slide 27's C/(C - a) is > 1 and overshoots.
  const [t, D, a] = [3, 1.5, 0.2];
  const tangency = (t + D) / (1 + a);
  const radiusThere = a * tangency;
  assert.ok(Math.abs((tangency - t) + radiusThere - D) < 1e-12);
  const C = Math.sqrt(a * a + 1);
  assert.ok(C / (C - a) > 1, 'slide factor is expansive');
  assert.ok(C / (C + a) < 1, 'tangency factor is contractive');
  assert.ok(Math.abs(1 / (1 + a) - 1 / (1 + Math.sin(Math.atan(a)) / Math.cos(Math.atan(a)))) < 1e-12);
});

test('the final snap is one Lipschitz step, never the geometric tail', () => {
  // Slide 25's tail, `(D - target)/(1 - D/prev)`, is unbounded as prev -> D, which is
  // what happens when the nearest feature changes between two samples. The snap used
  // instead can never exceed the free distance, so it can never land inside.
  const tail = (D, prev, target = 0) => (D - target) / Math.max(1 - D / prev, 0.2);
  const snap = (D, target = 0) => Math.max(D - target, 0);
  assert.equal(snap(0.1), 0.1);
  assert.ok(tail(0.1, 0.1 + 1e-9) > snap(0.1), 'the tail overshoots where the snap does not');
  // Already past the tangency target: no forward motion at all.
  assert.equal(snap(0.4, 0.5), 0);
});

test('the snap never puts the hit inside the surface, for any sweep kind', () => {
  // Directly checks the Lipschitz argument on a field whose nearest feature switches
  // mid-ray: a sphere in front, a plane behind. `prev/dist` ratios go to 1 here.
  const f = (p) => Math.min(Math.hypot(p[0], p[1], p[2] - 10) - 2, 12 - p[2]);
  for (const aperture of [0, 0.02, 0.1]) {
    for (const radius of [0, 0.5]) {
      const h = sweep(f, [0, 0, 0], [0, 0, 1], { radius, aperture });
      const p = [0, 0, h.t];
      // Clearance at the reported hit must still cover the swept shape.
      assert.ok(
        f(p) >= radius + aperture * h.t - 1e-9,
        `radius=${radius} aperture=${aperture}: clearance ${f(p)} < ${radius + aperture * h.t}`,
      );
    }
  }
});

// --- mirror of the pre-pass tile cone in src/render/raymarch.ts --------------

test('tile cone contains every ray in its 8x8 block', () => {
  const [W, H, TILE, SLACK] = [1280, 720, 8, 1.05];
  const tanHalfFov = Math.tan(Math.PI / 3 / 2);
  const aspect = W / H;
  const dirAt = (px, py) => {
    const [nx, ny] = [(px / W) * 2 - 1, 1 - (py / H) * 2];
    const v = [nx * tanHalfFov * aspect, ny * tanHalfFov, 1];
    const l = Math.hypot(...v);
    return v.map((c) => c / l);
  };
  const tanAngle = (a, b) => {
    const cd = a.reduce((s, v, k) => s + v * b[k], 0);
    return Math.sqrt(Math.max(1 / (cd * cd) - 1, 0));
  };
  let worstParaxial = 0;
  let worstCorner = 0;
  for (let ty = 0; ty < H / TILE; ty += 7) {
    for (let tx = 0; tx < W / TILE; tx += 11) {
      const cx = (tx + 0.5) * TILE;
      const cy = (ty + 0.5) * TILE;
      const axis = dirAt(cx, cy);
      // What the shader computes: max over the four corners, times the slack factor.
      // Which corner wins depends on the aspect ratio and on where the block sits, so
      // no single diagonal is safe to pick.
      let corners = 0;
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          corners = Math.max(corners, tanAngle(axis, dirAt(cx + sx * TILE / 2, cy + sy * TILE / 2)));
        }
      }
      const aperture = corners * SLACK;
      // What it must contain: every ray in the block, not just its corners.
      let worst = 0;
      for (let py = -TILE / 2; py <= TILE / 2; py += 0.5) {
        for (let px = -TILE / 2; px <= TILE / 2; px += 0.5) {
          worst = Math.max(worst, tanAngle(axis, dirAt(cx + px, cy + py)));
        }
      }
      worstCorner = Math.max(worstCorner, worst / aperture);
      // The paraxial estimate this replaced: one pixel's aperture times the block
      // circumradius. It undershoots badly off-centre.
      worstParaxial = Math.max(worstParaxial, worst / (tanHalfFov * (1 / H) * Math.SQRT1_2 * TILE));
    }
  }
  assert.ok(worstCorner <= 1, `tile cone too tight by ${worstCorner}x`);
  assert.ok(worstParaxial > 1.3, `paraxial estimate should undershoot, got ${worstParaxial}`);
});

// --- mirror of packTile/unpackTile in src/field/tilegrid.ts ------------------

test('tile coordinate packing round-trips up to 1023 per axis', () => {
  const pack = (x, y, z) => (x | (y << 10) | (z << 20)) >>> 0;
  const unpack = (v) => [v & 1023, (v >>> 10) & 1023, (v >>> 20) & 1023];
  for (const c of [[0, 0, 0], [1, 2, 3], [1023, 1023, 1023], [7, 0, 511]]) {
    assert.deepEqual(unpack(pack(...c)), c);
  }
});

// --- mirror of coneAO / cosineHemisphere in src/trace/shade.ts ---------------

/** Mirror of math/gpu.ts `cosineHemisphere`. */
function cosineHemisphere(n, u, v) {
  const a = u * 2 * Math.PI;
  const r = Math.sqrt(v);
  const z = Math.sqrt(Math.max(0, 1 - v));
  const t = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  const norm = (p) => { const l = Math.hypot(...p); return p.map((c) => c / l); };
  const tx = norm(cross(t, n));
  const ty = cross(n, tx);
  return norm(tx.map((c, k) => c * r * Math.cos(a) + ty[k] * r * Math.sin(a) + n[k] * z));
}

test('cosineHemisphere stays in the hemisphere and is cosine-weighted', () => {
  const n = [0, 1, 0];
  let sum = 0;
  const N = 4000;
  for (let i = 0; i < N; i++) {
    const dir = cosineHemisphere(n, (i * 0.618033988749895) % 1, (i + 0.5) / N);
    const ndl = dir[1];
    assert.ok(ndl >= -1e-9, `below the horizon: ${ndl}`);
    assert.ok(Math.abs(Math.hypot(...dir) - 1) < 1e-9, 'not unit length');
    sum += ndl;
  }
  // E[cos] over a cosine-weighted hemisphere is 2/3.
  assert.ok(Math.abs(sum / N - 2 / 3) < 0.02, `E[cos]=${sum / N}`);
});

test('AO darkens the crease where a sphere meets the ground', () => {
  // The defect this guards: offsetting the AO cone by `bias` *and* starting it at
  // t = bias blinds it to everything within ~2 voxels of the surface, so the crease
  // around a resting object stays fully bright - a light ring exactly where the
  // occlusion should be strongest.
  const voxel = 24 / 128;
  const aoDistance = 2.5;
  // Ground plane at y=0 plus a unit sphere resting on it.
  const f = (p) => Math.min(p[1], Math.hypot(p[0], p[1] - 1, p[2]) - 1);
  const ao = ({ bias, tMin, aperture }) => (p) => {
    let sum = 0;
    const N = 512;
    for (let i = 0; i < N; i++) {
      const dir = cosineHemisphere([0, 1, 0], (i * 0.618033988749895) % 1, (i + 0.5) / N);
      const o = [p[0], p[1] + bias, p[2]];
      const h = sweep(f, o, dir, { tMin, tMax: aoDistance, aperture, band: voxel * 4, eps: voxel * 0.5 });
      sum += h.hit ? Math.min(h.t / aoDistance, 1) : 1;
    }
    return sum / N;
  };
  const inCrease = [0.25, 0, 0]; // 0.25 from the contact point
  const openGround = [8, 0, 0];
  const fixed = ao({ bias: voxel, tMin: 0, aperture: 0.08 });
  const old = ao({ bias: voxel * 8, tMin: voxel * 8, aperture: 0.35 });

  assert.ok(fixed(openGround) > 0.9, `open ground must stay bright, got ${fixed(openGround)}`);
  assert.ok(fixed(inCrease) < 0.7, `crease must darken, got ${fixed(inCrease)}`);
  // And the old parameters are what failed to see it.
  assert.ok(
    old(inCrease) - fixed(inCrease) > 0.15,
    `old bias should miss the crease: old=${old(inCrease)} fixed=${fixed(inCrease)}`,
  );
});

// --- mirror of INTERP_SLACK in src/trace/march.ts ----------------------------

test('the coarse-mip discount makes an interpolated read conservative', () => {
  // Why the discount exists: a level whose voxel is wider than the feature reads it as
  // farther away than it is, and a sphere trace that believes that value steps clean
  // through it. Here: a 0.7-thick shell in a level with 3-unit voxels (mip 4 of a
  // 24-unit / 128^3 volume) - which is the demo's torus tube.
  const SLACK = 0.866; // sqrt(3)/2, half a cell diagonal
  const VOXEL = 3;
  const truth = (z) => Math.abs(z - 10) - 0.35;
  const coarse = (z) => {
    const g = Math.floor(z / VOXEL) * VOXEL;
    const f = (z - g) / VOXEL;
    return truth(g) * (1 - f) + truth(g + VOXEL) * f;
  };
  let worstRaw = 0;
  let worstFixed = 0;
  for (let z = 0; z < 20; z += 0.005) {
    worstRaw = Math.max(worstRaw, coarse(z) - truth(z));
    worstFixed = Math.max(worstFixed, coarse(z) - SLACK * VOXEL - truth(z));
  }
  assert.ok(worstRaw > 1, `raw read should over-estimate, got ${worstRaw}`);
  assert.ok(worstFixed <= 0, `discounted read still over-estimates by ${worstFixed}`);
});

test('sqrt(3)/2 voxels is the worst case for a trilinear read', () => {
  // The bound is reached by a point feature sitting at a cell centre: every corner is
  // half a diagonal away, so the interpolant returns that instead of 0.
  const h = 0.5; // half-voxel, so voxel = 1
  let corners = 0;
  for (const x of [-h, h]) {
    for (const y of [-h, h]) {
      for (const z of [-h, h]) corners += Math.hypot(x, y, z);
    }
  }
  const atCentre = corners / 8;
  assert.ok(Math.abs(atCentre - Math.sqrt(3) / 2) < 1e-12, `${atCentre}`);
  assert.ok(Math.abs(0.866 - Math.sqrt(3) / 2) < 1e-3, 'INTERP_SLACK must match sqrt(3)/2');
});
