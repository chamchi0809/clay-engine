import test from 'node:test';
import assert from 'node:assert/strict';

// JS mirror of the three things in `sim/fluid.ts` that are silently wrong if they are
// wrong: the hash cell size (a 3x3x3 scan has to be a *complete* neighbour search),
// the splat bounds (a footprint that is too small punches holes in the surface), and
// the fixed-point encoding `atomicMin` operates on (it has to be monotone in `v`, or
// the union of spheres comes out as a max).

const ENC_SCALE = 32767.5;
const ENC_SAT = 65535;
const enc = (v) => Math.floor((Math.min(Math.max(v, -1), 1) + 1) * ENC_SCALE);
const dec = (u) => u / ENC_SCALE - 1;

let seed = 12345;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const cloud = (n, lo, hi) =>
  Array.from({ length: n }, () => [
    lo + rnd() * (hi - lo),
    lo + rnd() * (hi - lo),
    lo + rnd() * (hi - lo),
  ]);

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('a 3x3x3 scan of an h-sized hash is a complete neighbour search', () => {
  const h = 1.0;
  const origin = [-12 - h, -4 - h, -12 - h];
  const gridRes = Math.ceil(24 / h) + 2;
  // Inside the volume (x,z in [-12,12], y in [-4,20]) and dense enough that most cells
  // are occupied - a sparse cloud would pass this test vacuously.
  const pts = cloud(400, -3, 3);

  const cellOf = (p) => p.map((c, i) => Math.floor((c - origin[i]) / h));
  const key = (c) => `${c[0]},${c[1]},${c[2]}`;
  const buckets = new Map();
  pts.forEach((p, i) => {
    const c = cellOf(p);
    assert.ok(!c.some((v) => v < 0 || v >= gridRes), 'grid must cover the volume');
    const k = key(c);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(i);
  });

  for (let i = 0; i < pts.length; i++) {
    const base = cellOf(pts[i]);
    const found = new Set();
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          for (const j of buckets.get(key([base[0] + dx, base[1] + dy, base[2] + dz])) ?? []) {
            if (dist(pts[i], pts[j]) < h) found.add(j);
          }
        }
      }
    }
    const brute = new Set(pts.flatMap((q, j) => (dist(pts[i], q) < h ? [j] : [])));
    const asc = (a, b) => a - b;
    assert.deepEqual([...found].sort(asc), [...brute].sort(asc), `particle ${i}`);
  }
});

test('splat bounds cover every voxel the particle can lower', () => {
  const res = 32;
  const voxel = 24 / res;
  const surfR = 0.5;
  for (const band of [2, 4]) {
    const bandWorld = band * voxel;
    const reach = surfR + bandWorld;
    for (const p of cloud(60, 1, 22)) {
      const lo = p.map((c) => Math.floor((c - reach) / voxel));
      const hi = p.map((c) => Math.floor((c + reach) / voxel));
      for (let z = 0; z < res; z++) {
        for (let y = 0; y < res; y++) {
          for (let x = 0; x < res; x++) {
            const c = [(x + 0.5) * voxel, (y + 0.5) * voxel, (z + 0.5) * voxel];
            if (dist(c, p) - surfR >= bandWorld) continue;
            assert.ok(
              x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && z >= lo[2] && z <= hi[2],
              `voxel ${x},${y},${z} affected but outside bounds for band ${band}`,
            );
          }
        }
      }
    }
  }
});

test('the fixed-point encoding turns atomicMin into a distance union', () => {
  assert.equal(enc(-1), 0);
  assert.equal(enc(1), ENC_SAT);
  // Monotone, so min-of-encoded == encoded-of-min.
  let prev = -1;
  for (let i = 0; i <= 400; i++) {
    const e = enc(-1.5 + i * 0.0075);
    assert.ok(e >= prev, 'encoding must be non-decreasing');
    prev = e;
  }
  for (let i = 0; i < 500; i++) {
    const a = rnd() * 4 - 2;
    const b = rnd() * 4 - 2;
    assert.equal(Math.min(enc(a), enc(b)), enc(Math.min(a, b)));
  }
  // Round-trip error stays under half a quantisation step of the band.
  for (let i = 0; i < 500; i++) {
    const v = rnd() * 2 - 1;
    assert.ok(Math.abs(dec(enc(v)) - v) <= 1 / ENC_SCALE);
  }
});

test('rest density is the density of the lattice it is derived from', () => {
  const spacing = 0.45;
  const h = spacing * 2.2;
  const poly6 = 315 / (64 * Math.PI * h ** 9);
  const mass = spacing ** 3;
  const reach = Math.ceil(h / spacing);
  let rest = 0;
  for (let z = -reach; z <= reach; z++) {
    for (let y = -reach; y <= reach; y++) {
      for (let x = -reach; x <= reach; x++) {
        const r2 = (x * x + y * y + z * z) * spacing * spacing;
        if (r2 < h * h) rest += mass * poly6 * (h * h - r2) ** 3;
      }
    }
  }
  // Measure it the way the shader does: sum poly6 over the real lattice around a point.
  const pts = [];
  for (let z = -6; z <= 6; z++) {
    for (let y = -6; y <= 6; y++) {
      for (let x = -6; x <= 6; x++) pts.push([x * spacing, y * spacing, z * spacing]);
    }
  }
  let measured = 0;
  for (const q of pts) {
    const r2 = q[0] ** 2 + q[1] ** 2 + q[2] ** 2;
    if (r2 < h * h) measured += mass * poly6 * (h * h - r2) ** 3;
  }
  assert.ok(Math.abs(measured - rest) < 1e-9, `${measured} vs ${rest}`);
  // Sanity: a fluid at rest spacing is near-incompressible, so rho0 ~ mass/spacing^3 = 1.
  assert.ok(rest > 0.5 && rest < 2, `rest density ${rest} is not physical`);
});
