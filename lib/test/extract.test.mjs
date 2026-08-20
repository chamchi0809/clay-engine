import { test } from 'node:test';
import assert from 'node:assert/strict';

/**
 * Mirror of `SurfaceExtractor`'s two grid passes. The GPU version differs only in how
 * it allocates ids; the dual-vertex placement and - the part that is easy to get wrong -
 * the four-cell tables and the winding flip are exactly what is checked here.
 */
function surfaceNets(dist, res, origin, cell) {
  const at = (x, y, z) => dist(origin[0] + x * cell, origin[1] + y * cell, origin[2] + z * cell);
  const flat = (i, j, k) => i + res * (j + res * k);
  const ids = new Int32Array(res * res * res).fill(-1);
  const verts = [];

  // --- pass 1: one dual vertex per cell that any of its 12 edges crosses -----
  for (let k = 0; k < res; k++) {
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const corner = (bx, by, bz) => [i + bx, j + by, k + bz];
        const dOf = (c) => at(c[0], c[1], c[2]);
        let sx = 0, sy = 0, sz = 0, w = 0;
        const edge = (a, b) => {
          const da = dOf(a), db = dOf(b);
          if (da * db >= 0) return;
          const t = Math.min(1, Math.max(0, da / (da - db)));
          sx += a[0] + (b[0] - a[0]) * t;
          sy += a[1] + (b[1] - a[1]) * t;
          sz += a[2] + (b[2] - a[2]) * t;
          w++;
        };
        for (const [by, bz] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
          edge(corner(0, by, bz), corner(1, by, bz));
          edge(corner(by, 0, bz), corner(by, 1, bz));
          edge(corner(by, bz, 0), corner(by, bz, 1));
        }
        if (w === 0) continue;
        ids[flat(i, j, k)] = verts.length;
        verts.push([
          origin[0] + (sx / w) * cell,
          origin[1] + (sy / w) * cell,
          origin[2] + (sz / w) * cell,
        ]);
      }
    }
  }

  // --- pass 2: one quad per lattice edge that straddles the surface ----------
  const tris = [];
  const emit = (a, b, c, dd, flip) => {
    if (a < 0 || b < 0 || c < 0 || dd < 0) return;
    const i1 = flip ? dd : b;
    const i3 = flip ? b : dd;
    tris.push([a, i1, c], [a, c, i3]);
  };
  const hi = res - 1;
  for (let k = 0; k < res; k++) {
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const dL = at(i, j, k);
        const flip = !(dL < 0);
        if (j >= 1 && k >= 1 && j <= hi && k <= hi && at(i + 1, j, k) * dL < 0) {
          emit(
            ids[flat(i, j - 1, k - 1)], ids[flat(i, j, k - 1)],
            ids[flat(i, j, k)], ids[flat(i, j - 1, k)], flip,
          );
        }
        if (i >= 1 && k >= 1 && i <= hi && k <= hi && at(i, j + 1, k) * dL < 0) {
          emit(
            ids[flat(i - 1, j, k - 1)], ids[flat(i - 1, j, k)],
            ids[flat(i, j, k)], ids[flat(i, j, k - 1)], flip,
          );
        }
        if (i >= 1 && j >= 1 && i <= hi && j <= hi && at(i, j, k + 1) * dL < 0) {
          emit(
            ids[flat(i - 1, j - 1, k)], ids[flat(i, j - 1, k)],
            ids[flat(i, j, k)], ids[flat(i - 1, j, k)], flip,
          );
        }
      }
    }
  }
  return { verts, tris };
}

const R = 3.1;
const sphere = (x, y, z) => Math.hypot(x, y, z) - R;
const res = 20;
const cell = 8 / res;
const origin = [-4, -4, -4];
const mesh = surfaceNets(sphere, res, origin, cell);

test('dual vertices land on the surface within a cell', () => {
  assert.ok(mesh.verts.length > 500, `too few vertices: ${mesh.verts.length}`);
  const worst = Math.max(...mesh.verts.map((v) => Math.abs(Math.hypot(...v) - R)));
  assert.ok(worst < cell, `worst deviation ${worst} >= cell ${cell}`);
});

test('the quad tables produce a closed, consistently oriented mesh', () => {
  // Every interior directed edge must appear exactly once, and its reverse exactly
  // once. A wrong cell table breaks the first half; a wrong flip breaks the second.
  const seen = new Map();
  for (const [a, b, c] of mesh.tris) {
    for (const e of [[a, b], [b, c], [c, a]]) {
      const k = `${e[0]},${e[1]}`;
      seen.set(k, (seen.get(k) ?? 0) + 1);
    }
  }
  const doubled = [...seen.entries()].filter(([, n]) => n !== 1);
  assert.deepEqual(doubled, [], 'a directed edge was used twice');
  const unpaired = [...seen.keys()].filter((k) => {
    const [a, b] = k.split(',');
    return !seen.has(`${b},${a}`);
  });
  assert.equal(unpaired.length, 0, `${unpaired.length} boundary edges on a closed sphere`);
});

test('winding faces outwards', () => {
  const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
  const cross = (p, q) => [
    p[1] * q[2] - p[2] * q[1],
    p[2] * q[0] - p[0] * q[2],
    p[0] * q[1] - p[1] * q[0],
  ];
  let outward = 0;
  for (const [a, b, c] of mesh.tris) {
    const va = mesh.verts[a], vb = mesh.verts[b], vc = mesh.verts[c];
    const n = cross(sub(vb, va), sub(vc, va));
    const mid = [(va[0] + vb[0] + vc[0]) / 3, (va[1] + vb[1] + vc[1]) / 3, (va[2] + vb[2] + vc[2]) / 3];
    if (n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] > 0) outward++;
  }
  assert.equal(outward, mesh.tris.length, `${mesh.tris.length - outward} inward-facing triangles`);
});

test('an inverted field flips every triangle', () => {
  const inv = surfaceNets((x, y, z) => -sphere(x, y, z), res, origin, cell);
  assert.equal(inv.tris.length, mesh.tris.length);
  const inward = inv.tris.filter(([a, b, c]) => {
    const va = inv.verts[a], vb = inv.verts[b], vc = inv.verts[c];
    const e1 = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
    const e2 = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const mid = [(va[0] + vb[0] + vc[0]) / 3, (va[1] + vb[1] + vc[1]) / 3, (va[2] + vb[2] + vc[2]) / 3];
    return n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] < 0;
  }).length;
  assert.equal(inward, inv.tris.length, 'inverting the field must invert the winding');
});
