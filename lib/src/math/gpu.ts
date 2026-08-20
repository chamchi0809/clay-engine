import tgpu, { d, std } from 'typegpu';

/** Quaternion helpers (xyzw). Shared by brush transforms and the PBD solver. */

export const quatMul = tgpu.fn([d.vec4f, d.vec4f], d.vec4f)((a, b) => {
  'use gpu';
  return d.vec4f(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  );
});

export const quatConj = tgpu.fn([d.vec4f], d.vec4f)((q) => {
  'use gpu';
  return d.vec4f(-q.x, -q.y, -q.z, q.w);
});

/** Rotate `v` by unit quaternion `q`. */
export const quatRotate = tgpu.fn([d.vec4f, d.vec3f], d.vec3f)((q, v) => {
  'use gpu';
  const t = std.cross(q.xyz, v) * 2;
  return v + t * q.w + std.cross(q.xyz, t);
});

export const quatFromAxisAngle = tgpu.fn([d.vec3f, d.f32], d.vec4f)((axis, angle) => {
  'use gpu';
  const h = angle * 0.5;
  return d.vec4f(axis * std.sin(h), std.cos(h));
});

export const quatToMat3 = tgpu.fn([d.vec4f], d.mat3x3f)((q) => {
  'use gpu';
  const x = q.x;
  const y = q.y;
  const z = q.z;
  const w = q.w;
  return d.mat3x3f(
    d.vec3f(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
    d.vec3f(2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
    d.vec3f(2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
  );
});

/**
 * Extract the rotational part of a 3x3 deformation matrix as a quaternion.
 *
 * Müller et al. 2016, "A Robust Method to Extract the Rotational Part of
 * Deformations" — a few Gauss-Newton style quaternion iterations. Claybook used a
 * ported CUDA SVD/PD solver for the same job (GDC'18 slide 50); this is the same
 * polar decomposition with a fraction of the register pressure, and warm-starting
 * from the previous frame's rotation converges in 1-2 iterations.
 */
export const extractRotation = tgpu.fn([d.mat3x3f, d.vec4f, d.u32], d.vec4f)(
  (A, qStart, iterations) => {
    'use gpu';
    let q = d.vec4f(qStart);
    for (let i = d.u32(0); i < iterations; i++) {
      const R = quatToMat3(q);
      const c0 = R.columns[0];
      const c1 = R.columns[1];
      const c2 = R.columns[2];
      const a0 = A.columns[0];
      const a1 = A.columns[1];
      const a2 = A.columns[2];
      const denom = std.abs(std.dot(c0, a0) + std.dot(c1, a1) + std.dot(c2, a2)) + 1e-9;
      const omega = (std.cross(c0, a0) + std.cross(c1, a1) + std.cross(c2, a2)) * (1 / denom);
      const w = std.length(omega);
      if (w < 1e-9) {
        break;
      }
      q = std.normalize(quatMul(quatFromAxisAngle(omega * (1 / w), w), q));
    }
    return d.vec4f(q);
  },
);

/** Polynomial smooth min (iq). Returns the blended distance. */
export const sminPoly = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((a, b, k) => {
  'use gpu';
  const h = std.clamp(0.5 + 0.5 * (b - a) / std.max(k, 1e-6), 0, 1);
  return std.mix(b, a, h) - std.max(k, 1e-6) * h * (1 - h);
});

/** Smooth min that also reports the blend weight, so material ids can follow the surface. */
export const sminPolyWeighted = tgpu.fn([d.f32, d.f32, d.f32], d.vec2f)((a, b, k) => {
  'use gpu';
  const kk = std.max(k, 1e-6);
  const h = std.clamp(0.5 + 0.5 * (b - a) / kk, 0, 1);
  return d.vec2f(std.mix(b, a, h) - kk * h * (1 - h), h);
});

/** Smooth max, used for cuts: max(a, -b) with a soft seam. */
export const smaxPoly = tgpu.fn([d.f32, d.f32, d.f32], d.f32)((a, b, k) => {
  'use gpu';
  const kk = std.max(k, 1e-6);
  const h = std.clamp(0.5 - 0.5 * (b - a) / kk, 0, 1);
  return std.mix(b, a, h) + kk * h * (1 - h);
});

/** Cheap hash -> [0,1). */
export const hash1 = tgpu.fn([d.u32], d.f32)((seed) => {
  'use gpu';
  let x = seed;
  x = x ^ (x >>> 16);
  x = x * d.u32(0x7feb352d);
  x = x ^ (x >>> 15);
  x = x * d.u32(0x846ca68b);
  x = x ^ (x >>> 16);
  return d.f32(x & d.u32(0xffffff)) * (1 / 16777216);
});

export const hash3 = tgpu.fn([d.u32], d.vec3f)((seed) => {
  'use gpu';
  return d.vec3f(hash1(seed * 3 + 0), hash1(seed * 3 + 1), hash1(seed * 3 + 2));
});

/** Cosine-weighted hemisphere sample around `n`. */
export const cosineHemisphere = tgpu.fn([d.vec3f, d.vec2f], d.vec3f)((n, uv) => {
  'use gpu';
  const a = uv.x * 6.2831853;
  const r = std.sqrt(uv.y);
  const z = std.sqrt(std.max(0, 1 - uv.y));
  const t = std.select(d.vec3f(1, 0, 0), d.vec3f(0, 1, 0), std.abs(n.y) < 0.9);
  const tx = std.normalize(std.cross(t, n));
  const ty = std.cross(n, tx);
  return std.normalize(tx * (r * std.cos(a)) + ty * (r * std.sin(a)) + n * z);
});

/** Smooth max with blend weight (h=1 -> `a` dominates). */
export const smaxPolyWeighted = tgpu.fn([d.f32, d.f32, d.f32], d.vec2f)((a, b, k) => {
  'use gpu';
  const kk = std.max(k, 1e-6);
  const h = std.clamp(0.5 - 0.5 * (b - a) / kk, 0, 1);
  return d.vec2f(std.mix(b, a, h) + kk * h * (1 - h), h);
});
