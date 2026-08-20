import tgpu, { d, std } from 'typegpu';
import * as sdf from '@typegpu/sdf';
import { quatConj, quatRotate, sminPolyWeighted, smaxPolyWeighted } from '../math/gpu.ts';

/**
 * A brush is one analytic primitive with a rigid transform, a CSG operation and a
 * material id. The world field is `N` brushes folded together with exponential-ish
 * smooth min/max (Claybook GDC'18 slide 9). Claybook baked each brush into a small
 * offline volume texture; we evaluate the primitive analytically instead, which is
 * exact at every mip level and removes the brush-asset pipeline. The important
 * property is preserved: runtime cost does not depend on total brush count, because
 * the sparse tile grid only ever loops over the brushes that touch a tile.
 */
export const BrushKind = {
  sphere: 0,
  box: 1,
  roundBox: 2,
  capsule: 3,
  torus: 4,
  cylinder: 5,
  plane: 6,
  boxFrame: 7,
} as const;
export type BrushKindName = keyof typeof BrushKind;

export const BrushOp = {
  /** Smooth union. */
  add: 0,
  /** Smooth subtraction. */
  cut: 1,
  /** Material-only: recolours whatever is already inside the primitive. */
  paint: 2,
} as const;
export type BrushOpName = keyof typeof BrushOp;

export const Brush = d.struct({
  /** World position of the primitive's origin. */
  pos: d.vec3f,
  /** Uniform scale. */
  scale: d.f32,
  /** Orientation, xyzw unit quaternion. */
  rot: d.vec4f,
  /** Primary extents, meaning depends on `kind`. */
  size: d.vec3f,
  /** Secondary radius (corner radius / tube radius / plane offset). */
  radius: d.f32,
  material: d.f32,
  /** Blend radius of the CSG operation, in world units. WGSL reserves `smooth`. */
  blend: d.f32,
  /** Conservative influence radius around `pos`, used for tile culling. */
  bound: d.f32,
  /** Material this brush is restricted to, or `-1` for no restriction. */
  mask: d.f32,
  kind: d.u32,
  op: d.u32,
  _padB: d.vec2u,
});
export type BrushValue = d.InferInput<typeof Brush>;

/** Unsigned/signed distance of a primitive in its own local space. */
export const evalBrushLocal = tgpu.fn([d.u32, d.vec3f, d.vec3f, d.f32], d.f32)(
  (kind, p, size, radius) => {
    'use gpu';
    if (kind === BrushKind.sphere) {
      return sdf.sdSphere(p, size.x);
    }
    if (kind === BrushKind.box) {
      return sdf.sdBox3d(p, size);
    }
    if (kind === BrushKind.roundBox) {
      return sdf.sdRoundedBox3d(p, size, radius);
    }
    if (kind === BrushKind.capsule) {
      return sdf.sdCapsule(p, d.vec3f(0, -size.y, 0), d.vec3f(0, size.y, 0), radius);
    }
    if (kind === BrushKind.torus) {
      const q = d.vec2f(std.length(p.xz) - size.x, p.y);
      return std.length(q) - radius;
    }
    if (kind === BrushKind.cylinder) {
      const dxy = d.vec2f(std.length(p.xz) - size.x, std.abs(p.y) - size.y);
      return std.min(std.max(dxy.x, dxy.y), 0) + std.length(std.max(dxy, d.vec2f()));
    }
    if (kind === BrushKind.plane) {
      return p.y - radius;
    }
    return sdf.sdBoxFrame3d(p, size, radius);
  },
);

/** World-space distance of a single brush. */
export const evalBrush = tgpu.fn([Brush, d.vec3f], d.f32)((brush, p) => {
  'use gpu';
  const invScale = 1 / std.max(brush.scale, 1e-5);
  const local = quatRotate(quatConj(brush.rot), p - brush.pos) * invScale;
  return evalBrushLocal(brush.kind, local, brush.size, brush.radius) * brush.scale;
});

/**
 * Fold one brush into an accumulator of `(distance, material)`.
 * Material follows the smooth-blend weight so the seam between two brushes gets a
 * gradient rather than a hard line.
 */
export const applyBrush = tgpu.fn([d.vec2f, Brush, d.vec3f], d.vec2f)((acc, brush, p) => {
  'use gpu';
  const dist = evalBrush(brush, p);
  // A masked brush only acts where the field already reads as its material, which is
  // what makes one substance deformable and another one not. The weight ramps rather
  // than steps because `acc.y` is itself a blend across every seam.
  //
  // ponytail: mixing the distance by this weight is not exactly 1-Lipschitz across a
  // material seam, so a mask whose edge sits inside the blend band can overstate how
  // far a ray may step. Narrow seams keep it under the tracer's mip slack; the upgrade
  // path is a separate hardness channel that scales the carve depth instead of gating it.
  const keep = std.select(
    1,
    1 - std.smoothstep(0.25, 0.75, std.abs(acc.y - brush.mask)),
    brush.mask >= 0,
  );
  if (brush.op === BrushOp.add) {
    const r = sminPolyWeighted(dist, acc.x, brush.blend);
    return std.mix(acc, d.vec2f(r.x, std.mix(acc.y, brush.material, r.y)), keep);
  }
  if (brush.op === BrushOp.cut) {
    const r = smaxPolyWeighted(-dist, acc.x, brush.blend);
    return std.mix(acc, d.vec2f(r.x, std.mix(acc.y, brush.material, r.y * 0.85)), keep);
  }
  // paint: distance untouched, material bleeds in over the smooth band.
  const w = 1 - std.smoothstep(-std.max(brush.blend, 1e-4), 0, dist);
  return d.vec2f(acc.x, std.mix(acc.y, brush.material, w * keep));
});

/** CPU-side brush description. `layer` only orders the upload; the GPU applies buffer order. */
export interface BrushDesc {
  kind: BrushKindName;
  op?: BrushOpName;
  pos?: readonly [number, number, number];
  /** Euler angles in radians, applied X then Y then Z. */
  euler?: readonly [number, number, number];
  /**
   * Extra rotation applied *after* {@link BrushDesc.euler}, xyzw unit quaternion. This is
   * the slot a runtime orientation goes in - a physics body's fitted rotation - so it
   * composes with whatever authored rotation the primitive already had.
   */
  quat?: readonly [number, number, number, number];
  scale?: number;
  size?: readonly [number, number, number] | number;
  radius?: number;
  material?: number;
  smooth?: number;
  /** Restricts the brush to voxels already reading this material. */
  onlyMaterial?: number;
  layer?: number;
}

function eulerToQuat(x: number, y: number, z: number): [number, number, number, number] {
  const [cx, sx] = [Math.cos(x / 2), Math.sin(x / 2)];
  const [cy, sy] = [Math.cos(y / 2), Math.sin(y / 2)];
  const [cz, sz] = [Math.cos(z / 2), Math.sin(z / 2)];
  return [
    sx * cy * cz + cx * sy * sz,
    cx * sy * cz - sx * cy * sz,
    cx * cy * sz + sx * sy * cz,
    cx * cy * cz - sx * sy * sz,
  ];
}

/** Hamilton product, xyzw. `a` is applied after `b`. */
function quatMul(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
): [number, number, number, number] {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

/** Conservative influence radius: primitive extent + smooth blend band. */
function brushBound(kind: BrushKindName, size: [number, number, number], radius: number, scale: number, smooth: number): number {
  const [sx, sy, sz] = size;
  let r: number;
  switch (kind) {
    case 'sphere':
      r = sx;
      break;
    case 'box':
    case 'boxFrame':
      r = Math.hypot(sx, sy, sz) + radius;
      break;
    case 'roundBox':
      r = Math.hypot(sx, sy, sz) + radius;
      break;
    case 'capsule':
      r = sy + radius;
      break;
    case 'torus':
      r = sx + radius;
      break;
    case 'cylinder':
      r = Math.hypot(sx, sy);
      break;
    case 'plane':
      // Infinite: never culled.
      return Number.POSITIVE_INFINITY;
  }
  return r * scale + smooth;
}

export function makeBrush(desc: BrushDesc): BrushValue {
  const size: [number, number, number] = typeof desc.size === 'number'
    ? [desc.size, desc.size, desc.size]
    : desc.size
      ? [desc.size[0], desc.size[1], desc.size[2]]
      : [1, 1, 1];
  const scale = desc.scale ?? 1;
  const radius = desc.radius ?? 0;
  const smooth = desc.smooth ?? 0.02;
  const [ex, ey, ez] = desc.euler ?? [0, 0, 0];
  const bound = brushBound(desc.kind, size, radius, scale, smooth);
  return {
    pos: desc.pos ? [desc.pos[0], desc.pos[1], desc.pos[2]] : [0, 0, 0],
    scale,
    rot: desc.quat ? quatMul(desc.quat, eulerToQuat(ex, ey, ez)) : eulerToQuat(ex, ey, ez),
    size,
    radius,
    material: desc.material ?? 0,
    blend: smooth,
    // A finite (but huge) bound keeps the GPU-side cull test free of inf/NaN.
    bound: Number.isFinite(bound) ? bound : 1e9,
    mask: desc.onlyMaterial ?? -1,
    kind: BrushKind[desc.kind],
    op: BrushOp[desc.op ?? 'add'],
    _padB: [0, 0],
  };
}

/** Sorts by layer so later layers overwrite earlier ones, then flattens. */
export function compileBrushes(list: readonly BrushDesc[]): BrushValue[] {
  return [...list]
    .map((b, i) => ({ b, i }))
    .sort((a, x) => (a.b.layer ?? 0) - (x.b.layer ?? 0) || a.i - x.i)
    .map(({ b }) => makeBrush(b));
}
