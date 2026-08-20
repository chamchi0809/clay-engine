import tgpu, { d, std } from 'typegpu';
import * as sdf from '@typegpu/sdf';
import { quatConj, quatRotate, sminPolyWeighted, smaxPolyWeighted } from '../math/gpu.ts';
import type { FieldGroups } from '../trace/field.ts';

/**
 * A brush is one primitive with a rigid transform, a CSG operation and a material id. The
 * world field is `N` brushes folded together with exponential-ish smooth min/max (Claybook
 * GDC'18 slide 9). Claybook baked every brush into a small offline volume texture; we
 * evaluate the analytic ones analytically instead, which is exact at every mip level and
 * removes the brush-asset pipeline for the common case. The important property is preserved:
 * runtime cost does not depend on total brush count, because the sparse tile grid only ever
 * loops over the brushes that touch a tile.
 *
 * The eight analytic kinds are the ones with a closed form worth hardcoding. `volume` is the
 * escape hatch for a shape that has none - a baked mesh - and reads a slot out of a
 * {@link BrushVolumeSource}. Anything else is a *custom kind*: see {@link BrushSet}.
 */
export const BuiltinBrushKind = {
  sphere: 0,
  box: 1,
  roundBox: 2,
  capsule: 3,
  torus: 4,
  cylinder: 5,
  plane: 6,
  boxFrame: 7,
  /** Samples a baked SDF slot. Only present when the set was given an atlas. */
  volume: 8,
} as const;
export type BuiltinBrushKindName = keyof typeof BuiltinBrushKind;

/** First id handed out to a custom kind. */
export const CUSTOM_KIND_BASE = 9;

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
  /**
   * Atlas slot for the `volume` kind, ignored by every other one. This was the struct's
   * second padding word, so baked meshes cost no extra bandwidth in the brush buffer.
   */
  slot: d.u32,
  _padB: d.u32,
});
export type BrushValue = d.InferInput<typeof Brush>;
/** The brush struct as a shader sees it - what `evalBrush`/`applyBrush` are handed. */
export type BrushIn = d.Infer<typeof Brush>;

/**
 * A user-defined primitive.
 *
 * The distance function is GPU code and looks like it - this is the one place the game API
 * asks for a `'use gpu'` closure, because a new primitive *is* a shader edit. It is compiled
 * into the brush fold, so a custom kind is a first-class brush: it bakes into the world, it
 * carves with `cut`, it respects `.only()`, and a soft body can morph into it.
 */
export interface CustomBrush {
  /**
   * Signed distance in the primitive's own local space, at unit scale. `size` and `radius`
   * are whatever the shape expression passed in.
   *
   * It must be a real distance function - 1-Lipschitz - or the tracer will overshoot it and
   * rays will tunnel through the surface. Scaling a distance down (`* 0.5`) is safe;
   * scaling it up is not.
   */
  sdf: (p: d.v3f, size: d.v3f, radius: number) => number;
  /**
   * Conservative influence radius about the primitive's origin, at unit scale and before
   * the blend band. Under-reporting it is the one unrecoverable mistake here: the tile cull
   * will skip tiles the brush actually reaches into, and the shape gets clipped along tile
   * boundaries in a way that looks like a corrupted bake.
   */
  bound: (size: readonly [number, number, number], radius: number) => number;
}

/** What the `volume` kind needs from a baked-SDF atlas. Implemented by `BrushAtlas`. */
export interface BrushVolumeSource {
  /**
   * Normalised distance of `slot` at `q`, where `q` is the query in the slot's own
   * `[-1, 1]^3` box. The result is in those same normalised units.
   */
  readonly sampleSlot: (slot: number, q: d.v3f) => number;
  /** Bind groups any pipeline folding a `volume` brush must be `.with()`-ed. */
  readonly groups: FieldGroups;
}

/** CPU-side brush description. `layer` only orders the upload; the GPU applies buffer order. */
export interface BrushDesc {
  /** A builtin kind name, or the name a custom kind was registered under. */
  kind: string;
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
  /** Atlas slot, for the `volume` kind. */
  slot?: number;
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

export function normalizeSize(
  size: readonly [number, number, number] | number | undefined,
): [number, number, number] {
  return typeof size === 'number'
    ? [size, size, size]
    : size
      ? [size[0], size[1], size[2]]
      : [1, 1, 1];
}

/** Conservative influence radius of a builtin kind, at unit scale, before the blend band. */
function builtinBound(
  kind: BuiltinBrushKindName,
  size: readonly [number, number, number],
  radius: number,
): number {
  const [sx, sy, sz] = size;
  switch (kind) {
    case 'sphere':
      return sx;
    case 'box':
    case 'boxFrame':
    case 'roundBox':
      return Math.hypot(sx, sy, sz) + radius;
    case 'capsule':
      return sy + radius;
    case 'torus':
      return sx + radius;
    case 'cylinder':
      return Math.hypot(sx, sy);
    case 'plane':
      // Infinite: never culled.
      return Number.POSITIVE_INFINITY;
    case 'volume':
      // The bake box is a cube of half-extent `size.x`; its diagonal covers any surface in it.
      return sx * Math.sqrt(3);
  }
}

/**
 * The analytic primitives. `volume` is absent on purpose - it needs an atlas, so it is
 * chained on top by {@link BrushSet} only when there is one, and falls through to "empty"
 * when there is not.
 */
const evalAnalyticLocal = tgpu.fn([d.u32, d.vec3f, d.vec3f, d.f32], d.f32)(
  (kind, p, size, radius) => {
    'use gpu';
    if (kind === BuiltinBrushKind.sphere) {
      return sdf.sdSphere(p, size.x);
    }
    if (kind === BuiltinBrushKind.box) {
      return sdf.sdBox3d(p, size);
    }
    if (kind === BuiltinBrushKind.roundBox) {
      return sdf.sdRoundedBox3d(p, size, radius);
    }
    if (kind === BuiltinBrushKind.capsule) {
      return sdf.sdCapsule(p, d.vec3f(0, -size.y, 0), d.vec3f(0, size.y, 0), radius);
    }
    if (kind === BuiltinBrushKind.torus) {
      const q = d.vec2f(std.length(p.xz) - size.x, p.y);
      return std.length(q) - radius;
    }
    if (kind === BuiltinBrushKind.cylinder) {
      const dxy = d.vec2f(std.length(p.xz) - size.x, std.abs(p.y) - size.y);
      return std.min(std.max(dxy.x, dxy.y), 0) + std.length(std.max(dxy, d.vec2f()));
    }
    if (kind === BuiltinBrushKind.plane) {
      return p.y - radius;
    }
    if (kind === BuiltinBrushKind.boxFrame) {
      return sdf.sdBoxFrame3d(p, size, radius);
    }
    // An unknown kind reads as empty rather than as a surface at the origin: a set built
    // without an atlas must ignore a `volume` brush, not paint a blob where it was.
    return 1e9;
  },
);

export interface BrushSetOptions {
  /**
   * Custom kinds, by the name a shape expression refers to them by. Registered once, when
   * the set is made, because the fold is compiled into every pipeline that bakes or edits a
   * field - there is no way to add a kind to a shader that already exists.
   */
  custom?: Readonly<Record<string, CustomBrush>>;
  /** Enables the `volume` kind. */
  atlas?: BrushVolumeSource | null;
}

/**
 * The set of primitives a field is built out of, and the shader functions that fold them.
 *
 * This exists as an object rather than as module-level functions because the fold is
 * *generated*: `evalLocal` is an if-chain over the kinds this set knows, so a set with three
 * custom brushes and a mesh atlas produces different WGSL than the default one. Every
 * pipeline that bakes or edits a volume bakes one set's fold into itself, which is why a set
 * is fixed before any pipeline is built.
 */
export class BrushSet {
  /** Kind name to shader id, builtins included. */
  readonly kindIds: ReadonlyMap<string, number>;
  /** Bind groups any pipeline using this set's fold must be `.with()`-ed. */
  readonly groups: FieldGroups;
  /** `(kind, p, size, radius, slot) -> distance` in the primitive's local space. */
  readonly evalLocal: (
    kind: number,
    p: d.v3f,
    size: d.v3f,
    radius: number,
    slot: number,
  ) => number;
  /** World-space distance of a single brush. */
  readonly evalBrush: (brush: BrushIn, p: d.v3f) => number;
  /** Folds one brush into an accumulator of `(distance, material)`. */
  readonly applyBrush: (acc: d.v2f, brush: BrushIn, p: d.v3f) => d.v2f;

  private readonly customBounds: ReadonlyMap<string, CustomBrush['bound']>;

  constructor(options: BrushSetOptions = {}) {
    const custom = options.custom ?? {};
    const atlas = options.atlas ?? null;

    const ids = new Map<string, number>(
      Object.entries(BuiltinBrushKind).map(([name, id]) => [name, id]),
    );
    const bounds = new Map<string, CustomBrush['bound']>();
    const chained: { id: number; sdf: CustomBrush['sdf'] }[] = [];
    let next = CUSTOM_KIND_BASE;
    for (const [name, spec] of Object.entries(custom)) {
      if (ids.has(name)) {
        throw new Error(
          `BrushSet: '${name}' is a builtin primitive. Pick another name for the custom one.`,
        );
      }
      ids.set(name, next);
      bounds.set(name, spec.bound);
      chained.push({ id: next, sdf: spec.sdf });
      next++;
    }
    this.kindIds = ids;
    this.customBounds = bounds;
    this.groups = atlas ? atlas.groups : [];

    // Each link is its own `tgpu.fn` so the return type is pinned: an inferred chain mixes
    // the tail's abstract-int fallback with the branches' f32 and fails to resolve.
    let chain = evalAnalyticLocal as unknown as (
      kind: number,
      p: d.v3f,
      size: d.v3f,
      radius: number,
    ) => number;
    for (const { id, sdf: own } of chained) {
      const prev = chain;
      chain = tgpu.fn([d.u32, d.vec3f, d.vec3f, d.f32], d.f32)((kind, p, size, radius) => {
        'use gpu';
        if (kind === id) {
          return own(p, size, radius);
        }
        return prev(kind, p, size, radius);
      });
    }
    const analyticAndCustom = chain;

    // `volume` is chained last and separately, because it is the one kind that needs the
    // slot index, and threading a slot through every custom closure to serve one builtin
    // would put an argument nobody uses in the public signature.
    if (atlas) {
      const sampleSlot = atlas.sampleSlot;
      this.evalLocal = tgpu.fn([d.u32, d.vec3f, d.vec3f, d.f32, d.u32], d.f32)(
        (kind, p, size, radius, slot) => {
          'use gpu';
          if (kind === BuiltinBrushKind.volume) {
            // The bake box is a cube, so one half-extent describes it and the field stays
            // isotropic - which a non-uniform squash of a baked mesh would not.
            const h = std.max(size.x, 1e-5);
            // Outside the box the distance to the box is a valid lower bound (the surface
            // is strictly inside it) and avoids reading clamped border texels.
            const dBox = sdf.sdBox3d(p, d.vec3f(h));
            if (dBox > 0) {
              return dBox;
            }
            return sampleSlot(slot, p / h) * h;
          }
          return analyticAndCustom(kind, p, size, radius);
        },
      );
    } else {
      this.evalLocal = tgpu.fn([d.u32, d.vec3f, d.vec3f, d.f32, d.u32], d.f32)(
        (kind, p, size, radius, _slot) => {
          'use gpu';
          return analyticAndCustom(kind, p, size, radius);
        },
      );
    }

    const evalLocal = this.evalLocal;
    this.evalBrush = tgpu.fn([Brush, d.vec3f], d.f32)((brush, p) => {
      'use gpu';
      const invScale = 1 / std.max(brush.scale, 1e-5);
      const local = quatRotate(quatConj(brush.rot), p - brush.pos) * invScale;
      return evalLocal(brush.kind, local, brush.size, brush.radius, brush.slot) * brush.scale;
    });

    const evalBrush = this.evalBrush;
    // Material follows the smooth-blend weight so the seam between two brushes gets a
    // gradient rather than a hard line.
    this.applyBrush = tgpu.fn([d.vec2f, Brush, d.vec3f], d.vec2f)((acc, brush, p) => {
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
  }

  /** Shader id of a kind. Throws on an unregistered name, which is otherwise silent. */
  kindId(kind: string): number {
    const id = this.kindIds.get(kind);
    if (id === undefined) {
      const known = [...this.kindIds.keys()].join(', ');
      throw new Error(
        `BrushSet: unknown primitive '${kind}'. Register it in \`Game.create({ brushes })\`. `
          + `Known: ${known}`,
      );
    }
    return id;
  }

  /**
   * Conservative influence radius about a primitive's origin, at unit scale and before the
   * blend band. Shared by the tile cull and by `shapeBounds`, so a custom kind only ever
   * declares it once.
   */
  reachOf(
    kind: string,
    size: readonly [number, number, number] | number | undefined,
    radius = 0,
  ): number {
    const s = normalizeSize(size);
    const custom = this.customBounds.get(kind);
    if (custom) {
      return custom(s, radius);
    }
    this.kindId(kind);
    return builtinBound(kind as BuiltinBrushKindName, s, radius);
  }

  make(desc: BrushDesc): BrushValue {
    const size = normalizeSize(desc.size);
    const scale = desc.scale ?? 1;
    const radius = desc.radius ?? 0;
    const smooth = desc.smooth ?? 0.02;
    const [ex, ey, ez] = desc.euler ?? [0, 0, 0];
    const bound = this.reachOf(desc.kind, size, radius) * scale + smooth;
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
      kind: this.kindId(desc.kind),
      op: BrushOp[desc.op ?? 'add'],
      slot: desc.slot ?? 0,
      _padB: 0,
    };
  }

  /** Sorts by layer so later layers overwrite earlier ones, then flattens. */
  compile(list: readonly BrushDesc[]): BrushValue[] {
    return [...list]
      .map((b, i) => ({ b, i }))
      .sort((a, x) => (a.b.layer ?? 0) - (x.b.layer ?? 0) || a.i - x.i)
      .map(({ b }) => this.make(b));
  }
}

/**
 * The eight analytic primitives and nothing else.
 *
 * Used by anything that builds a field without a game around it - `SdfScene`, the fluid
 * bake - and as the default for every field-side class, so registering custom brushes is
 * additive rather than something every call site has to thread through.
 */
export const defaultBrushSet = new BrushSet();

export const makeBrush = (desc: BrushDesc): BrushValue => defaultBrushSet.make(desc);
export const compileBrushes = (list: readonly BrushDesc[]): BrushValue[] =>
  defaultBrushSet.compile(list);
