import type { BrushDesc, BrushKindName, BrushOpName } from '../field/brush.ts';

/**
 * One primitive in a shape, plus the CSG operation that folds it into everything
 * before it. This is {@link BrushDesc} with the material named rather than indexed -
 * a game says `'stone'`, the palette turns that into a slot number.
 */
export interface ShapeNode extends Omit<BrushDesc, 'material' | 'onlyMaterial'> {
  material?: string | number;
  onlyMaterial?: string | number;
}

/**
 * A shape is a *list*, not a tree, because that is what the GPU evaluates: the field
 * is `N` primitives folded left to right with smooth min/max (Claybook GDC'18 slide 9),
 * and the sparse tile grid only ever loops over the primitives that touch a tile, so
 * cost does not grow with total count.
 *
 * `union` concatenates and `cut` concatenates with the operations flipped, which is
 * exactly right as long as a subtraction does not itself contain a subtraction - see
 * {@link cut}. Everything else about the tree survives the flattening.
 *
 * Instances are immutable; every method returns a new shape.
 */
export class Shape {
  readonly nodes: readonly ShapeNode[];

  constructor(nodes: readonly ShapeNode[]) {
    this.nodes = nodes;
  }

  private map(f: (n: ShapeNode) => ShapeNode): Shape {
    return new Shape(this.nodes.map(f));
  }

  /** Moves the whole shape. Composes: `.at(a).at(b)` lands at `a + b`. */
  at(pos: readonly [number, number, number]): Shape {
    return this.map((n) => ({
      ...n,
      pos: [(n.pos?.[0] ?? 0) + pos[0], (n.pos?.[1] ?? 0) + pos[1], (n.pos?.[2] ?? 0) + pos[2]],
    }));
  }

  /** Euler radians, X then Y then Z, about each primitive's own origin. */
  rotate(euler: readonly [number, number, number]): Shape {
    return this.map((n) => ({ ...n, euler: [euler[0], euler[1], euler[2]] }));
  }

  /**
   * Rotation applied after {@link Shape.rotate}, as an xyzw unit quaternion. Runtime
   * orientations arrive as quaternions - a body's fitted rotation, a lerp between two
   * frames - and Euler angles cannot take one without going through a conversion that
   * loses the composition.
   */
  turn(q: readonly [number, number, number, number]): Shape {
    return this.map((n) => ({ ...n, quat: [q[0], q[1], q[2], q[3]] }));
  }

  /** Uniform scale about each primitive's own origin. */
  scale(s: number): Shape {
    return this.map((n) => ({ ...n, scale: (n.scale ?? 1) * s }));
  }

  /** Palette slot name, or a raw index. Applies to every primitive that has none yet. */
  material(m: string | number): Shape {
    return this.map((n) => (n.material === undefined ? { ...n, material: m } : n));
  }

  /**
   * Restricts the shape to whatever is already made of `m`, leaving every other
   * material untouched. This is what makes one substance deformable and another one
   * not: `cut(...).only('clay')` dents the clay and stops at the stone under it.
   */
  only(m: string | number): Shape {
    return this.map((n) => ({ ...n, onlyMaterial: m }));
  }

  /** Blend radius of this shape's CSG seams, world units. */
  smooth(r: number): Shape {
    return this.map((n) => ({ ...n, smooth: r }));
  }
}

const prim = (kind: BrushKindName, n: Omit<ShapeNode, 'kind'>): Shape =>
  new Shape([{ kind, ...n }]);

/**
 * Primitives are authored at the origin in their own local space and moved with
 * `.at()`. One form each, no positional overloads: `sdf.sphere(2).at([3, 1, 2])`.
 */
export const sdf = {
  sphere: (radius: number) => prim('sphere', { size: radius }),
  box: (half: readonly [number, number, number]) => prim('box', { size: half }),
  roundBox: (half: readonly [number, number, number], corner: number) =>
    prim('roundBox', { size: half, radius: corner }),
  /** Along Y, `halfLength` from the origin to each cap centre. */
  capsule: (halfLength: number, radius: number) =>
    prim('capsule', { size: [0, halfLength, 0], radius }),
  torus: (ring: number, tube: number) => prim('torus', { size: ring, radius: tube }),
  /** Along Y. */
  cylinder: (radius: number, halfHeight: number) =>
    prim('cylinder', { size: [radius, halfHeight, radius] }),
  /** Infinite horizontal half-space; everything below `y` is solid. */
  plane: (y = 0) => prim('plane', { radius: y }),
  boxFrame: (half: readonly [number, number, number], thickness: number) =>
    prim('boxFrame', { size: half, radius: thickness }),

  /** Smooth union of everything given. */
  union: (...shapes: readonly Shape[]): Shape =>
    new Shape(shapes.flatMap((s) => s.nodes)),

  /**
   * `a` minus everything after it.
   *
   * The flattening flips each removed primitive's operation, which reproduces the tree
   * exactly for `a \ (b | c)` - the common case, one solid with several holes. It
   * cannot reproduce `a \ (b \ c)`, because a fold has nowhere to put the inner
   * result, so that throws rather than silently building the wrong level.
   */
  cut: (a: Shape, ...rest: readonly Shape[]): Shape =>
    new Shape([
      ...a.nodes,
      ...rest.flatMap((s) =>
        s.nodes.map((n): ShapeNode => {
          if (n.op === 'cut') {
            throw new Error(
              'sdf.cut: a subtraction inside a subtraction cannot be flattened into a '
                + 'brush fold. Rewrite `cut(a, cut(b, c))` as a union of the pieces you want removed.',
            );
          }
          return { ...n, op: n.op === 'paint' ? 'paint' : 'cut' };
        }),
      ),
    ]),

  /** Recolours whatever is already inside `region`, without changing the surface. */
  paint: (region: Shape, material: string | number): Shape =>
    new Shape(region.nodes.map((n) => ({ ...n, op: 'paint' as BrushOpName, material }))),

  /** An empty shape. Unions to a no-op; useful as a fold seed. */
  none: (): Shape => new Shape([]),
};

/** Turns named materials into palette slots, yielding plain engine brushes. */
export function compileShape(
  shape: Shape,
  resolveMaterial: (name: string) => number,
): BrushDesc[] {
  return shape.nodes.map((n) => ({
    ...n,
    material: typeof n.material === 'string' ? resolveMaterial(n.material) : (n.material ?? 0),
    onlyMaterial: typeof n.onlyMaterial === 'string'
      ? resolveMaterial(n.onlyMaterial)
      : n.onlyMaterial,
  }));
}

/** Half-extent of one primitive about its own origin, before its `pos` offset. */
function nodeReach(n: ShapeNode): number {
  const s = n.size;
  const size: [number, number, number] = typeof s === 'number'
    ? [s, s, s]
    : s ? [s[0], s[1], s[2]] : [1, 1, 1];
  const r = n.radius ?? 0;
  const scale = n.scale ?? 1;
  switch (n.kind) {
    case 'sphere':
      return size[0] * scale;
    case 'capsule':
      return (size[1] + r) * scale;
    case 'torus':
      return (size[0] + r) * scale;
    case 'plane':
      return Number.POSITIVE_INFINITY;
    default:
      return (Math.hypot(size[0], size[1], size[2]) + r) * scale;
  }
}

/**
 * Conservative sphere around a shape's *additive* primitives. Subtractions and paints
 * never grow a body, so they are ignored; a shape made only of those has no bounds.
 * Used to size a clay body's extraction box, its collider and its bake, so those are
 * derived from the shape instead of being a number the game has to keep in sync.
 */
export function shapeBounds(shape: Shape): {
  center: [number, number, number];
  radius: number;
} {
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (const n of shape.nodes) {
    if (n.op === 'cut' || n.op === 'paint') {
      continue;
    }
    const reach = nodeReach(n) + (n.smooth ?? 0);
    const p = n.pos ?? [0, 0, 0];
    lo = lo.map((v, i) => Math.min(v, p[i]! - reach));
    hi = hi.map((v, i) => Math.max(v, p[i]! + reach));
  }
  if (!Number.isFinite(lo[0]!) || !Number.isFinite(hi[0]!)) {
    throw new Error('shapeBounds: shape has no additive primitive with finite extent');
  }
  const center: [number, number, number] = [
    (lo[0]! + hi[0]!) / 2, (lo[1]! + hi[1]!) / 2, (lo[2]! + hi[2]!) / 2,
  ];
  return {
    center,
    radius: Math.max(hi[0]! - center[0]!, hi[1]! - center[1]!, hi[2]! - center[2]!),
  };
}
