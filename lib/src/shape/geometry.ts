/**
 * three.js's geometry catalogue, as triangles.
 *
 * Every constructor three.js ships - `BoxGeometry` through `WireframeGeometry` - has a
 * counterpart here under the same name, taking the same parameters by the same names and
 * producing the same surface. The difference is what comes out: a {@link MeshData}, which
 * is positions and indices and nothing else, because a distance field has no use for
 * normals or UVs. Shading comes from the field's own gradient and the material comes from
 * the brush, so the two attributes three.js spends most of its generator code on are the
 * two this engine would throw away.
 *
 * What a game does with one:
 *
 * ```ts
 * const knot = await game.loadMesh(geometry.torusKnot({ radius: 1, tube: 0.3 }));
 * level.shape = sdf.union(ground, sdf.mesh(knot).at([0, 2, 0]).material('stone'));
 * ```
 *
 * Two of these shapes are *solids* the engine already has a closed form for - see
 * `sdf.box`, `sdf.sphere`, `sdf.cone`, `sdf.octahedron` and the rest in `shape/sdf.ts`.
 * Prefer those: an analytic brush is exact at every mip level, costs no atlas slot and no
 * bake, and can be scaled and rotated for free. The generators here are for the shapes
 * with no closed form - a torus knot, a lathe, an extruded outline, a subdivided
 * polyhedron - and for feeding a mesh pipeline that already speaks three.js parameters.
 *
 * ## Surfaces with no inside
 *
 * `plane`, `circle`, `ring`, `shape`, an open `lathe` or `tube`, `edges` and `wireframe`
 * describe surfaces, not solids. A surface encloses no volume, so baking one as a solid
 * gives an empty field - correctly, and uselessly. Bake it with a thickness instead and
 * the baker offsets the surface into a shell of that thickness:
 *
 * ```ts
 * const dish = await game.loadMesh(geometry.lathe({ points }), { thickness: 0.05 });
 * ```
 *
 * `edges` and `wireframe` go further and hand back solid rods, since a line segment has
 * no surface either.
 */

import type { MeshData } from './mesh.ts';
import { offsetRing, signedArea, triangulateShape, type Shape2D, type Vec2 } from './polygon.ts';

export type { Shape2D, Vec2 };
export type Vec3 = readonly [number, number, number];

const TAU = Math.PI * 2;

/**
 * Positions and indices under construction.
 *
 * Vertices are not welded across faces or across a seam, exactly as in three.js: a box
 * has 24 corners and a full sphere repeats its first column at `phi = 2*pi`. Nothing
 * downstream cares - the baker reads triangles, and a duplicated position bakes to the
 * same distance as a shared one.
 */
class Builder {
  readonly positions: number[] = [];
  readonly indices: number[] = [];

  /** Appends a vertex and hands back its index. */
  vertex(x: number, y: number, z: number): number {
    this.positions.push(x, y, z);
    return this.positions.length / 3 - 1;
  }

  /** Appends a triangle, dropping the degenerate ones a pole or an apex produces. */
  tri(a: number, b: number, c: number): void {
    if (a !== b && b !== c && c !== a) {
      this.indices.push(a, b, c);
    }
  }

  /** Two triangles over four corners given in order around the quad. */
  quad(a: number, b: number, c: number, d: number): void {
    this.tri(a, b, c);
    this.tri(a, c, d);
  }

  data(): MeshData {
    if (this.indices.length === 0) {
      throw new Error('geometry: the parameters produced no triangles');
    }
    return {
      positions: new Float32Array(this.positions),
      indices: new Uint32Array(this.indices),
    };
  }
}

/** three.js clamps its segment counts rather than emitting a broken mesh; so do we. */
const segments = (value: number | undefined, fallback: number, min = 1): number =>
  Math.max(min, Math.floor(value ?? fallback));

// --- flat outlines --------------------------------------------------------

export interface PlaneOptions {
  width?: number;
  height?: number;
  widthSegments?: number;
  heightSegments?: number;
}

/** A rectangle in the XY plane, facing +Z. Flat: bake it with a `thickness`. */
export function plane(o: PlaneOptions = {}): MeshData {
  const width = o.width ?? 1;
  const height = o.height ?? 1;
  const gx = segments(o.widthSegments, 1);
  const gy = segments(o.heightSegments, 1);
  const b = new Builder();
  const segW = width / gx;
  const segH = height / gy;
  for (let iy = 0; iy <= gy; iy++) {
    const y = iy * segH - height / 2;
    for (let ix = 0; ix <= gx; ix++) {
      b.vertex(ix * segW - width / 2, -y, 0);
    }
  }
  for (let iy = 0; iy < gy; iy++) {
    for (let ix = 0; ix < gx; ix++) {
      const a = ix + (gx + 1) * iy;
      const c = ix + (gx + 1) * (iy + 1);
      b.tri(a, c, ix + 1 + (gx + 1) * iy);
      b.tri(c, ix + 1 + (gx + 1) * (iy + 1), ix + 1 + (gx + 1) * iy);
    }
  }
  return b.data();
}

export interface CircleOptions {
  radius?: number;
  segments?: number;
  thetaStart?: number;
  thetaLength?: number;
}

/** A disc in the XY plane, facing +Z. Flat: bake it with a `thickness`. */
export function circle(o: CircleOptions = {}): MeshData {
  const radius = o.radius ?? 1;
  const n = segments(o.segments, 32, 3);
  const thetaStart = o.thetaStart ?? 0;
  const thetaLength = o.thetaLength ?? TAU;
  const b = new Builder();
  const centre = b.vertex(0, 0, 0);
  for (let s = 0; s <= n; s++) {
    const theta = thetaStart + (s / n) * thetaLength;
    b.vertex(radius * Math.cos(theta), radius * Math.sin(theta), 0);
  }
  for (let i = 1; i <= n; i++) {
    b.tri(centre + i, centre + i + 1, centre);
  }
  return b.data();
}

export interface RingOptions {
  innerRadius?: number;
  outerRadius?: number;
  thetaSegments?: number;
  phiSegments?: number;
  thetaStart?: number;
  thetaLength?: number;
}

/** An annulus in the XY plane, facing +Z. Flat: bake it with a `thickness`. */
export function ring(o: RingOptions = {}): MeshData {
  const inner = o.innerRadius ?? 0.5;
  const outer = o.outerRadius ?? 1;
  const thetaSegments = segments(o.thetaSegments, 32, 3);
  const phiSegments = segments(o.phiSegments, 1);
  const thetaStart = o.thetaStart ?? 0;
  const thetaLength = o.thetaLength ?? TAU;
  const b = new Builder();
  for (let j = 0; j <= phiSegments; j++) {
    const r = inner + (j / phiSegments) * (outer - inner);
    for (let i = 0; i <= thetaSegments; i++) {
      const theta = thetaStart + (i / thetaSegments) * thetaLength;
      b.vertex(r * Math.cos(theta), r * Math.sin(theta), 0);
    }
  }
  for (let j = 0; j < phiSegments; j++) {
    const level = j * (thetaSegments + 1);
    for (let i = 0; i < thetaSegments; i++) {
      const a = i + level;
      const bb = a + thetaSegments + 1;
      b.tri(a, bb, a + 1);
      b.tri(bb, bb + 1, a + 1);
    }
  }
  return b.data();
}

export interface ShapeOptions {
  /** One outline with holes, or several. three.js's `Shape`, already sampled. */
  shapes: Shape2D | readonly Shape2D[];
}

/**
 * A triangulated outline in the XY plane, facing +Z. Flat: bake it with a `thickness`.
 *
 * three.js takes `Shape` objects made of curves and a `curveSegments` count to sample
 * them with. This takes the points: a three.js shape converts with
 * `shape.extractPoints(curveSegments)`, whose `{ shape, holes }` is a {@link Shape2D}
 * under different field names.
 */
export function shape(o: ShapeOptions): MeshData {
  const b = new Builder();
  for (const s of asShapes(o.shapes)) {
    const tri = triangulateShape(s);
    const base = b.positions.length / 3;
    for (const p of tri.points) {
      b.vertex(p[0], p[1], 0);
    }
    for (let i = 0; i < tri.indices.length; i += 3) {
      b.tri(base + tri.indices[i], base + tri.indices[i + 1], base + tri.indices[i + 2]);
    }
  }
  return b.data();
}

const asShapes = (s: Shape2D | readonly Shape2D[]): readonly Shape2D[] =>
  Array.isArray(s) ? (s as readonly Shape2D[]) : [s as Shape2D];

// --- solids of revolution -------------------------------------------------

export interface SphereOptions {
  radius?: number;
  widthSegments?: number;
  heightSegments?: number;
  phiStart?: number;
  phiLength?: number;
  thetaStart?: number;
  thetaLength?: number;
}

/** A sphere about the origin, poles on Y. `sdf.sphere` is the analytic one. */
export function sphere(o: SphereOptions = {}): MeshData {
  const radius = o.radius ?? 1;
  const widthSegments = segments(o.widthSegments, 32, 3);
  const heightSegments = segments(o.heightSegments, 16, 2);
  const phiStart = o.phiStart ?? 0;
  const phiLength = o.phiLength ?? TAU;
  const thetaStart = o.thetaStart ?? 0;
  const thetaLength = o.thetaLength ?? Math.PI;
  const thetaEnd = Math.min(thetaStart + thetaLength, Math.PI);

  const b = new Builder();
  const grid: number[][] = [];
  for (let iy = 0; iy <= heightSegments; iy++) {
    const row: number[] = [];
    const v = iy / heightSegments;
    const theta = thetaStart + v * thetaLength;
    for (let ix = 0; ix <= widthSegments; ix++) {
      const phi = phiStart + (ix / widthSegments) * phiLength;
      row.push(b.vertex(
        -radius * Math.cos(phi) * Math.sin(theta),
        radius * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta),
      ));
    }
    grid.push(row);
  }
  for (let iy = 0; iy < heightSegments; iy++) {
    for (let ix = 0; ix < widthSegments; ix++) {
      const a = grid[iy][ix + 1];
      const c = grid[iy + 1][ix];
      const d = grid[iy + 1][ix + 1];
      // The pole rows collapse to a point, so one of the two triangles is degenerate
      // there. three.js skips it by hand; `tri` drops it either way.
      if (iy !== 0 || thetaStart > 0) {
        b.tri(a, grid[iy][ix], d);
      }
      if (iy !== heightSegments - 1 || thetaEnd < Math.PI) {
        b.tri(grid[iy][ix], c, d);
      }
    }
  }
  return b.data();
}

export interface CylinderOptions {
  radiusTop?: number;
  radiusBottom?: number;
  height?: number;
  radialSegments?: number;
  heightSegments?: number;
  openEnded?: boolean;
  thetaStart?: number;
  thetaLength?: number;
}

/**
 * A cylinder or a truncated cone, standing on Y, centred on the origin.
 *
 * `sdf.cylinder` and `sdf.cappedCone` are the analytic ones, and cover everything here
 * except an open end or a partial sweep.
 */
export function cylinder(o: CylinderOptions = {}): MeshData {
  const radiusTop = o.radiusTop ?? 1;
  const radiusBottom = o.radiusBottom ?? 1;
  const height = o.height ?? 1;
  const radialSegments = segments(o.radialSegments, 32, 3);
  const heightSegments = segments(o.heightSegments, 1);
  const openEnded = o.openEnded ?? false;
  const thetaStart = o.thetaStart ?? 0;
  const thetaLength = o.thetaLength ?? TAU;
  const half = height / 2;

  const b = new Builder();
  const grid: number[][] = [];
  for (let iy = 0; iy <= heightSegments; iy++) {
    const row: number[] = [];
    const v = iy / heightSegments;
    const r = v * (radiusBottom - radiusTop) + radiusTop;
    for (let ix = 0; ix <= radialSegments; ix++) {
      const theta = (ix / radialSegments) * thetaLength + thetaStart;
      row.push(b.vertex(r * Math.sin(theta), -v * height + half, r * Math.cos(theta)));
    }
    grid.push(row);
  }
  for (let iy = 0; iy < heightSegments; iy++) {
    for (let ix = 0; ix < radialSegments; ix++) {
      const a = grid[iy][ix];
      const bb = grid[iy + 1][ix];
      const c = grid[iy + 1][ix + 1];
      const dd = grid[iy][ix + 1];
      b.tri(a, bb, dd);
      b.tri(bb, c, dd);
    }
  }
  if (!openEnded) {
    for (const top of [true, false]) {
      const r = top ? radiusTop : radiusBottom;
      if (r <= 0) {
        continue;
      }
      const sign = top ? 1 : -1;
      const centre = b.vertex(0, half * sign, 0);
      const first = b.positions.length / 3;
      for (let ix = 0; ix <= radialSegments; ix++) {
        const theta = (ix / radialSegments) * thetaLength + thetaStart;
        b.vertex(r * Math.sin(theta), half * sign, r * Math.cos(theta));
      }
      for (let ix = 0; ix < radialSegments; ix++) {
        const i = first + ix;
        if (top) {
          b.tri(i, i + 1, centre);
        } else {
          b.tri(i + 1, i, centre);
        }
      }
    }
  }
  return b.data();
}

export interface ConeOptions {
  radius?: number;
  height?: number;
  radialSegments?: number;
  heightSegments?: number;
  openEnded?: boolean;
  thetaStart?: number;
  thetaLength?: number;
}

/**
 * A cone standing on Y, apex up, centred on the origin. three.js's `ConeGeometry`, which
 * is a cylinder with no top. `sdf.cone` is the analytic one.
 */
export function cone(o: ConeOptions = {}): MeshData {
  return cylinder({
    radiusTop: 0,
    radiusBottom: o.radius ?? 1,
    height: o.height,
    radialSegments: o.radialSegments,
    heightSegments: o.heightSegments,
    openEnded: o.openEnded,
    thetaStart: o.thetaStart,
    thetaLength: o.thetaLength,
  });
}

export interface CapsuleOptions {
  radius?: number;
  /** Length of the *cylindrical* middle. Total height is `height + 2 * radius`. */
  height?: number;
  capSegments?: number;
  radialSegments?: number;
  heightSegments?: number;
}

/** A capsule along Y. `sdf.capsule` is the analytic one. */
export function capsule(o: CapsuleOptions = {}): MeshData {
  const radius = o.radius ?? 1;
  const height = o.height ?? 1;
  const capSegments = segments(o.capSegments, 4);
  const radialSegments = segments(o.radialSegments, 8, 3);
  const heightSegments = segments(o.heightSegments, 1);
  const half = height / 2;

  // The profile three.js lathes: a quarter arc up from the bottom pole, the straight
  // side, then a quarter arc to the top pole. Bottom to top, so the surface faces out.
  const points: Vec2[] = [];
  for (let i = 0; i <= capSegments; i++) {
    const a = Math.PI * 1.5 + (i / capSegments) * (Math.PI / 2);
    points.push([radius * Math.cos(a), -half + radius * Math.sin(a)]);
  }
  for (let i = 1; i < heightSegments; i++) {
    points.push([radius, -half + (i / heightSegments) * height]);
  }
  for (let i = 0; i <= capSegments; i++) {
    const a = (i / capSegments) * (Math.PI / 2);
    points.push([radius * Math.cos(a), half + radius * Math.sin(a)]);
  }
  return lathe({ points, segments: radialSegments });
}

export interface LatheOptions {
  /** The half-profile in the XY plane, `x` off the axis and `y` along it, bottom to top. */
  points: readonly Vec2[];
  segments?: number;
  phiStart?: number;
  phiLength?: number;
}

/**
 * A profile swept around the Y axis.
 *
 * Points run bottom to top, `x` away from the axis; that order is what puts the outside
 * of the surface outward. A profile that does not return to the axis at both ends leaves
 * an open shape - bake it with a `thickness`.
 */
export function lathe(o: LatheOptions): MeshData {
  const points = o.points;
  if (points.length < 2) {
    throw new Error('geometry.lathe: the profile needs at least two points');
  }
  const n = segments(o.segments, 12, 2);
  const phiStart = o.phiStart ?? 0;
  const phiLength = o.phiLength ?? TAU;
  const b = new Builder();
  for (let i = 0; i <= n; i++) {
    const phi = phiStart + (i / n) * phiLength;
    const sin = Math.sin(phi);
    const cos = Math.cos(phi);
    for (const p of points) {
      b.vertex(p[0] * sin, p[1], p[0] * cos);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < points.length - 1; j++) {
      const base = j + i * points.length;
      const a = base;
      const bb = base + points.length;
      b.tri(a, bb, base + 1);
      b.tri(bb + 1, base + 1, bb);
    }
  }
  return b.data();
}

// --- tori -----------------------------------------------------------------

export interface TorusOptions {
  radius?: number;
  tube?: number;
  radialSegments?: number;
  tubularSegments?: number;
  arc?: number;
}

/**
 * A torus in the XY plane, hole along Z - three.js's orientation.
 *
 * `sdf.torus` is the analytic one and lies in XZ with its hole along Y, because that is
 * the axis every other primitive in this engine stands on. Rotate whichever one you use
 * rather than assuming they agree.
 */
export function torus(o: TorusOptions = {}): MeshData {
  const radius = o.radius ?? 1;
  const tube = o.tube ?? 0.4;
  const radialSegments = segments(o.radialSegments, 12, 3);
  const tubularSegments = segments(o.tubularSegments, 48, 3);
  const arc = o.arc ?? TAU;
  const b = new Builder();
  for (let j = 0; j <= radialSegments; j++) {
    for (let i = 0; i <= tubularSegments; i++) {
      const u = (i / tubularSegments) * arc;
      const v = (j / radialSegments) * TAU;
      b.vertex(
        (radius + tube * Math.cos(v)) * Math.cos(u),
        (radius + tube * Math.cos(v)) * Math.sin(u),
        tube * Math.sin(v),
      );
    }
  }
  for (let j = 1; j <= radialSegments; j++) {
    for (let i = 1; i <= tubularSegments; i++) {
      const a = (tubularSegments + 1) * j + i - 1;
      const bb = (tubularSegments + 1) * (j - 1) + i - 1;
      const c = (tubularSegments + 1) * (j - 1) + i;
      const dd = (tubularSegments + 1) * j + i;
      b.tri(a, bb, dd);
      b.tri(bb, c, dd);
    }
  }
  return b.data();
}

export interface TorusKnotOptions {
  radius?: number;
  tube?: number;
  tubularSegments?: number;
  radialSegments?: number;
  /** Windings around the torus's axis of symmetry. */
  p?: number;
  /** Windings around the torus's inner circle. */
  q?: number;
}

/** A (p, q) torus knot. No closed form worth the name - this is what a bake is for. */
export function torusKnot(o: TorusKnotOptions = {}): MeshData {
  const radius = o.radius ?? 1;
  const tube = o.tube ?? 0.4;
  const tubularSegments = segments(o.tubularSegments, 64, 3);
  const radialSegments = segments(o.radialSegments, 8, 3);
  const p = o.p ?? 2;
  const q = o.q ?? 3;

  const onCurve = (u: number): Vec3 => {
    const cu = Math.cos(u);
    const su = Math.sin(u);
    const quOverP = (q / p) * u;
    const cs = Math.cos(quOverP);
    return [
      radius * (2 + cs) * 0.5 * cu,
      radius * (2 + cs) * su * 0.5,
      radius * Math.sin(quOverP) * 0.5,
    ];
  };

  const b = new Builder();
  for (let i = 0; i <= tubularSegments; i++) {
    const u = (i / tubularSegments) * p * TAU;
    const p1 = onCurve(u);
    const p2 = onCurve(u + 0.01);
    // Tangent and the curve's own centre direction give a frame without integrating one:
    // `N` here is the direction the curve bends in, which is stable along a knot.
    const t = sub(p2, p1);
    const n0 = add(p2, p1);
    const bn = normalize(cross(t, n0));
    const nn = normalize(cross(bn, t));
    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * TAU;
      const cx = -tube * Math.cos(v);
      const cy = tube * Math.sin(v);
      b.vertex(
        p1[0] + cx * nn[0] + cy * bn[0],
        p1[1] + cx * nn[1] + cy * bn[1],
        p1[2] + cx * nn[2] + cy * bn[2],
      );
    }
  }
  gridFaces(b, tubularSegments, radialSegments);
  return b.data();
}

export interface TubeOptions {
  /**
   * The spine, sampled at `t` in `[0, 1]`. three.js takes a `Curve` and walks it by arc
   * length; this walks `t` directly, so a curve whose speed varies gets segments whose
   * lengths vary with it.
   */
  path: (t: number) => Vec3;
  tubularSegments?: number;
  radius?: number;
  radialSegments?: number;
  /** Joins the last ring back to the first, and smooths the frame across the seam. */
  closed?: boolean;
}

/** A tube swept along a curve. Open at both ends - bake it with a `thickness`. */
export function tube(o: TubeOptions): MeshData {
  const tubularSegments = segments(o.tubularSegments, 64, 1);
  const radius = o.radius ?? 1;
  const radialSegments = segments(o.radialSegments, 8, 3);
  const closed = o.closed ?? false;
  const frames = frenetFrames(o.path, tubularSegments, closed);

  const b = new Builder();
  for (let i = 0; i <= tubularSegments; i++) {
    const point = o.path(closed && i === tubularSegments ? 0 : i / tubularSegments);
    const n = frames.normals[i];
    const bn = frames.binormals[i];
    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * TAU;
      const sin = Math.sin(v);
      const cos = -Math.cos(v);
      b.vertex(
        point[0] + radius * (cos * n[0] + sin * bn[0]),
        point[1] + radius * (cos * n[1] + sin * bn[1]),
        point[2] + radius * (cos * n[2] + sin * bn[2]),
      );
    }
  }
  gridFaces(b, tubularSegments, radialSegments);
  return b.data();
}

/** The `(a, b, d) (b, c, d)` fan over a `(rows + 1) x (cols + 1)` grid of vertices. */
function gridFaces(b: Builder, rows: number, cols: number): void {
  for (let j = 1; j <= rows; j++) {
    for (let i = 1; i <= cols; i++) {
      const a = (cols + 1) * (j - 1) + (i - 1);
      const bb = (cols + 1) * j + (i - 1);
      const c = (cols + 1) * j + i;
      const dd = (cols + 1) * (j - 1) + i;
      b.tri(a, bb, dd);
      b.tri(bb, c, dd);
    }
  }
}

// --- polyhedra ------------------------------------------------------------

export interface PolyhedronOptions {
  /** Vertex positions, xyz triples. Only their directions matter; `radius` sets the size. */
  vertices: ArrayLike<number>;
  /** Triangle indices into {@link vertices}, wound outward. */
  indices: ArrayLike<number>;
  radius?: number;
  /** Subdivisions of every face. Each level pushes the new vertices out to `radius`. */
  detail?: number;
}

/**
 * A convex polyhedron, optionally subdivided towards a sphere.
 *
 * `radius` is the circumradius: every vertex is pushed out to exactly that distance, which
 * is why subdividing turns any of these into a geodesic sphere.
 */
export function polyhedron(o: PolyhedronOptions): MeshData {
  const radius = o.radius ?? 1;
  const detail = Math.max(0, Math.floor(o.detail ?? 0));
  const b = new Builder();
  const at = (i: number): Vec3 => [o.vertices[i * 3], o.vertices[i * 3 + 1], o.vertices[i * 3 + 2]];
  const push = (p: Vec3): number => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1;
    return b.vertex((p[0] / len) * radius, (p[1] / len) * radius, (p[2] / len) * radius);
  };

  for (let f = 0; f < o.indices.length; f += 3) {
    // three.js's `subdivideFace`: a barycentric grid over the triangle, so every level of
    // detail keeps the original edges and splits them evenly.
    const a = at(o.indices[f]);
    const bv = at(o.indices[f + 1]);
    const c = at(o.indices[f + 2]);
    const cols = detail + 1;
    const grid: Vec3[][] = [];
    for (let i = 0; i <= cols; i++) {
      const aj = lerp(a, c, i / cols);
      const bj = lerp(bv, c, i / cols);
      const rows = cols - i;
      const row: Vec3[] = [];
      for (let j = 0; j <= rows; j++) {
        row.push(j === 0 && i === cols ? aj : lerp(aj, bj, j / rows));
      }
      grid.push(row);
    }
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < 2 * (cols - i) - 1; j++) {
        const k = Math.floor(j / 2);
        if (j % 2 === 0) {
          b.tri(push(grid[i][k + 1]), push(grid[i + 1][k]), push(grid[i][k]));
        } else {
          b.tri(push(grid[i][k + 1]), push(grid[i + 1][k + 1]), push(grid[i + 1][k]));
        }
      }
    }
  }
  return b.data();
}

export interface PlatonicOptions {
  radius?: number;
  detail?: number;
}

const PHI = (1 + Math.sqrt(5)) / 2;

/** A regular tetrahedron. `sdf.tetrahedron` is the analytic one. */
export function tetrahedron(o: PlatonicOptions = {}): MeshData {
  return polyhedron({
    vertices: [1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1],
    indices: [2, 1, 0, 0, 3, 2, 1, 3, 0, 2, 3, 1],
    ...o,
  });
}

/** A regular octahedron. `sdf.octahedron` is the analytic one. */
export function octahedron(o: PlatonicOptions = {}): MeshData {
  return polyhedron({
    vertices: [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1],
    indices: [0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2],
    ...o,
  });
}

/** A regular icosahedron. `sdf.icosahedron` is the analytic one. */
export function icosahedron(o: PlatonicOptions = {}): MeshData {
  return polyhedron({
    vertices: [
      -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI, 0,
      0, -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI,
      PHI, 0, -1, PHI, 0, 1, -PHI, 0, -1, -PHI, 0, 1,
    ],
    indices: [
      0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
      1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
      3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
      4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    ],
    ...o,
  });
}

/** A regular dodecahedron. `sdf.dodecahedron` is the analytic one. */
export function dodecahedron(o: PlatonicOptions = {}): MeshData {
  const r = 1 / PHI;
  return polyhedron({
    vertices: [
      -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1,
      1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1,
      0, -r, -PHI, 0, -r, PHI, 0, r, -PHI, 0, r, PHI,
      -r, -PHI, 0, -r, PHI, 0, r, -PHI, 0, r, PHI, 0,
      -PHI, 0, -r, PHI, 0, -r, -PHI, 0, r, PHI, 0, r,
    ],
    indices: [
      3, 11, 7, 3, 7, 15, 3, 15, 13, 7, 19, 17, 7, 17, 6, 7, 6, 15,
      17, 4, 8, 17, 8, 10, 17, 10, 6, 8, 0, 16, 8, 16, 2, 8, 2, 10,
      0, 12, 1, 0, 1, 18, 0, 18, 16, 6, 10, 2, 6, 2, 13, 6, 13, 15,
      2, 16, 18, 2, 18, 3, 2, 3, 13, 18, 1, 9, 18, 9, 11, 18, 11, 3,
      4, 14, 12, 4, 12, 0, 4, 0, 8, 11, 9, 5, 11, 5, 19, 11, 19, 7,
      19, 5, 14, 19, 14, 4, 19, 4, 17, 1, 12, 14, 1, 14, 5, 1, 5, 9,
    ],
    ...o,
  });
}

// --- the box --------------------------------------------------------------

export interface BoxOptions {
  width?: number;
  height?: number;
  depth?: number;
  widthSegments?: number;
  heightSegments?: number;
  depthSegments?: number;
}

/** A box about the origin. `sdf.box` is the analytic one. */
export function box(o: BoxOptions = {}): MeshData {
  const width = o.width ?? 1;
  const height = o.height ?? 1;
  const depth = o.depth ?? 1;
  const gx = segments(o.widthSegments, 1);
  const gy = segments(o.heightSegments, 1);
  const gz = segments(o.depthSegments, 1);
  const b = new Builder();

  // three.js's `buildPlane`, with the axes named by index: `u` and `v` span the face and
  // `w` is the one it is offset along.
  const face = (
    u: number, v: number, w: number,
    udir: number, vdir: number,
    su: number, sv: number, sw: number,
    gu: number, gv: number,
  ): void => {
    const base = b.positions.length / 3;
    const segU = su / gu;
    const segV = sv / gv;
    for (let iv = 0; iv <= gv; iv++) {
      const y = iv * segV - sv / 2;
      for (let iu = 0; iu <= gu; iu++) {
        const x = iu * segU - su / 2;
        const p: [number, number, number] = [0, 0, 0];
        p[u] = x * udir;
        p[v] = y * vdir;
        p[w] = sw / 2;
        b.vertex(p[0], p[1], p[2]);
      }
    }
    for (let iv = 0; iv < gv; iv++) {
      for (let iu = 0; iu < gu; iu++) {
        const a = base + iu + (gu + 1) * iv;
        const c = base + iu + (gu + 1) * (iv + 1);
        b.tri(a, c, a + 1);
        b.tri(c, c + 1, a + 1);
      }
    }
  };

  face(2, 1, 0, -1, -1, depth, height, width, gz, gy);
  face(2, 1, 0, 1, -1, depth, height, -width, gz, gy);
  face(0, 2, 1, 1, 1, width, depth, height, gx, gz);
  face(0, 2, 1, 1, -1, width, depth, -height, gx, gz);
  face(0, 1, 2, 1, -1, width, height, depth, gx, gy);
  face(0, 1, 2, -1, -1, width, height, -depth, gx, gy);
  return b.data();
}

// --- extrusion ------------------------------------------------------------

export interface ExtrudeOptions {
  /** One outline with holes, or several. See {@link ShapeOptions.shapes}. */
  shapes: Shape2D | readonly Shape2D[];
  /** How far along +Z to extrude. Ignored when {@link extrudePath} is given. */
  depth?: number;
  /** Subdivisions along the extrusion. */
  steps?: number;
  bevelEnabled?: boolean;
  /** How far the bevel reaches along the extrusion axis, at each end. */
  bevelThickness?: number;
  /** How far the bevel reaches in from the outline. */
  bevelSize?: number;
  /** A constant inset applied on top of the bevel. */
  bevelOffset?: number;
  bevelSegments?: number;
  /**
   * A spine to extrude along, sampled at `t` in `[0, 1]`, instead of straight along +Z.
   * As in three.js, a spine turns the bevel off: the two do not compose.
   */
  extrudePath?: (t: number) => Vec3;
}

/**
 * An outline swept into a solid, with three.js's bevel.
 *
 * The outline is triangulated once and reused as both lids; the walls run between
 * consecutive rings. A bevel adds rings at each end, each one contracted along the
 * outline's angle bisectors so the corners stay corners - the same construction
 * three.js's `getBevelVec` makes.
 */
export function extrude(o: ExtrudeOptions): MeshData {
  const depth = o.depth ?? 1;
  const steps = segments(o.steps, 1);
  const spine = o.extrudePath;
  const bevelEnabled = (o.bevelEnabled ?? true) && !spine;
  const bevelThickness = o.bevelThickness ?? 0.2;
  const bevelSize = o.bevelSize ?? bevelThickness - 0.1;
  const bevelOffset = o.bevelOffset ?? 0;
  const bevelSegments = bevelEnabled ? segments(o.bevelSegments, 3) : 0;

  const b = new Builder();
  for (const s of asShapes(o.shapes)) {
    const tri = triangulateShape(s);
    const ringOf = (index: number): Vec2[] => {
      const start = tri.rings[index];
      const end = index + 1 < tri.rings.length ? tri.rings[index + 1] : tri.points.length;
      return tri.points.slice(start, end);
    };
    const ringCount = tri.rings.length;
    // The triangulator hands back a counter-clockwise contour and clockwise holes, which
    // is exactly the winding `offsetRing` needs to know which way "in" is.
    const rings = Array.from({ length: ringCount }, (_, i) => ringOf(i));

    // Each layer is one ring of vertices: how far it is contracted, and where it sits.
    const layers: { inset: number; place: (p: Vec2) => Vec3 }[] = [];
    const flat = (z: number) => (p: Vec2): Vec3 => [p[0], p[1], z];
    if (spine) {
      const frames = frenetFrames(spine, steps, false);
      for (let i = 0; i <= steps; i++) {
        const point = spine(i / steps);
        const n = frames.normals[i];
        const bn = frames.binormals[i];
        layers.push({
          inset: 0,
          place: (p) => [
            point[0] + p[0] * n[0] + p[1] * bn[0],
            point[1] + p[0] * n[1] + p[1] * bn[1],
            point[2] + p[0] * n[2] + p[1] * bn[2],
          ],
        });
      }
    } else {
      for (let i = 0; i < bevelSegments; i++) {
        const t = i / bevelSegments;
        layers.push({
          inset: bevelSize * Math.sin(t * Math.PI / 2) + bevelOffset,
          place: flat(-bevelThickness * Math.cos(t * Math.PI / 2)),
        });
      }
      const full = bevelEnabled ? bevelSize + bevelOffset : 0;
      for (let i = 0; i <= steps; i++) {
        layers.push({ inset: full, place: flat((depth * i) / steps) });
      }
      for (let i = bevelSegments - 1; i >= 0; i--) {
        const t = i / bevelSegments;
        layers.push({
          inset: bevelSize * Math.sin(t * Math.PI / 2) + bevelOffset,
          place: flat(depth + bevelThickness * Math.cos(t * Math.PI / 2)),
        });
      }
    }

    // Vertices, layer by layer, each layer holding every ring in `tri.points` order - so
    // the lid triangulation indexes straight into a layer.
    const stride = tri.points.length;
    const base = b.positions.length / 3;
    for (const layer of layers) {
      for (let r = 0; r < ringCount; r++) {
        const ring = rings[r];
        const moved = layer.inset === 0
          ? ring
          : offsetRing(ring, r === 0 ? layer.inset : -layer.inset, r === 0);
        for (const p of moved) {
          const q = layer.place(p);
          b.vertex(q[0], q[1], q[2]);
        }
      }
    }

    // Lids. The triangulation is counter-clockwise seen from +Z, so the far one keeps its
    // winding and the near one is flipped.
    const last = base + (layers.length - 1) * stride;
    for (let i = 0; i < tri.indices.length; i += 3) {
      const [x, y, z] = [tri.indices[i], tri.indices[i + 1], tri.indices[i + 2]];
      b.tri(base + z, base + y, base + x);
      b.tri(last + x, last + y, last + z);
    }

    // Walls, one band per pair of layers, one loop per ring.
    for (let l = 0; l < layers.length - 1; l++) {
      const lo = base + l * stride;
      const hi = lo + stride;
      for (let r = 0; r < ringCount; r++) {
        const start = tri.rings[r];
        const count = rings[r].length;
        for (let i = 0; i < count; i++) {
          const a = start + i;
          const c = start + ((i + 1) % count);
          b.quad(lo + a, lo + c, hi + c, hi + a);
        }
      }
    }
  }
  return b.data();
}

// --- edges and wireframes -------------------------------------------------

export interface EdgesOptions {
  geometry: MeshData;
  /** Degrees. An edge shows when its two faces differ by more than this. */
  thresholdAngle?: number;
  /** Rod thickness. Defaults to 1% of the mesh's bounding-box diagonal. */
  thickness?: number;
}

/**
 * The sharp edges of a mesh, as solid rods.
 *
 * three.js's `EdgesGeometry` draws lines, which is a thing a rasteriser can do and a
 * distance field cannot: a segment has no volume, so there would be nothing to bake. The
 * rods are the honest translation - square in section, `thickness` across - and the
 * result is a cage you can union, cut with, or bake like any other mesh.
 */
export function edges(o: EdgesOptions): MeshData {
  const threshold = Math.cos(((o.thresholdAngle ?? 1) * Math.PI) / 180);
  return rods(sharpEdges(o.geometry, threshold), o.geometry, o.thickness);
}

export interface WireframeOptions {
  geometry: MeshData;
  /** Rod thickness. Defaults to 1% of the mesh's bounding-box diagonal. */
  thickness?: number;
}

/** Every triangle edge of a mesh, as solid rods. See {@link edges}. */
export function wireframe(o: WireframeOptions): MeshData {
  // A dot product never exceeds 1, so a threshold of 2 keeps every edge - flat ones too.
  return rods(sharpEdges(o.geometry, 2), o.geometry, o.thickness);
}

/** Edge segments whose two faces differ by more than `threshold`, plus every border edge. */
function sharpEdges(mesh: MeshData, threshold: number): [Vec3, Vec3][] {
  const pos = mesh.positions;
  const idx = mesh.indices;
  const count = idx ? idx.length : pos.length / 3;
  const corner = (t: number, k: number): number => (idx ? idx[t * 3 + k] : t * 3 + k);
  const point = (v: number): Vec3 => [pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]];

  // Vertices are welded by position first: three.js does the same, because a face that
  // was split for its UVs must not read as a crease.
  const ids = new Map<string, number>();
  const key = (v: number): number => {
    const p = point(v);
    const k = `${p[0].toFixed(6)},${p[1].toFixed(6)},${p[2].toFixed(6)}`;
    const found = ids.get(k);
    if (found !== undefined) {
      return found;
    }
    ids.set(k, v);
    return v;
  };

  const seen = new Map<string, { a: number; b: number; normal: Vec3 } | null>();
  const out: [Vec3, Vec3][] = [];
  for (let t = 0; t < Math.floor(count / 3); t++) {
    const v = [corner(t, 0), corner(t, 1), corner(t, 2)];
    const w = v.map(key);
    const normal = normalize(cross(sub(point(v[1]), point(v[0])), sub(point(v[2]), point(v[0]))));
    for (let e = 0; e < 3; e++) {
      const a = w[e];
      const bb = w[(e + 1) % 3];
      const k = a < bb ? `${a}_${bb}` : `${bb}_${a}`;
      const other = seen.get(k);
      if (other === undefined) {
        seen.set(k, { a: v[e], b: v[(e + 1) % 3], normal });
        continue;
      }
      if (other && dot(other.normal, normal) <= threshold) {
        out.push([point(other.a), point(other.b)]);
      }
      // Marked as handled, so a third face sharing the edge cannot emit it twice.
      seen.set(k, null);
    }
  }
  // An edge only one face touches is a border, and a border is always an edge.
  for (const entry of seen.values()) {
    if (entry) {
      out.push([point(entry.a), point(entry.b)]);
    }
  }
  return out;
}

/** A square prism around each segment. */
function rods(segs: readonly [Vec3, Vec3][], source: MeshData, thickness?: number): MeshData {
  const half = (thickness ?? 0.01 * diagonal(source)) / 2;
  const b = new Builder();
  for (const [p, q] of segs) {
    const dir = sub(q, p);
    const len = Math.hypot(dir[0], dir[1], dir[2]);
    if (len < 1e-12) {
      continue;
    }
    const t = scale(dir, 1 / len);
    // Any axis that is not the segment's own direction gives a usable first basis vector.
    const seed: Vec3 = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = normalize(cross(seed, t));
    const v = cross(t, u);
    const corners = (o: Vec3): number[] =>
      [0, 1, 2, 3].map((k) => {
        const a = (k * Math.PI) / 2 + Math.PI / 4;
        const s = half * Math.SQRT2;
        return b.vertex(
          o[0] + s * (Math.cos(a) * u[0] + Math.sin(a) * v[0]),
          o[1] + s * (Math.cos(a) * u[1] + Math.sin(a) * v[1]),
          o[2] + s * (Math.cos(a) * u[2] + Math.sin(a) * v[2]),
        );
      });
    const lo = corners(p);
    const hi = corners(q);
    for (let k = 0; k < 4; k++) {
      const n = (k + 1) % 4;
      b.quad(lo[k], lo[n], hi[n], hi[k]);
    }
    b.quad(hi[0], hi[1], hi[2], hi[3]);
    b.quad(lo[3], lo[2], lo[1], lo[0]);
  }
  if (b.indices.length === 0) {
    throw new Error('geometry.edges: the mesh has no edges past the threshold');
  }
  return b.data();
}

function diagonal(mesh: MeshData): number {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], mesh.positions[i + a]);
      hi[a] = Math.max(hi[a], mesh.positions[i + a]);
    }
  }
  const d = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return d > 0 ? d : 1;
}

// --- frames and vector arithmetic ----------------------------------------

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (a: Vec3): Vec3 => {
  const len = Math.hypot(a[0], a[1], a[2]);
  return len > 0 ? scale(a, 1 / len) : [0, 0, 1];
};
const lerp = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/** Rodrigues' rotation of `v` about the unit axis `axis` by `angle`. */
function rotateAbout(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(
    add(scale(v, c), scale(cross(axis, v), s)),
    scale(axis, dot(axis, v) * (1 - c)),
  );
}

/**
 * A frame per sample along a curve, carried forward rather than recomputed.
 *
 * The Frenet frame proper flips wherever a curve's curvature vanishes, which puts a
 * quarter turn in a tube for no reason at all. three.js's `computeFrenetFrames` - and
 * this - transports the first normal along instead, rotating it only by the turn between
 * consecutive tangents, and closes the loop by spreading the leftover twist evenly.
 */
function frenetFrames(
  path: (t: number) => Vec3,
  count: number,
  closed: boolean,
): { tangents: Vec3[]; normals: Vec3[]; binormals: Vec3[] } {
  const delta = 1e-4;
  const tangents: Vec3[] = [];
  // A closed spine wraps its difference rather than clamping it. Clamping leaves the first
  // and last tangents as one-sided differences taken from opposite sides of the same point,
  // so they disagree by about `delta` times the curvature - and the seam's two rings come out
  // rotated apart by that much, leaving a hairline gap no amount of closing rotation can
  // shut. Wrapping makes both of them the same central difference, and the seam closes.
  const wrap = (t: number): number => t - Math.floor(t);
  for (let i = 0; i <= count; i++) {
    const t = i / count;
    const t1 = closed ? wrap(t - delta) : Math.max(t - delta, 0);
    const t2 = closed ? wrap(t + delta) : Math.min(t + delta, 1);
    tangents.push(normalize(sub(path(t2), path(t1))));
  }

  // Start from whichever axis the first tangent leans on least, so the cross product is
  // well conditioned.
  const t0 = tangents[0];
  const abs: Vec3 = [Math.abs(t0[0]), Math.abs(t0[1]), Math.abs(t0[2])];
  const axis: Vec3 = abs[0] <= abs[1] && abs[0] <= abs[2]
    ? [1, 0, 0]
    : abs[1] <= abs[2] ? [0, 1, 0] : [0, 0, 1];
  const normals: Vec3[] = [normalize(cross(t0, normalize(cross(t0, axis))))];
  const binormals: Vec3[] = [cross(t0, normals[0])];

  for (let i = 1; i <= count; i++) {
    let n = normals[i - 1];
    const turn = cross(tangents[i - 1], tangents[i]);
    if (Math.hypot(turn[0], turn[1], turn[2]) > Number.EPSILON) {
      const angle = Math.acos(Math.min(1, Math.max(-1, dot(tangents[i - 1], tangents[i]))));
      n = rotateAbout(n, normalize(turn), angle);
    }
    normals.push(n);
    binormals.push(cross(tangents[i], n));
  }

  if (closed) {
    let theta = Math.acos(Math.min(1, Math.max(-1, dot(normals[0], normals[count])))) / count;
    if (dot(tangents[0], cross(normals[0], normals[count])) > 0) {
      theta = -theta;
    }
    for (let i = 1; i <= count; i++) {
      normals[i] = rotateAbout(normals[i], tangents[i], theta * i);
      binormals[i] = cross(tangents[i], normals[i]);
    }
  }
  return { tangents, normals, binormals };
}

/**
 * The catalogue, by three.js's names.
 *
 * Exported both ways: `geometry.box(...)` reads like three.js's `new THREE.BoxGeometry`,
 * and the individual functions are importable for a game that only wants one of them.
 */
export const geometry = {
  box,
  capsule,
  circle,
  cone,
  cylinder,
  dodecahedron,
  edges,
  extrude,
  icosahedron,
  lathe,
  octahedron,
  plane,
  polyhedron,
  ring,
  shape,
  sphere,
  tetrahedron,
  torus,
  torusKnot,
  tube,
  wireframe,
};

/** Signed area of a 2D outline, re-exported for a game that needs to check its winding. */
export { signedArea, triangulateShape };
