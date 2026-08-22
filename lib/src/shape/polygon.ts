/**
 * Outlines in the plane, cut into triangles.
 *
 * Two of three.js's geometries - `ShapeGeometry` and `ExtrudeGeometry` - start from a
 * closed outline with holes in it rather than from a formula, and both need it
 * triangulated before anything can be built on top: the flat one *is* the triangulation,
 * and the extruded one caps its two ends with it. three.js runs ear clipping there
 * (`ShapeUtils.triangulateShape`, which is earcut); so does this.
 *
 * Nothing here knows about the GPU or about distance fields. It is plain arithmetic over
 * plain arrays, which is what makes it testable without a driver.
 */

export type Vec2 = readonly [number, number];

/**
 * A closed outline, optionally with holes punched in it. three.js's `Shape` with its
 * curves already sampled - `shape.extractPoints(curveSegments)` hands back exactly this
 * pair, so a three.js `Shape` converts in one line.
 *
 * Winding is not asked for and not trusted: the triangulator orients the outer contour
 * and the holes itself, because half the outlines in the wild come out of an SVG or a
 * font and are wound whichever way the exporter felt like.
 */
export interface Shape2D {
  /** The outer outline. A repeated closing point is fine; it is dropped. */
  contour: readonly Vec2[];
  /** Holes, each its own closed outline. */
  holes?: readonly (readonly Vec2[])[];
}

/** Signed area of a ring, by the trapezoid rule. Positive is counter-clockwise. */
export function signedArea(points: readonly Vec2[]): number {
  let sum = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    sum += (points[j][0] - points[i][0]) * (points[i][1] + points[j][1]);
  }
  return sum / 2;
}

/**
 * A flattened outline: the contour's points followed by each hole's, in order.
 *
 * Extrusion needs this list rather than the triangles alone - the side walls run between
 * consecutive points of each ring - so the split between "which points" and "which
 * triangles over them" is part of the return value rather than an implementation detail.
 */
export interface Triangulation {
  /** Contour points, then each hole's, with any repeated closing point removed. */
  points: Vec2[];
  /** Where each ring starts in {@link points}. `rings[0]` is always 0. */
  rings: number[];
  /** Triangle corners, three indices into {@link points} each. */
  indices: number[];
}

/** Drops a closing point that repeats the first one, which fonts and SVG both emit. */
function openRing(ring: readonly Vec2[]): Vec2[] {
  const points = ring.slice();
  while (points.length > 1) {
    const a = points[0];
    const b = points[points.length - 1];
    if (Math.abs(a[0] - b[0]) > 1e-12 || Math.abs(a[1] - b[1]) > 1e-12) {
      break;
    }
    points.pop();
  }
  return points;
}

/**
 * Ear clipping with hole bridges (Eberly; the implementation earcut.js made standard).
 *
 * The z-order acceleration earcut uses is left out: it pays off in the tens of thousands
 * of points, and an outline that arrives here is a letterform or a level's floor plan.
 * What is kept is the part that matters for correctness - holes are joined to the contour
 * by a bridge edge, so the whole thing becomes one simply-connected ring before a single
 * ear is clipped.
 */
export function triangulateShape(shape: Shape2D): Triangulation {
  const contour = openRing(shape.contour);
  if (contour.length < 3) {
    throw new Error('triangulateShape: the contour needs at least three points');
  }
  const points: Vec2[] = [...contour];
  const rings = [0];
  const holeStarts: number[] = [];
  for (const hole of shape.holes ?? []) {
    const ring = openRing(hole);
    if (ring.length < 3) {
      continue;
    }
    rings.push(points.length);
    holeStarts.push(points.length);
    points.push(...ring);
  }

  const indices: number[] = [];
  // The outer ring is linked counter-clockwise and the holes clockwise, which is what
  // makes a bridge fold a hole into the contour rather than tie it in a knot.
  let outer = linkRing(points, 0, holeStarts[0] ?? points.length, true);
  if (outer) {
    outer = eliminateHoles(points, holeStarts, outer);
    earcut(outer, indices);
  }
  return { points, rings, indices };
}

/** Area of a triangulation's triangles. A cheap way to assert a triangulation is sane. */
export function triangleArea(tri: Triangulation): number {
  let sum = 0;
  for (let i = 0; i < tri.indices.length; i += 3) {
    const a = tri.points[tri.indices[i]];
    const b = tri.points[tri.indices[i + 1]];
    const c = tri.points[tri.indices[i + 2]];
    sum += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
  }
  return sum;
}

/**
 * Offsets a ring along each vertex's angle bisector. Positive `distance` moves *into* the
 * area the ring encloses, whichever way round it is wound.
 *
 * This is three.js's `getBevelVec`: the corner lands where it is `distance` from *both*
 * edges that meet there, so a bevelled outline keeps its corners sharp instead of
 * rounding them off. A reflex corner comes out pushed the other way, which is the same
 * formula with the sign it already has.
 *
 * `ccw` is the ring's winding, which the caller already knows - the triangulator decided
 * it - and which is what says which side of an edge the interior is on.
 */
export function offsetRing(ring: readonly Vec2[], distance: number, ccw: boolean): Vec2[] {
  const n = ring.length;
  const out: Vec2[] = [];
  const sign = ccw ? 1 : -1;
  for (let i = 0; i < n; i++) {
    const prev = ring[(i + n - 1) % n];
    const curr = ring[i];
    const next = ring[(i + 1) % n];
    // Inward normals of the two edges, as three.js writes them: rotate the edge by 90
    // degrees and normalise.
    const ax = curr[0] - prev[0];
    const ay = curr[1] - prev[1];
    const bx = next[0] - curr[0];
    const by = next[1] - curr[1];
    const la = Math.hypot(ax, ay) || 1;
    const lb = Math.hypot(bx, by) || 1;
    const nax = -ay / la;
    const nay = ax / la;
    const nbx = -by / lb;
    const nby = bx / lb;
    // Intersection of the two offset edges. `det` is the sine of the turn, so it vanishes
    // exactly when the edges are parallel - and then the bisector *is* the shared normal.
    const det = nax * nby - nay * nbx;
    if (Math.abs(det) < 1e-9) {
      out.push([curr[0] + sign * distance * nax, curr[1] + sign * distance * nay]);
      continue;
    }
    // Solve for the offset corner: the vector `v` with `v . na == v . nb == 1`, so that
    // scaling it by `distance` lands `distance` from *both* edges. Writing it as
    // `v = na + s * perp(na)` satisfies the first equation for free and leaves one
    // unknown, which the second equation pins down.
    const s = (1 - (nax * nbx + nay * nby)) / det;
    const vx = nax + s * (-nay);
    const vy = nay + s * nax;
    // A hairpin turn sends the bisector to infinity; clamp it to something a mesh can
    // survive rather than emitting a vertex at 1e9.
    const len = Math.hypot(vx, vy);
    const scale = Math.min(len, 8) / (len || 1);
    out.push([curr[0] + sign * distance * vx * scale, curr[1] + sign * distance * vy * scale]);
  }
  return out;
}

// --- ear clipping ---------------------------------------------------------
// A doubly linked ring of vertices, each remembering the index it came from, so the
// triangles that come out index the caller's point list rather than a copy of it.

interface Node {
  i: number;
  x: number;
  y: number;
  prev: Node;
  next: Node;
  /** A bridge's second copy of a vertex. Never clipped as an ear on its own. */
  steiner: boolean;
}

function insertNode(i: number, x: number, y: number, last: Node | null): Node {
  const p: Node = { i, x, y, prev: null as unknown as Node, next: null as unknown as Node, steiner: false };
  if (!last) {
    p.prev = p;
    p.next = p;
  } else {
    p.next = last.next;
    p.prev = last;
    last.next.prev = p;
    last.next = p;
  }
  return p;
}

function removeNode(p: Node): void {
  p.next.prev = p.prev;
  p.prev.next = p.next;
}

const same = (a: Node, b: Node): boolean => a.x === b.x && a.y === b.y;

/** Links `points[start, end)` into a ring wound the way `ccw` asks for. */
function linkRing(points: readonly Vec2[], start: number, end: number, ccw: boolean): Node | null {
  let area = 0;
  for (let i = start, j = end - 1; i < end; j = i++) {
    area += (points[j][0] - points[i][0]) * (points[i][1] + points[j][1]);
  }
  let last: Node | null = null;
  if (ccw === area > 0) {
    for (let i = start; i < end; i++) {
      last = insertNode(i, points[i][0], points[i][1], last);
    }
  } else {
    for (let i = end - 1; i >= start; i--) {
      last = insertNode(i, points[i][0], points[i][1], last);
    }
  }
  if (last && same(last, last.next)) {
    removeNode(last);
    last = last.next;
  }
  return last;
}

/** Twice the signed area of the triangle `p q r`. Negative is a convex (CCW) corner. */
const cross = (p: Node, q: Node, r: Node): number =>
  (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

function inTriangle(
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
  px: number, py: number,
): boolean {
  return (cx - px) * (ay - py) >= (ax - px) * (cy - py)
    && (ax - px) * (by - py) >= (bx - px) * (ay - py)
    && (bx - px) * (cy - py) >= (cx - px) * (by - py);
}

/** Drops duplicate and collinear vertices, which are what stall the clip. */
function filterPoints(start: Node | null, end: Node | null = start): Node | null {
  if (!start) {
    return null;
  }
  let p = start;
  let tail = end ?? start;
  for (;;) {
    if (!p.steiner && (same(p, p.next) || cross(p.prev, p, p.next) === 0)) {
      removeNode(p);
      p = tail = p.prev;
      if (p === p.next) {
        return null;
      }
      continue;
    }
    p = p.next;
    if (p === tail) {
      return p;
    }
  }
}

/** Whether the corner at `ear` is convex and holds no other vertex of the ring. */
function isEar(ear: Node): boolean {
  const a = ear.prev;
  const b = ear;
  const c = ear.next;
  if (cross(a, b, c) >= 0) {
    // Reflex, or degenerate.
    return false;
  }
  const minX = Math.min(a.x, b.x, c.x);
  const maxX = Math.max(a.x, b.x, c.x);
  const minY = Math.min(a.y, b.y, c.y);
  const maxY = Math.max(a.y, b.y, c.y);
  let p = c.next;
  while (p !== a) {
    if (
      p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY
      && inTriangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y)
      && cross(p.prev, p, p.next) >= 0
    ) {
      return false;
    }
    p = p.next;
  }
  return true;
}

/**
 * Clips ears until the ring is a triangle.
 *
 * `pass` is the escape from an outline no ear will bite: pass 0 gives up and asks for a
 * filtered ring, pass 1 clips the corner it is on regardless. That last step can emit
 * overlapping triangles for a self-intersecting outline - but the baker signs its field
 * with a generalised winding number, which sums overlaps rather than being confused by
 * them, so a best-effort fan bakes into the shape a human would have drawn. Looping
 * forever, the alternative, does not.
 */
function earcut(ear: Node | null, out: number[], pass = 0): void {
  if (!ear) {
    return;
  }
  let node = ear;
  let stop = ear;
  while (node.prev !== node.next) {
    const prev = node.prev;
    const next = node.next;
    if (isEar(node) || (pass > 0 && node === stop)) {
      out.push(prev.i, node.i, next.i);
      removeNode(node);
      node = next.next;
      stop = next.next;
      continue;
    }
    node = next;
    if (node === stop) {
      if (pass === 0) {
        const filtered = filterPoints(node);
        if (filtered) {
          earcut(filtered, out, 1);
        }
      }
      return;
    }
  }
}

/** Folds every hole into the outer ring with a bridge edge. */
function eliminateHoles(points: readonly Vec2[], starts: readonly number[], outerNode: Node): Node {
  const queue: Node[] = [];
  for (let h = 0; h < starts.length; h++) {
    const start = starts[h];
    const end = h + 1 < starts.length ? starts[h + 1] : points.length;
    const ring = linkRing(points, start, end, false);
    if (!ring) {
      continue;
    }
    if (ring === ring.next) {
      ring.steiner = true;
    }
    queue.push(leftmost(ring));
  }
  // Left to right, so a bridge is always drawn to a part of the polygon that has already
  // been merged - never across a hole that is still floating.
  queue.sort((a, b) => a.x - b.x || a.y - b.y);
  let outer = outerNode;
  for (const hole of queue) {
    outer = eliminateHole(hole, outer);
  }
  return outer;
}

function leftmost(start: Node): Node {
  let p = start;
  let best = start;
  do {
    if (p.x < best.x || (p.x === best.x && p.y < best.y)) {
      best = p;
    }
    p = p.next;
  } while (p !== start);
  return best;
}

function eliminateHole(hole: Node, outerNode: Node): Node {
  const bridge = findHoleBridge(hole, outerNode);
  if (!bridge) {
    return outerNode;
  }
  const reverse = splitPolygon(bridge, hole);
  filterPoints(reverse, reverse.next);
  return filterPoints(bridge, bridge.next) ?? bridge;
}

/**
 * Whether the segment `a`-`b` leaves `a` on the inside of the ring. A bridge that fails
 * this leaves the polygon at its first step, however clear the rest of its path is.
 */
function locallyInside(a: Node, b: Node): boolean {
  return cross(a.prev, a, a.next) < 0
    ? cross(a, b, a.next) >= 0 && cross(a, a.prev, b) >= 0
    : cross(a, b, a.prev) < 0 || cross(a, a.next, b) < 0;
}

/**
 * The vertex of the outer ring a hole's leftmost point can see.
 *
 * Cast a ray left from the hole, take the edge it hits, and start from that edge's
 * rightmost end - then walk the ring keeping any vertex that lies inside the triangle
 * between the hole point, the hit and the candidate, because such a vertex would block
 * the bridge. earcut's rule, and the reason bridges never cross the outline.
 */
function findHoleBridge(hole: Node, outerNode: Node): Node | null {
  let p = outerNode;
  let qx = -Infinity;
  let m: Node | null = null;
  do {
    // Only edges the horizontal ray actually spans.
    if (hole.y <= p.y && hole.y >= p.next.y && p.next.y !== p.y) {
      const x = p.x + ((hole.y - p.y) * (p.next.x - p.x)) / (p.next.y - p.y);
      if (x <= hole.x && x > qx) {
        qx = x;
        m = p.x < p.next.x ? p : p.next;
        if (x === hole.x) {
          // The hole touches this edge: no better bridge exists.
          return m;
        }
      }
    }
    p = p.next;
  } while (p !== outerNode);

  if (!m) {
    return null;
  }
  const stop = m;
  const mx = m.x;
  const my = m.y;
  let tanMin = Infinity;
  p = m;
  do {
    const inside = hole.x >= p.x && p.x >= mx && hole.x !== p.x
      && inTriangle(
        hole.y < my ? hole.x : qx, hole.y,
        mx, my,
        hole.y < my ? qx : hole.x, hole.y,
        p.x, p.y,
      );
    if (inside) {
      // Nearest by angle wins; a tie is broken to the right, and a tie there to whichever
      // vertex's own wedge the other sits in.
      const tan = Math.abs(hole.y - p.y) / (hole.x - p.x);
      if (
        locallyInside(p, hole)
        && (tan < tanMin
          || (tan === tanMin
            && (p.x > m.x || (p.x === m.x && sectorContainsSector(m, p)))))
      ) {
        m = p;
        tanMin = tan;
      }
    }
    p = p.next;
  } while (p !== stop);
  return m;
}

/** Whether `m`'s two edges lie inside `p`'s, which breaks a tie between equal tangents. */
function sectorContainsSector(m: Node, p: Node): boolean {
  return cross(m.prev, m, p.prev) < 0 && cross(p.next, m, m.next) < 0;
}

/**
 * Cuts the ring along `a`-`b` and stitches the two sides into one. Each endpoint gains a
 * copy of itself, which is why the copies are marked as bridges and never clipped alone.
 */
function splitPolygon(a: Node, b: Node): Node {
  const a2: Node = { i: a.i, x: a.x, y: a.y, prev: null as unknown as Node, next: null as unknown as Node, steiner: false };
  const b2: Node = { i: b.i, x: b.x, y: b.y, prev: null as unknown as Node, next: null as unknown as Node, steiner: false };
  const an = a.next;
  const bp = b.prev;

  a.next = b;
  b.prev = a;
  a2.next = an;
  an.prev = a2;
  b2.next = a2;
  a2.prev = b2;
  bp.next = b2;
  b2.prev = bp;
  return b2;
}
