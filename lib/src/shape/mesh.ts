/**
 * Triangle geometry on its way to becoming a brush.
 *
 * Nothing here touches the GPU: it turns whatever a game has - a typed array from a glTF
 * loader, an OBJ file, three.js `BufferGeometry` attributes - into the one shape the baker
 * wants, which is a flat list of triangles in the bake box's own space.
 */

export interface MeshData {
  /** Vertex positions, xyz triples. */
  positions: ArrayLike<number>;
  /** Triangle indices. Omit for a soup of consecutive triples. */
  indices?: ArrayLike<number>;
}

export interface NormalizedMesh {
  /** Triangles in the bake box's `[-1, 1]^3` space, nine floats each. */
  triangles: Float32Array;
  triangleCount: number;
  /**
   * Half-extent of the bake box, in the *source* mesh's units. This is the number the
   * brush gets as its `size`, which is what makes a baked mesh come out at the size it
   * was modelled at.
   */
  half: number;
  /**
   * The margin `fit` leaves, as a fraction of the half-extent: every surface point is at
   * least this far inside every wall of the bake box.
   *
   * The brush needs it, not just the bake. Outside the box the brush has no field to read,
   * and the distance to the *box* is zero at the wall - which puts a zero isosurface on the
   * whole bake box and makes it render as a shell. `dBox + inset` is the fix, and it is
   * sound precisely because the box is convex: a segment from a point outside to any surface
   * point crosses the wall, so it is at least `dBox` long before the wall and `inset` after.
   */
  inset: number;
  /**
   * Half the surface thickness, as a fraction of the half-extent - so in the same units as
   * {@link NormalizedMesh.triangles}. Zero for an ordinary solid bake.
   */
  shell: number;
  /** Centre of the source mesh's bounding box, in its own units. */
  center: [number, number, number];
}

/**
 * A mesh that lives in an atlas slot. What `game.loadMesh` hands back and `sdf.mesh` turns
 * into a brush.
 *
 * Opaque on purpose: a slot index and a half-extent are all a brush needs, and the triangles
 * are gone by the time this exists - they were only ever input to the bake.
 */
export interface BakedMesh {
  readonly slot: number;
  /** The brush's `size`: the bake box's half-extent, in the source mesh's units. */
  readonly half: number;
  /** The brush's `radius`: {@link NormalizedMesh.inset}, which the fold needs outside the box. */
  readonly inset: number;
  /** Where the source mesh's bounding box was centred, for a game that has to line up with it. */
  readonly center: readonly [number, number, number];
  readonly triangleCount: number;
}

/**
 * Recentres and rescales a mesh into the cube a slot bakes.
 *
 * The pivot is the bounding box centre, not the origin the mesh was modelled about: a
 * brush rotates and scales about its own origin, and a shape that orbits some far-off
 * modelling origin when it spins is never what anyone wanted.
 *
 * `fit` leaves the shape short of the box wall. Some margin is needed - the field just
 * inside the wall has to be positive, or there is no exterior for the tracer to approach
 * the surface through - and 0.9 costs 10% of the resolution on the widest axis to get it.
 *
 * `thickness` gives the *surface* a volume: half of it either side. A plane, a ring, an
 * open lathe - anything a rasteriser is happy to draw and a distance field cannot represent,
 * because a surface of zero thickness has no inside - becomes a slab it can. The box grows
 * by half the thickness so the shell keeps the same margin the triangles would have had.
 */
export function normalizeMesh(mesh: MeshData, fit = 0.9, thickness = 0): NormalizedMesh {
  const pos = mesh.positions;
  const idx = mesh.indices;
  const count = idx ? idx.length : pos.length / 3;
  if (count < 3) {
    throw new Error('normalizeMesh: need at least one triangle');
  }
  const triangleCount = Math.floor(count / 3);

  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < pos.length; v += 3) {
    for (let a = 0; a < 3; a++) {
      const x = pos[v + a]!;
      lo[a] = Math.min(lo[a]!, x);
      hi[a] = Math.max(hi[a]!, x);
    }
  }
  if (!Number.isFinite(lo[0]!)) {
    throw new Error('normalizeMesh: positions contain no finite values');
  }
  const center: [number, number, number] = [
    (lo[0]! + hi[0]!) / 2, (lo[1]! + hi[1]!) / 2, (lo[2]! + hi[2]!) / 2,
  ];
  // The widest axis sets the scale, so the aspect ratio survives. A flat mesh - a leaf, a
  // sign - bakes into a slab of the cube and wastes the rest, which is the honest trade
  // for one cubic slot shape.
  const reach = Math.max(
    hi[0]! - center[0]!, hi[1]! - center[1]!, hi[2]! - center[2]!,
  ) + thickness / 2;
  if (!(reach > 0)) {
    throw new Error('normalizeMesh: the mesh has no extent - every vertex is the same point');
  }
  // A mesh that is flat on some axis encloses nothing, so the winding number is zero
  // everywhere and the slot bakes as empty space. That is a silently invisible object, and
  // the fix is one option, so say so here rather than let it ship.
  if (thickness <= 0 && [0, 1, 2].some((a) => hi[a]! - lo[a]! <= 0)) {
    throw new Error(
      'normalizeMesh: the mesh is flat, so it has no volume to bake. Pass `thickness` to '
        + 'give the surface one - `game.loadMesh(mesh, { thickness: 0.05 })`.',
    );
  }
  const half = reach / fit;

  const triangles = new Float32Array(triangleCount * 9);
  for (let t = 0; t < triangleCount; t++) {
    for (let corner = 0; corner < 3; corner++) {
      const v = (idx ? idx[t * 3 + corner]! : t * 3 + corner) * 3;
      for (let a = 0; a < 3; a++) {
        triangles[t * 9 + corner * 3 + a] = (pos[v + a]! - center[a]!) / half;
      }
    }
  }
  // `fit` sets the widest axis, so no vertex exceeds `fit` on any axis and the margin is
  // guaranteed on all six walls, not just the two the widest axis touches. A shell eats
  // into that margin by exactly `shell`, which is why `reach` paid for it up front.
  return { triangles, triangleCount, half, inset: 1 - fit, shell: thickness / 2 / half, center };
}

/**
 * The subset of Wavefront OBJ that describes a surface: `v` and `f`.
 *
 * Normals, UVs, materials and groups are all dropped, because a distance field has no use
 * for any of them - the bake reads positions and the shading comes from the field's own
 * gradient. Faces with more than three corners are fanned.
 */
export function parseObj(text: string): MeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (line.length === 0 || line.startsWith('#')) {
      continue;
    }
    const parts = line.split(/\s+/);
    if (parts[0] === 'v') {
      positions.push(+parts[1]!, +parts[2]!, +parts[3]!);
      continue;
    }
    if (parts[0] !== 'f') {
      continue;
    }
    // `f` corners are `v`, `v/vt`, `v/vt/vn` or `v//vn`; only the first field matters.
    // OBJ indices are 1-based, and a negative one counts back from the newest vertex.
    const corners = parts.slice(1).map((c) => {
      const i = parseInt(c.split('/')[0]!, 10);
      return i > 0 ? i - 1 : positions.length / 3 + i;
    });
    for (let k = 2; k < corners.length; k++) {
      indices.push(corners[0]!, corners[k - 1]!, corners[k]!);
    }
  }
  if (positions.length === 0 || indices.length === 0) {
    throw new Error('parseObj: no `v`/`f` lines found - is this an OBJ file?');
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}
