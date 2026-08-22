import { d, std, type CustomBrush, type MeshData } from '@clay/engine';

/**
 * A hexagonal prism, standing along Y, with rounded edges.
 *
 * This is the whole of what registering a primitive costs: a distance function and a
 * bound. Once `Game.create({ brushes: { hexPrism } })` has seen it, it is a brush like
 * `sdf.sphere` - it bakes into the world, it carves with `cut`, it respects `.only()`,
 * and a soft body can morph into it.
 *
 * `size.x` is the flat-to-flat radius, `size.y` the half-height. `radius` rounds it.
 */
export const hexPrism: CustomBrush = {
  sdf: (p: d.v3f, size: d.v3f, radius: number) => {
    'use gpu';
    // Quilez's hexagonal prism, with the cross-section moved onto XZ so the extrusion
    // runs along Y and the thing reads as a column. `k.xy` is the normal of the mirror
    // plane that folds a twelfth of the plane onto the rest; `k.z` is 1/sqrt(3).
    const k = d.vec3f(-0.8660254, 0.5, 0.57735);
    const q = std.abs(p);
    const flat = std.max(size.x - radius, 1e-4);
    const half = std.max(size.y - radius, 1e-4);
    const xz = q.xz - k.xy * (2 * std.min(std.dot(k.xy, q.xz), 0));
    // x: distance within the hexagon's plane, signed. y: distance along the axis.
    const dd = d.vec2f(
      std.length(xz - d.vec2f(std.clamp(xz.x, -k.z * flat, k.z * flat), flat))
        * std.sign(xz.y - flat),
      q.y - half,
    );
    // The standard extrusion combine, then the round. Both keep it 1-Lipschitz.
    return std.min(std.max(dd.x, dd.y), 0) + std.length(std.max(dd, d.vec2f())) - radius;
  },
  // A hexagon's corners sit at flat-to-flat over cos(30 degrees), so the corner column is
  // the farthest thing from the axis. `sdf` shrinks the core by `radius` before rounding
  // it back out, so `size` is already the outer extent and this is slack rather than tight.
  //
  // Slack is the right way to be wrong: over-reporting costs a few culled tiles being
  // visited for nothing, while under-reporting clips the shape along tile boundaries.
  bound: (size, radius) => Math.hypot(size[0] / 0.8660254, size[1]) + radius,
};

/**
 * A low-poly rock: an icosahedron subdivided once, with each vertex pushed to its own
 * radius. 80 triangles, no formula - which is the point. This is the shape a closed-form
 * primitive cannot express and a baker can.
 *
 * The jitter is a hash of the vertex index rather than `Math.random`, so the rock is the
 * same rock on every reload.
 */
export function rock(): MeshData {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts: [number, number, number][] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map(([x, y, z]) => {
    const len = Math.hypot(x, y, z);
    return [x / len, y / len, z / len] as [number, number, number];
  });
  let faces: [number, number, number][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];

  // One four-way split. Every child keeps its parent's winding, which is what lets the
  // bake sign the field: a mesh wound consistently has a solid inside, whichever way
  // round it is.
  const midpoints = new Map<string, number>();
  const midpoint = (a: number, b: number): number => {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    const found = midpoints.get(key);
    if (found !== undefined) {
      return found;
    }
    const [ax, ay, az] = verts[a];
    const [bx, by, bz] = verts[b];
    const len = Math.hypot(ax + bx, ay + by, az + bz);
    verts.push([(ax + bx) / len, (ay + by) / len, (az + bz) / len]);
    const index = verts.length - 1;
    midpoints.set(key, index);
    return index;
  };
  faces = faces.flatMap(([a, b, c]) => {
    const ab = midpoint(a, b);
    const bc = midpoint(b, c);
    const ca = midpoint(c, a);
    return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]] as [number, number, number][];
  });

  // Radial jitter, shared by every face that touches the vertex, so the surface stays
  // closed. A sphere would bake correctly and prove nothing.
  const positions = verts.flatMap(([x, y, z], i) => {
    const hash = Math.sin(i * 12.9898) * 43758.5453;
    const r = 0.8 + 0.28 * (hash - Math.floor(hash));
    return [x * r, y * r * 1.15, z * r];
  });

  return { positions, indices: faces.flat() };
}

/**
 * A tetrahedron as OBJ text, so the example exercises the other input `loadMesh` takes.
 * The loader is the point here, not the shape - `parseObj` fans quads, tolerates
 * `v/vt/vn` corners and negative indices, and ignores everything else.
 */
export const TETRA_OBJ = `# tetrahedron
v  0.000  0.850  0.000
v  0.943 -0.333  0.000
v -0.471 -0.333  0.816
v -0.471 -0.333 -0.816
vn 0 1 0

f 1//1 3//1 2//1
f 1//1 4//1 3//1
f 1//1 2//1 4//1
f 2//1 3//1 4//1
`;
