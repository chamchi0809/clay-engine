import { sdf, type Shape } from '@clay/engine';

/** One page of the book. Claybook's levels were literal pages, so pages they are. */
export interface Page {
  name: string;
  hint: string;
  /** Where the clay ball is (re)spawned. */
  spawn: [number, number, number];
  /** Reach this to finish the page. */
  goal: [number, number, number];
  goalRadius: number;
  /** Where `F` pours water from. */
  tap: [number, number, number];
  shape: Shape;
}

/**
 * A floor with a rounded lip, so the ball can be lost off the side but not trivially.
 *
 * Two layers, and the split is the whole point: a stone core that nothing can dig
 * through, under a clay skin that keeps the imprint of everything that rolls over it.
 * Claybook's pages worked the same way - the surface you play on is the deformable one,
 * and the hard layer underneath is what stops the ball from carving its way out.
 */
const plate = (pos: [number, number, number], half: [number, number, number]) =>
  sdf.union(
    sdf.roundBox(half, 0.25).at(pos).material('stone').smooth(0.3),
    // Sunk half-way into the stone, so the groove bottoms out on rock instead of
    // punching through into open air.
    sdf.roundBox([half[0] - 0.12, 0.3, half[2] - 0.12], 0.2)
      .at([pos[0], pos[1] + half[1], pos[2]])
      .material('clay')
      .smooth(0.25),
  );

const pad = (pos: [number, number, number]) =>
  sdf.cylinder(1.6, 0.35).at(pos).material('gold').smooth(0.2);

export const pages: Page[] = [
  {
    name: 'Roll',
    hint: 'WASD rolls, Space hops. Reach the gold pad.',
    spawn: [-7, 2.4, 6],
    goal: [7.5, 0.6, -6.5],
    goalRadius: 1.6,
    tap: [0, 8, 0],
    shape: sdf.union(
      plate([0, -1.4, 0], [11, 1, 11]),
      // A ridge across the middle with one low saddle: rolling round it is faster than
      // over it, which is the whole tutorial.
      sdf.cut(
        sdf.roundBox([9.5, 1.4, 0.9], 0.3).at([0, 0.4, 0]).material('stone').smooth(0.4),
        sdf.sphere(2.0).at([2.5, 1.2, 0]).smooth(0.5),
      ),
      // Clay blobs: soft, and the same material the player is made of.
      sdf.sphere(1.7).at([-3, 0.2, -4]).material('clay').smooth(0.6),
      sdf.capsule(2.2, 0.8).rotate([0, 0, Math.PI / 2]).at([4, 0.3, 4.5]).material('clay').smooth(0.5),
      pad([7.5, 0.1, -6.5]),
    ),
  },
  {
    name: 'Squeeze',
    hint: 'Press 3 for the rod and roll it through the slot.',
    spawn: [-8, 2.4, 0],
    goal: [8, 0.8, 0],
    goalRadius: 1.6,
    tap: [-8, 9, 0],
    shape: sdf.union(
      plate([0, -1.4, 0], [11, 1, 5]),
      // A wall with a slot only a flattened / elongated shape fits through.
      sdf.cut(
        sdf.box([0.7, 2.6, 5]).at([0, 1.6, 0]).material('stone').smooth(0.2),
        sdf.roundBox([1.2, 0.45, 1.6], 0.2).at([0, 0.35, 0]).smooth(0.25),
      ),
      sdf.torus(2.0, 0.55).at([-4, 0.9, 0]).material('clay').smooth(0.4),
      pad([8, 0.1, 0]),
    ),
  },
  {
    name: 'Erode',
    hint: 'F opens the tap. Water carves the dam; so does Q.',
    spawn: [-7.5, 3.0, 0],
    goal: [7.5, 0.8, 0],
    goalRadius: 1.6,
    tap: [-4.5, 7.5, 0],
    shape: sdf.union(
      plate([-5, -1.4, 0], [6, 1, 5]),
      plate([6.5, -1.4, 0], [4.5, 1, 5]),
      // A soft clay dam across the gap. Thin enough that a minute of water opens it.
      sdf.roundBox([1.4, 1.8, 5], 0.3).at([1, 0.2, 0]).material('clay').smooth(0.5),
      sdf.roundBox([1.4, 0.8, 5], 0.3).at([1, -1.6, 0]).material('stone').smooth(0.4),
      pad([7.5, 0.1, 0]),
    ),
  },
];

/** The three shapes the ball can be. Claybook shipped four; three shows the mechanic. */
export const shapes = {
  ball: sdf.sphere(0.95),
  cube: sdf.roundBox([0.7, 0.7, 0.7], 0.25),
  rod: sdf.capsule(0.95, 0.48).rotate([0, 0, Math.PI / 2]),
} satisfies Record<string, Shape>;

export type ShapeName = keyof typeof shapes;
export const shapeNames = Object.keys(shapes) as ShapeName[];
