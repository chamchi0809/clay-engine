export interface OrbitOptions {
  /** Radians, clockwise from `+z` looking down. */
  yaw: number;
  /** Radians above the horizon. */
  pitch: number;
  /** Straight-line distance from the target point to the eye. */
  distance: number;
  /** Raises the eye above the target. */
  height?: number;
  /** Raises the look-at point above the target. */
  lookHeight?: number;
}

export interface View {
  position: [number, number, number];
  target: [number, number, number];
}

/**
 * Third-person orbit, as pure arithmetic. It touches nothing - hand the result to a
 * camera, or to a raycast, or throw it away.
 *
 * Kept out of the camera on purpose: where a camera *should* be is game design (lead
 * the player, dodge walls, snap on landing) and every game wants a different answer.
 * This is the one everybody starts with, not a policy the engine imposes.
 */
export function orbit(around: readonly [number, number, number], o: OrbitOptions): View {
  const horiz = Math.cos(o.pitch) * o.distance;
  const vert = Math.sin(o.pitch) * o.distance;
  const height = o.height ?? 0;
  const lookHeight = o.lookHeight ?? 0;
  return {
    position: [
      around[0] - Math.sin(o.yaw) * horiz,
      around[1] + vert + height,
      around[2] - Math.cos(o.yaw) * horiz,
    ],
    target: [around[0], around[1] + lookHeight, around[2]],
  };
}
