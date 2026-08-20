/**
 * clay-engine - the game API.
 *
 * Everything here is written so a game never sees a GPU. Spawn objects into a `Game`,
 * describe shapes with `sdf`, and let the frame loop do the rest:
 *
 * ```ts
 * const game = await Game.create({ canvas, materials: { clay: {...}, stone: {...} } });
 * game.spawn.sun();
 * const camera = game.spawn.camera();
 * game.spawn.solid({ shape: sdf.box([12, 1, 12]).at([0, -1, 0]).material('stone') });
 * const ball = game.spawn.softBody({ shape: sdf.sphere(1).material('clay') });
 * game.start(() => {
 *   ball.addForce([0, 0, -30]);
 *   Object.assign(camera, orbit(ball.smoothPosition, { yaw, pitch, distance: 8 }));
 * });
 * ```
 *
 * The engine underneath is available as `@clay/engine/core` and is what a new *kind* of
 * object is written against - see `Entity`. The split is the point: extending the engine
 * is GPU work and looks like it; using it is not and does not.
 */

export * from './shape/sdf.ts';
export * from './game/index.ts';
export type { MaterialValue, DirLightValue } from './trace/shade.ts';
/** For `Game.create({ brushes })`. The one place the game API asks for GPU code. */
export type { CustomBrush } from './field/brush.ts';
export type { TracedField } from './trace/field.ts';
export type { ShadingOptions } from './trace/shade.ts';
export type { TracerOptions } from './trace/march.ts';
