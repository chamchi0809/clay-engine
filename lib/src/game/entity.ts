import type { TgpuComputePass, TgpuRenderCommands, TgpuRoot } from 'typegpu';
import type { Camera } from '../render/camera.ts';
import type { MaterialPalette } from '../trace/shade.ts';
import type { TracedField } from '../trace/field.ts';
import type { Game } from './game.ts';

/**
 * How a force is interpreted. Matches the mental model every physics engine uses,
 * because a game programmer already has it.
 */
export type ForceMode =
  /** World units per second squared, applied for one frame. */
  | 'acceleration'
  /** World units per second, added to the velocity at once. */
  | 'velocity';

/**
 * What an entity gets when the game builds its pipelines.
 *
 * This is where the engine's internals *are* exposed - and deliberately so. Adding a new
 * kind of thing to a game means writing GPU work, so a new entity type is engine-level
 * code and gets engine-level access. What matters is that a game which only *uses*
 * entities never sees any of it.
 */
export interface EntityContext {
  root: TgpuRoot;
  /**
   * Everything solid enough to hit, as one field, optionally minus one entity - pass
   * `this` so a simulated body does not collide with its own bake and lock solid.
   */
  colliders(exclude?: Entity): TracedField;
  /** Everything traced this frame, solid and simulated. */
  scene: TracedField;
  camera: Camera;
  palette: MaterialPalette;
  paletteCount: number;
  /** Play-area bounds, from {@link GameOptions.bounds}. */
  bounds: { origin: readonly [number, number, number]; size: number };
  /** Resolves a material name to its palette index. */
  material(name: string | number | undefined): number;
}

/**
 * Anything that exists in a game.
 *
 * The set of entity types is open: `spawn.*` is sugar for `new Solid(game, opts)` and a
 * third party writes `new MyThing(game, opts)` with no registration step. All an entity
 * has to do is call `game.attach(this)` from its constructor.
 *
 * The two-phase construction is not ceremony. A render pipeline bakes the field it
 * traces into shader code, so every field that will be traced must exist before any
 * pipeline does. Therefore: allocate buffers, volumes and fields in the constructor;
 * create pipelines in {@link build}, which the game calls on the first frame after the
 * entity set last changed.
 */
export interface Entity {
  /** The game it belongs to. {@link GameObject} sets this for you. */
  readonly game: Game;
  /**
   * This entity as a signed distance field. Two independent roles, hence two flags:
   * {@link traced} decides whether the renderer draws it, {@link collidable} whether
   * anything else can hit it. A rasterised body is collidable but not traced; a fluid is
   * traced but not collidable *by itself*.
   */
  readonly field?: TracedField | null;
  /**
   * Whether the renderer traces {@link field}. Default true. False for anything that
   * draws itself through {@link drawGeometry} instead - a mesh, a sprite, a decal.
   *
   * Not free to leave on: every traced field costs a bind group and a texture sample on
   * every primary, shadow and AO ray, and WebGPU allows four bind groups in total.
   */
  readonly traced?: boolean;
  /**
   * Whether other things should collide with {@link field}. Default true. A fluid sets
   * it false: its own bake is not something to bounce off, it is the fluid itself.
   */
  readonly collidable?: boolean;
  /**
   * Create pipelines.
   *
   * Called before the first frame, and again after the entity set changes - because a
   * pipeline is compiled against one specific field and spawning something traceable
   * invalidates it. An entity holding state it cannot cheaply recreate (particles in
   * flight, allocated buffers) should build once and ignore later calls.
   */
  build?(ctx: EntityContext): void;
  /** GPU compute work, recorded before the frame's render passes. */
  simulate?(pass: TgpuComputePass): void;
  /** Rasterised geometry sharing the G-buffer. Depth-tests against the traced scene. */
  drawGeometry?(pass: TgpuRenderCommands): void;
  /** CPU-side work: readbacks, filters, timers. `dt` is the real frame time. */
  sync?(dt: number): void;
  /** Release GPU resources. Called by `game.despawn`. */
  destroy?(): void;
}

/** Base class that wires an entity into its game. Implementing {@link Entity} is enough. */
export abstract class GameObject implements Entity {
  readonly game: Game;

  constructor(game: Game) {
    this.game = game;
    game.attach(this);
  }

  despawn(): void {
    this.game.despawn(this);
  }
}
