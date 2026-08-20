import { d } from '../gpu.ts';
import { DirLight } from '../trace/shade.ts';
import { GameObject } from './entity.ts';
import type { TgpuUniform } from 'typegpu';
import type { Game } from './game.ts';

export interface SunSpawnOptions {
  /** Direction the light travels, pointing away from the sun. Normalised for you. */
  direction?: readonly [number, number, number];
  color?: readonly [number, number, number];
  intensity?: number;
  /**
   * Angular radius, as a fraction. 0 is a point light and gives hard shadows; the real
   * sun is about 0.005 and Claybook's look wants an order of magnitude more than that.
   */
  softness?: number;
}

/**
 * The directional light. Spawned, mutable, and there is exactly one that matters - the
 * shading path is a single-light deferred resolve, so a second sun is ignored.
 */
export class Sun extends GameObject {
  /** Engine-level: what {@link DeferredResolve} binds. */
  readonly uniform: TgpuUniform<typeof DirLight>;

  private state: {
    dir: [number, number, number];
    size: number;
    color: [number, number, number];
    intensity: number;
  };

  constructor(game: Game, options: SunSpawnOptions = {}) {
    super(game);
    const dir = normalise(options.direction ?? [-0.45, -0.78, -0.43]);
    this.state = {
      dir,
      size: options.softness ?? 0.06,
      color: [...(options.color ?? [1, 0.95, 0.85])] as [number, number, number],
      intensity: options.intensity ?? 3.2,
    };
    this.uniform = game.root.createUniform(DirLight, this.state);
  }

  set direction(v: readonly [number, number, number]) {
    this.state.dir = normalise(v);
    this.uniform.writePartial({ dir: d.vec3f(...this.state.dir) });
  }
  get direction(): readonly [number, number, number] {
    return this.state.dir;
  }

  set color(v: readonly [number, number, number]) {
    this.state.color = [v[0], v[1], v[2]];
    this.uniform.writePartial({ color: d.vec3f(...this.state.color) });
  }
  get color(): readonly [number, number, number] {
    return this.state.color;
  }

  set intensity(v: number) {
    this.state.intensity = v;
    this.uniform.writePartial({ intensity: v });
  }
  get intensity(): number {
    return this.state.intensity;
  }

  set softness(v: number) {
    this.state.size = v;
    this.uniform.writePartial({ size: v });
  }
  get softness(): number {
    return this.state.size;
  }
}

function normalise(v: readonly [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
