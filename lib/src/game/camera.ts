import { Camera } from '../render/camera.ts';
import { GameObject } from './entity.ts';
import type { EntityContext } from './entity.ts';
import type { Game } from './game.ts';

export interface CameraSpawnOptions {
  /** Vertical field of view, radians. */
  fov?: number;
  near?: number;
  far?: number;
  position?: readonly [number, number, number];
  target?: readonly [number, number, number];
}

/**
 * The eye. A spawned object like everything else, and it does not decide where it should
 * be - the game writes {@link position} and {@link target} every frame.
 *
 * `orbit()` is a separate pure function rather than a method here, because a third-person
 * orbit is one camera design out of many and the engine has no business picking it.
 */
export class GameCamera extends GameObject {
  /** The low-level camera. Engine-level code needs its uniform; games do not. */
  readonly camera: Camera;

  constructor(game: Game, options: CameraSpawnOptions = {}) {
    super(game);
    this.camera = new Camera(game.root, {
      fovY: options.fov ?? Math.PI / 3,
      near: options.near,
      far: options.far,
    });
    if (options.position) {
      this.position = options.position;
    }
    if (options.target) {
      this.target = options.target;
    }
  }

  get position(): readonly [number, number, number] {
    return this.camera.pos;
  }
  set position(v: readonly [number, number, number]) {
    this.camera.pos = [v[0], v[1], v[2]];
  }

  get target(): readonly [number, number, number] {
    return this.camera.target;
  }
  set target(v: readonly [number, number, number]) {
    this.camera.target = [v[0], v[1], v[2]];
  }

  /** Look direction, unit length. Refreshed after each rendered frame. */
  get forward(): [number, number, number] {
    const f = this.camera.fwd;
    return [f[0]!, f[1]!, f[2]!];
  }

  /** Right-hand basis vector, unit length. Handy for camera-relative movement. */
  get right(): [number, number, number] {
    const r = this.camera.right;
    return [r[0]!, r[1]!, r[2]!];
  }

  /**
   * World-space ray through a pixel. Use it for picking: hand the ray to whatever you
   * want to hit-test against.
   */
  rayAt(x: number, y: number): { origin: [number, number, number]; dir: [number, number, number] } {
    return this.camera.screenRay(x, y);
  }

  build(_ctx: EntityContext): void {}
}
