import { d } from 'typegpu';
import type { TgpuRoot, TgpuUniform } from 'typegpu';
import { mat4, vec3 } from 'wgpu-matrix';

/**
 * Everything a ray-marching shader needs from the camera.
 *
 * A raymarcher wants a basis and an aperture, not a projection matrix, so the basis
 * is stored explicitly. `viewProj` is still here for two things a raymarcher cannot
 * do without: writing `fragDepth` so rasterised geometry can depth-test against the
 * traced world, and reprojecting last frame's pixel for temporal accumulation.
 */
export const CameraUniform = d.struct({
  pos: d.vec3f,
  /** `tan(fovY / 2)`. */
  tanHalfFov: d.f32,
  right: d.vec3f,
  aspect: d.f32,
  up: d.vec3f,
  near: d.f32,
  fwd: d.vec3f,
  far: d.f32,
  viewProj: d.mat4x4f,
  prevViewProj: d.mat4x4f,
  /** Last frame's eye position - the temporal filter needs it to validate a reprojection. */
  prevPos: d.vec3f,
  _padPrev: d.f32,
  resolution: d.vec2f,
  invResolution: d.vec2f,
  /** Frame counter - seeds the per-pixel jitter of AO, shadows and TAA. */
  frame: d.u32,
  _pad: d.vec3u,
});

export interface CameraOptions {
  fovY?: number;
  near?: number;
  far?: number;
}

export class Camera {
  pos: [number, number, number] = [0, 4, 8];
  target: [number, number, number] = [0, 0, 0];
  worldUp: [number, number, number] = [0, 1, 0];
  fovY: number;
  near: number;
  far: number;
  width = 1;
  height = 1;
  frame = 0;

  readonly uniform: TgpuUniform<typeof CameraUniform>;

  /** Basis vectors, refreshed by {@link update}. Handy for CPU-side input mapping. */
  readonly right = new Float32Array(3);
  readonly up = new Float32Array(3);
  readonly fwd = new Float32Array(3);

  private readonly view = new Float32Array(16);
  private readonly proj = new Float32Array(16);
  private readonly viewProj = new Float32Array(16);
  private readonly prevViewProj = new Float32Array(16);
  private readonly prevPos: [number, number, number] = [0, 4, 8];

  constructor(root: TgpuRoot, options: CameraOptions = {}) {
    this.fovY = options.fovY ?? Math.PI / 3;
    this.near = options.near ?? 0.05;
    this.far = options.far ?? 400;
    this.uniform = root.createUniform(CameraUniform);
    mat4.identity(this.prevViewProj);
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width | 0);
    this.height = Math.max(1, height | 0);
  }

  /** Recompute the basis and matrices, then upload. Call once per frame. */
  update(): void {
    this.prevViewProj.set(this.viewProj);
    mat4.lookAt(this.pos, this.target, this.worldUp, this.view);
    mat4.perspective(this.fovY, this.width / this.height, this.near, this.far, this.proj);
    mat4.multiply(this.proj, this.view, this.viewProj);

    // The view matrix rows are the camera basis; forward is -z in view space.
    vec3.subtract(this.target, this.pos, this.fwd);
    vec3.normalize(this.fwd, this.fwd);
    vec3.cross(this.fwd, this.worldUp, this.right);
    vec3.normalize(this.right, this.right);
    vec3.cross(this.right, this.fwd, this.up);

    this.uniform.write({
      pos: this.pos,
      tanHalfFov: Math.tan(this.fovY / 2),
      right: [this.right[0], this.right[1], this.right[2]],
      aspect: this.width / this.height,
      up: [this.up[0], this.up[1], this.up[2]],
      near: this.near,
      fwd: [this.fwd[0], this.fwd[1], this.fwd[2]],
      far: this.far,
      viewProj: this.viewProj,
      prevViewProj: this.prevViewProj,
      prevPos: [...this.prevPos],
      _padPrev: 0,
      resolution: [this.width, this.height],
      invResolution: [1 / this.width, 1 / this.height],
      frame: this.frame,
      _pad: [0, 0, 0],
    });
    this.prevPos[0] = this.pos[0];
    this.prevPos[1] = this.pos[1];
    this.prevPos[2] = this.pos[2];
    this.frame = (this.frame + 1) >>> 0;
  }

  /**
   * CPU-side ray through a pixel, in the same convention as the shader below.
   * `x`/`y` in pixels, y down. Used for mouse picking - the demo carves the world
   * wherever the player clicks.
   */
  screenRay(x: number, y: number): { origin: [number, number, number]; dir: [number, number, number] } {
    const ndcX = (x / this.width) * 2 - 1;
    const ndcY = 1 - (y / this.height) * 2;
    const t = Math.tan(this.fovY / 2);
    const sx = ndcX * t * (this.width / this.height);
    const sy = ndcY * t;
    const dir: [number, number, number] = [
      this.fwd[0] + this.right[0] * sx + this.up[0] * sy,
      this.fwd[1] + this.right[1] * sx + this.up[1] * sy,
      this.fwd[2] + this.right[2] * sx + this.up[2] * sy,
    ];
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    return {
      origin: [...this.pos],
      dir: [dir[0] / len, dir[1] / len, dir[2] / len],
    };
  }
}

/** Matching shader-side ray: `uv` in `[0,1]`, y down (as produced by a fullscreen triangle). */
export function makeCameraRay(camera: TgpuUniform<typeof CameraUniform>) {
  return (uv: d.v2f) => {
    'use gpu';
    const c = camera.$;
    const ndc = d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2);
    const dir = c.fwd
      + c.right * (ndc.x * c.tanHalfFov * c.aspect)
      + c.up * (ndc.y * c.tanHalfFov);
    return dir;
  };
}
