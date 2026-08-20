import * as sdf from '@typegpu/sdf';
import type { SdfSceneContext } from '@clay/engine/core';
import {
  d,
  std,
  SdfScene,
  analyticField,
  unionField,
  type MaterialValue,
} from '@clay/engine/core';

/**
 * The same engine, no clay in sight.
 *
 * This example exists to make the engine's generality checkable rather than claimed:
 * it never authors a brush, never allocates an SDF volume worth reading, and runs no
 * simulation. The scene is a handful of `@typegpu/sdf` primitives behind
 * `analyticField`, and it still gets the full render path - hierarchical sphere
 * tracing, the 8x8 cone-trace pre-pass, ray-traced soft shadows, cone AO, the spatial
 * filter and temporal accumulation - because all of that is written against
 * `TracedField` and nothing else.
 */
const canvas = document.getElementById('view') as HTMLCanvasElement;
const status = document.getElementById('status') as HTMLElement;
const errorBox = document.getElementById('error') as HTMLElement;

const materials: MaterialValue[] = [
  { albedo: [0.85, 0.86, 0.9], roughness: 0.6, emissive: [0, 0, 0], metallic: 0.0 },
  { albedo: [0.9, 0.35, 0.28], roughness: 0.3, emissive: [0, 0, 0], metallic: 0.1 },
  { albedo: [0.25, 0.62, 0.85], roughness: 0.15, emissive: [0, 0, 0], metallic: 0.6 },
];

async function main() {
  const scene = await SdfScene.create({
    canvas,
    materials,
    // A volume is still constructed - it is just never traced, so make it the smallest
    // legal one instead of paying for a 128^3 texture nothing reads.
    volume: { resolution: 8, worldSize: 1, origin: [0, 0, 0], band: 2 },
    camera: { fovY: Math.PI / 3, far: 80 },
    light: { dir: [-0.4, -0.82, -0.4], size: 0.05, color: [1, 0.96, 0.9], intensity: 3.4 },
    compose: ({ root }: SdfSceneContext) => {
      // One uniform is the whole animation rig: the field closure reads it, so moving
      // geometry costs a 16-byte upload and no rebuild.
      const time = root.createUniform(d.f32, 0);
      Object.assign(globalThis, { setTime: (t: number) => time.write(t) });

      const ground = analyticField((p: d.v3f) => {
        'use gpu';
        return sdf.sdPlane(p, d.vec3f(0, 1, 0), 0);
      }, { band: 12, epsilon: 0.002, material: 0 });

      const blobs = analyticField((p: d.v3f) => {
        'use gpu';
        const t = time.$;
        const spin = d.vec3f(std.cos(t) * 2.2, 1.6 + std.sin(t * 1.7) * 0.7, std.sin(t) * 2.2);
        const ball = sdf.sdSphere(p - spin, 1.0);
        const frame = sdf.sdBoxFrame3d(p - d.vec3f(0, 1.7, 0), d.vec3f(1.5), 0.12);
        return sdf.opSmoothUnion(frame, ball, 0.6);
      }, { band: 12, epsilon: 0.002, material: 1 });

      const arm = analyticField((p: d.v3f) => {
        'use gpu';
        return sdf.sdCapsule(p, d.vec3f(-2.4, 0.5, 0), d.vec3f(2.4, 0.5, 0), 0.35);
      }, { band: 12, epsilon: 0.002, material: 2 });

      // Three analytic fields, one per material, unioned by the same combinator the
      // Claybook demo uses to merge its fluid bake into the world.
      return unionField(ground, unionField(blobs, arm));
    },
    shading: { aoDistance: 2.5 },
  });

  const setTime = (globalThis as { setTime?: (t: number) => void }).setTime!;
  let frames = 0;
  let last = 0;
  const start = performance.now();

  const loop = (now: number) => {
    const t = (now - start) / 1000;
    setTime(t);
    // Slow orbit, so the temporal filter has to reproject every pixel every frame.
    const r = 7.5;
    scene.camera.pos = [Math.sin(t * 0.25) * r, 3.4 + Math.sin(t * 0.4) * 0.6, Math.cos(t * 0.25) * r];
    scene.camera.target = [0, 1.4, 0];
    scene.render();
    frames++;
    if (now - last > 400) {
      status.textContent = `${canvas.width}x${canvas.height} · ${Math.round((frames * 1000) / (now - last))} fps · analytic field, no volume`;
      frames = 0;
      last = now;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err: unknown) => {
  errorBox.style.display = 'grid';
  errorBox.textContent = String(err instanceof Error ? (err.stack ?? err.message) : err);
});
