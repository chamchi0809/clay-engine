import { create, globals } from 'webgpu';

/**
 * Node has no `navigator.gpu`. Dawn's binding provides one, plus the `GPUBufferUsage`-style
 * enum globals that a browser has ambiently and TypeGPU reads directly.
 *
 * This is what makes the GPU half of the engine testable at all: the brush fold, the mesh
 * bake and the tile cull are compute shaders whose bugs are silent - a slightly wrong sign,
 * a tile culled one voxel early - and a JS mirror of a shader is not the shader. Everything
 * under `test/gpu/` runs the real pipeline against the real driver and reads the result back.
 */
let installed = false;

export function installWebGPU() {
  if (installed) {
    return;
  }
  installed = true;
  Object.assign(globalThis, globals);
  globalThis.navigator ??= {};
  Object.defineProperty(globalThis.navigator, 'gpu', {
    value: create([]),
    configurable: true,
  });
}

/**
 * Whether a bake can actually run here. A CI runner has the Dawn binary but often no driver
 * behind it, which hands back no adapter - that is a skip, not a failure.
 */
export async function hasWebGPU() {
  installWebGPU();
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

/**
 * Every limit the adapter will give. The brush fold binds a storage texture, the brush array
 * and the volume atlas in one compute stage, and the mesh bake wants a triangle buffer far
 * past the 128 MiB default guarantee, so the defaults are not an option.
 *
 * @returns {Promise<Record<string, number>>}
 */
export async function adapterLimits() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('test/gpu: no WebGPU adapter');
  }
  /** @type {Record<string, number>} */
  const limits = {};
  // GPUSupportedLimits keeps its values in prototype getters, so for-in is the way in.
  for (const key in adapter.limits) {
    const value = adapter.limits[key];
    if (typeof value === 'number') {
      limits[key] = value;
    }
  }
  return limits;
}
