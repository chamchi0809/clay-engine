/**
 * clay-engine - a signed-distance-field renderer and simulator for WebGPU.
 *
 * The layering is deliberate and one-directional:
 *
 *   math/    pure GPU helpers (quaternions, smooth min/max, hashes)
 *   field/   authoring and storage of a mip-mapped SDF volume (brushes, tiles, mips)
 *   trace/   `TracedField` + tracing and shading over *any* field, volume or analytic
 *   render/  camera, G-buffer, the passes that turn a field into pixels
 *   sim/     particle physics that reads fields and writes back into them
 *   scene.ts optional wiring of all of the above
 *
 * Nothing in `trace/`, `render/` or `sim/` knows about brushes or about Claybook. The
 * only contract between the halves is {@link TracedField}, which an analytic SDF
 * satisfies in a dozen lines - see `analyticField`.
 */

export * from './gpu.ts';
export * from './math/gpu.ts';

export * from './field/atlas.ts';
export * from './field/brush.ts';
export * from './field/builder.ts';
export * from './field/meshbake.ts';
export * from './field/mips.ts';
export * from './field/modify.ts';
export * from './field/tilegrid.ts';
export * from './field/volume.ts';

export * from './trace/field.ts';
export * from './trace/march.ts';
export * from './trace/shade.ts';

export * from './render/camera.ts';
export * from './render/composite.ts';
export * from './render/deferred.ts';
export * from './render/gbuffer.ts';
export * from './render/raymarch.ts';

export * from './sim/particles.ts';
export * from './sim/extract.ts';
export * from './sim/pbd.ts';
export * from './sim/splat.ts';
export * from './sim/meshdraw.ts';
export * from './sim/fluid.ts';

export * from './scene.ts';
export * from './shape/mesh.ts';
export * from './shape/sdf.ts';
export * from './game/index.ts';
