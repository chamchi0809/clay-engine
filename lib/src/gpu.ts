/**
 * The slice of TypeGPU a game needs in order to write its own fields, uniforms and
 * `'use gpu'` closures, re-exported so that a game depends on `@clay/engine` and on
 * nothing else.
 *
 * This is not politeness about import paths. `TracedField` is extended by handing the
 * engine shader closures written in the game's own source, and those closures and the
 * engine's must be built by one TypeGPU - two copies resolved from two dependency
 * declarations produce two sets of incompatible schema objects, and the failure shows
 * up as a resolution error deep inside a pipeline. One import, one instance.
 *
 * See also `@clay/engine/vite`, which is the build half of the same contract.
 */
export { d, std } from 'typegpu';

export type {
  TgpuBindGroupLayout,
  TgpuBuffer,
  TgpuComputePass,
  TgpuComputePipeline,
  TgpuMutable,
  TgpuReadonly,
  TgpuRenderCommands,
  TgpuRenderPipeline,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
