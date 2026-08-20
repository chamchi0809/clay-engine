import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { TracedField } from '../trace/field.ts';

const ProbeParams = d.struct({ start: d.vec3f, count: d.u32, step: d.vec3f, _pad: d.f32 });

/**
 * Reads a {@link TracedField} back to the CPU along a line segment.
 *
 * A traced field is only ever observed through a ray that stops at the first zero
 * crossing, so the *interior* of a volume can be arbitrarily wrong without anything
 * looking wrong on screen - and the collider samples exactly that interior. This is the
 * only way to see it.
 *
 * ponytail: builds a pipeline per call. It is a diagnostic, not a hot path; if it ever
 * ends up in one, hoist the pipeline and keep the buffer.
 */
export async function probeField(
  root: TgpuRoot,
  field: TracedField,
  from: readonly [number, number, number],
  to: readonly [number, number, number],
  count = 64,
): Promise<{ t: number; distance: number; band: number }[]> {
  const out = root.createMutable(d.arrayOf(d.vec2f, count));
  const params = root.createUniform(ProbeParams, {
    start: d.vec3f(...from),
    count,
    step: d.vec3f(
      (to[0] - from[0]) / (count - 1),
      (to[1] - from[1]) / (count - 1),
      (to[2] - from[2]) / (count - 1),
    ),
    _pad: 0,
  });
  const pipeline = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        const i = gid.x;
        if (i >= params.$.count) {
          return;
        }
        const p = params.$.start + params.$.step * d.f32(i);
        out.$[i] = d.vec2f(field.sample(p, 0));
      },
    ),
  });
  let pass = pipeline;
  for (const g of field.groups) {
    pass = pass.with(g);
  }
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  const raw = await out.read();
  return raw.map((v, i) => ({
    t: i / (count - 1),
    distance: +v.x.toFixed(4),
    band: +v.y.toFixed(4),
  }));
}
