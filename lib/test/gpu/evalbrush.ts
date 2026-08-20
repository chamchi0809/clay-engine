import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { Brush, defaultBrushSet, type BrushDesc, type BrushSet } from '../../src/field/brush.ts';

/**
 * Evaluates one brush at a list of points, through a set's real fold, on the GPU.
 *
 * This is the whole point of `test/gpu/`: `evalBrush` is the function every bake, every
 * collider query and every ray step goes through, and the only way to know what it returns
 * is to run it on a driver and read the buffer back.
 */
export function evalWith(root: TgpuRoot, set: BrushSet = defaultBrushSet) {
  const evalBrush = set.evalBrush;
  return async (
    brush: BrushDesc,
    points: readonly [number, number, number][],
  ): Promise<number[]> => {
    const n = points.length;
    const brushBuf = root.createUniform(Brush, set.make(brush));
    const pointsBuf = root.createReadonly(d.arrayOf(d.vec3f, n));
    pointsBuf.write(points.map(([x, y, z]) => d.vec3f(x, y, z)));
    const out = root.createMutable(d.arrayOf(d.f32, n));

    let pipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x >= d.u32(n)) {
          return;
        }
        out.$[gid.x] = evalBrush(brushBuf.$, pointsBuf.$[gid.x]);
      }),
    });
    // Empty for an analytic-only set; one group when the set samples a mesh atlas.
    for (const g of set.groups) {
      pipeline = pipeline.with(g);
    }
    pipeline.dispatchWorkgroups(Math.ceil(n / 64));
    return [...(await out.buffer.read())];
  };
}

export const near = (actual: number, expected: number, msg: string, eps = 2e-3): void => {
  if (!(Math.abs(actual - expected) < eps)) {
    throw new Error(`${msg}: expected ~${expected}, got ${actual}`);
  }
};
