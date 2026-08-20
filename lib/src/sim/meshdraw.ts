import tgpu, { d, std } from 'typegpu';
import type { TgpuRenderCommands, TgpuRenderPipeline, TgpuRoot, TgpuUniform } from 'typegpu';
import { quatRotate } from '../math/gpu.ts';
import { CameraUniform } from '../render/camera.ts';
import { makePaletteSampler, type MaterialPalette } from '../trace/shade.ts';
import type { SurfaceExtractor } from './extract.ts';

export interface ParticleMeshOptions {
  paletteCount: number;
  /** Overrides the per-particle material id. Useful for a uniform-coloured body. */
  material?: number;
}

/**
 * Rasterises an extracted particle cloud into the same G-buffer the raymarcher fills,
 * so simulated bodies depth-test and shade exactly like traced geometry.
 *
 * There is no vertex buffer and no index buffer in the WebGPU sense: the draw is
 * `drawIndirect` over a GPU-written vertex count, and the vertex shader reads
 * `particles[indices[vertexIndex]]`. Nothing about the mesh ever reaches the CPU -
 * extraction, simulation and drawing are one GPU-resident chain, which is what lets a
 * body be re-extracted every frame (Claybook GDC'18 slides 45-48).
 *
 * Normals come from the *rest* normal rotated by the body's fitted rotation rather
 * than from the triangle: the rest normal is the field's gradient at extraction time,
 * so it is smooth, and shape matching already gives the rigid part of the deformation
 * for free.
 *
 * ponytail: the mesh is not fed back into the traced field, so a clay body receives
 * world shadows and AO but casts none, and does not occlude itself. The upgrade path is
 * the same sparse bake the fluid uses - splat particles into a volume and `unionField`
 * it with the world.
 */
export class ParticleMesh {
  private readonly pipeline: TgpuRenderPipeline;
  private readonly extractor: SurfaceExtractor;

  constructor(
    root: TgpuRoot,
    extractor: SurfaceExtractor,
    camera: TgpuUniform<typeof CameraUniform>,
    palette: MaterialPalette,
    options: ParticleMeshOptions,
  ) {
    this.extractor = extractor;
    const samplePalette = makePaletteSampler(palette, options.paletteCount);
    // A vertex stage may only bind read-only storage, so bind read-only views of the
    // very same buffers the compute passes write.
    const indices = extractor.indices.buffer.as('readonly');
    const particles = extractor.set.particles.buffer.as('readonly');
    const bodies = extractor.set.bodies.buffer.as('readonly');
    const forced = options.material;

    const vertex = tgpu.vertexFn({
      in: { vi: d.builtin.vertexIndex },
      out: {
        pos: d.builtin.position,
        world: d.vec3f,
        normal: d.vec3f,
        mat: d.f32,
      },
    })(({ vi }) => {
      'use gpu';
      const p = particles.$[indices.$[vi]];
      const b = bodies.$[p.body];
      return {
        pos: camera.$.viewProj * d.vec4f(p.pos, 1),
        world: p.pos,
        normal: quatRotate(b.rot, p.restNormal),
        mat: p.material,
      };
    });

    const fragment = tgpu.fragmentFn({
      in: { world: d.vec3f, normal: d.vec3f, mat: d.f32 },
      out: { albedo: d.vec4f, normalT: d.vec4f },
    })(({ world, normal, mat }) => {
      'use gpu';
      const id = forced === undefined ? mat : d.f32(forced);
      const m = samplePalette(id);
      const n = std.normalize(normal);
      // `.w` must be the distance from the eye: the resolve pass reconstructs the world
      // position as `camera.pos + dir * t`, and `dir` is the ray through this pixel.
      return {
        albedo: d.vec4f(m.albedo, id * (1 / 255)),
        normalT: d.vec4f(n, std.length(world - camera.$.pos)),
      };
    });

    this.pipeline = root.createRenderPipeline({
      vertex,
      fragment,
      targets: {
        albedo: { format: 'rgba8unorm' },
        normalT: { format: 'rgba16float' },
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less-equal',
      },
      primitive: { cullMode: 'back' },
    });
  }

  /** Appends the body to a G-buffer pass. Vertex count comes from the GPU. */
  draw(pass: TgpuRenderCommands): void {
    this.pipeline.with(pass).drawIndirect(this.extractor.drawArgsBuffer);
  }
}
