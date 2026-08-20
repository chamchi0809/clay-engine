import tgpu, { common, d, std } from 'typegpu';
import type {
  TgpuCommandEncoder,
  TgpuComputePipeline,
  TgpuRenderCommands,
  TgpuRenderPipeline,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
import type { TracedField } from '../trace/field.ts';
import { makeTracer, type TracerOptions } from '../trace/march.ts';
import { makePaletteSampler, type MaterialPalette } from '../trace/shade.ts';
import { CameraUniform, makeCameraRay } from './camera.ts';
import {
  GBuffer,
  PREPASS_TILE,
  prepassReadLayout,
  prepassWriteLayout,
} from './gbuffer.ts';

/** Safety factor on the tile cone's half-angle. Undershooting punches holes in silhouettes. */
const APERTURE_SLACK = 1.05;

export interface RaymarcherOptions extends TracerOptions {
  /** Iterations for the coarse 8x8 pre-pass. Cheap: one ray per 64 pixels. */
  prepassSteps?: number;
  /** Palette length, needed as a JS constant for the fractional-id lookup. */
  paletteCount: number;
}

/**
 * Primary-visibility pass: cone-trace pre-pass, then one cone per pixel into a
 * G-buffer.
 *
 * The pre-pass (Claybook GDC'18 slide 28) traces the *outer bounding cone* of each
 * 8x8 pixel block. Every ray in the block lies inside that cone, so wherever the cone
 * first touches geometry is a distance every ray in the block can skip for free -
 * 1/64th of the work removes most of the empty-space marching.
 *
 * Primary rays are cone traces rather than plain rays, which is Claybook's "perfect
 * LOD": a ray stops as soon as the remaining distance drops below its own pixel
 * footprint, so a wall fills the screen for the same cost whether it is 1 m or 100 m
 * of detail away.
 */
export class SdfRaymarcher {
  readonly gbuffer: GBuffer;
  private readonly prepassPipeline: TgpuComputePipeline;
  private readonly primaryPipeline: TgpuRenderPipeline;
  private readonly field: TracedField;

  constructor(
    root: TgpuRoot,
    field: TracedField,
    camera: TgpuUniform<typeof CameraUniform>,
    gbuffer: GBuffer,
    palette: MaterialPalette,
    options: RaymarcherOptions,
  ) {
    this.field = field;
    this.gbuffer = gbuffer;
    const ray = makeCameraRay(camera);
    /** Tangent of the angle between the tile's axis and the ray through `uv`. */
    const cornerTan = (axis: d.v3f, uv: d.v2f) => {
      'use gpu';
      const cosA = std.max(std.dot(std.normalize(ray(uv)), axis), 1e-4);
      return std.sqrt(std.max(1 / (cosA * cosA) - 1, 0));
    };
    const samplePalette = makePaletteSampler(palette, options.paletteCount);
    const tracer = makeTracer(field, options);
    const coarse = makeTracer(field, {
      ...options,
      maxSteps: options.prepassSteps ?? 64,
    });

    this.prepassPipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [8, 8],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        const c = camera.$;
        const dims = std.ceil(c.resolution * (1 / PREPASS_TILE));
        if (d.f32(gid.x) >= dims.x || d.f32(gid.y) >= dims.y) {
          return;
        }
        const centerPx = (d.vec2f(d.f32(gid.x), d.f32(gid.y)) + d.vec2f(0.5)) * PREPASS_TILE;
        const dir = std.normalize(ray(centerPx * c.invResolution));
        // The cone must *contain* every ray in the tile, so its half-angle is the
        // largest angle from the axis to any of the four block corners. Scaling one
        // pixel's aperture by the block circumradius is only right in the paraxial
        // limit and undershoots by ~40% off-centre, which punches 8-pixel holes in
        // silhouettes. Which corner wins depends on the aspect ratio and on where the
        // block sits, so measure all four - this runs once per 64 pixels.
        const h = PREPASS_TILE * 0.5;
        const px = centerPx * c.invResolution;
        const dpx = d.vec2f(h, 0) * c.invResolution;
        const dpy = d.vec2f(0, h) * c.invResolution;
        const aperture = std.max(
          std.max(cornerTan(dir, px - dpx - dpy), cornerTan(dir, px + dpx - dpy)),
          std.max(cornerTan(dir, px - dpx + dpy), cornerTan(dir, px + dpx + dpy)),
        ) * APERTURE_SLACK;
        const hit = coarse.cone(c.pos, dir, c.near, c.far, aperture);
        // `minStep` is the only forward slack left in the loop; give it back so the
        // stored `t` is a strict lower bound for every ray in the tile.
        std.textureStore(
          prepassWriteLayout.$.out,
          d.vec2u(gid.x, gid.y),
          d.vec4f(std.max(hit.t - coarse.stepSlack(), c.near), d.f32(hit.hit), 0, 1),
        );
      }),
    });

    this.primaryPipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({
        in: { pos: d.builtin.position, uv: d.vec2f },
        out: { albedo: d.vec4f, normalT: d.vec4f, depth: d.builtin.fragDepth },
      })(({ pos, uv }) => {
        'use gpu';
        const c = camera.$;
        const dir = std.normalize(ray(uv));
        const tile = d.vec2i(
          d.i32(pos.x) / d.i32(PREPASS_TILE),
          d.i32(pos.y) / d.i32(PREPASS_TILE),
        );
        const pre = std.textureLoad(prepassReadLayout.$.prepass, tile, 0);
        // The pre-pass cone stopped at the first thing it could touch, so its `t` is a
        // safe start for every ray in the tile. Missing tiles report their escape `t`,
        // which is equally safe.
        const t0 = std.max(pre.x, c.near);
        const aperture = c.tanHalfFov * c.invResolution.y;
        const hit = tracer.cone(c.pos, dir, t0, c.far, aperture);
        if (hit.hit === 0) {
          // Sky: `normalT.w < 0` is the mask the resolve pass tests.
          return {
            albedo: d.vec4f(0, 0, 0, 0),
            normalT: d.vec4f(0, 0, 0, -1),
            depth: d.f32(1),
          };
        }
        const p = c.pos + dir * hit.t;
        const id = field.field(p, 0).y;
        const m = samplePalette(id);
        const clip = c.viewProj * d.vec4f(p, 1);
        return {
          albedo: d.vec4f(m.albedo, id * (1 / 255)),
          normalT: d.vec4f(field.normal(p), hit.t),
          depth: std.clamp(clip.z / std.max(clip.w, 1e-6), 0, 1),
        };
      }),
      targets: {
        albedo: { format: 'rgba8unorm' },
        normalT: { format: 'rgba16float' },
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less-equal',
      },
    });
  }

  /** Coarse cone-trace pre-pass. Records its own compute pass into `encoder`. */
  prepass(encoder: TgpuCommandEncoder): void {
    const t = this.gbuffer.current;
    let p = this.prepassPipeline.with(encoder).with(t.prepassWriteGroup);
    for (const g of this.field.groups) {
      p = p.with(g);
    }
    p.dispatchWorkgroups(Math.ceil(t.prepassWidth / 8), Math.ceil(t.prepassHeight / 8));
  }

  /**
   * Fills the G-buffer. Draws into a caller-owned pass so rasterised geometry (clay
   * body meshes) can be appended to the same attachments and depth-test against the
   * traced world.
   */
  draw(pass: TgpuRenderCommands): void {
    let p = this.primaryPipeline.with(pass).with(this.gbuffer.current.prepassReadGroup);
    for (const g of this.field.groups) {
      p = p.with(g);
    }
    p.draw(3);
  }
}
