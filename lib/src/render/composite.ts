import tgpu, { common, d, std } from 'typegpu';
import type {
  TgpuBindGroup,
  TgpuRenderCommands,
  TgpuRenderPipeline,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
import {
  DirLight,
  fresnelSchlick,
  makePaletteSampler,
  shadeAmbient,
  shadeDirect,
  type MaterialPalette,
} from '../trace/shade.ts';
import { CameraUniform, makeCameraRay } from './camera.ts';
import { defaultSky, tonemap } from './deferred.ts';
import { gbufferLayout, lightReadLayout, transparentReadLayout } from './gbuffer.ts';

export interface CompositeOptions {
  /** Palette length, needed as a JS constant for the fractional-id lookup. */
  paletteCount: number;
  /** Swapchain format of the presented target. */
  presentFormat: GPUTextureFormat;
  /** Sky radiance. Must match the resolve's, since glass reflects it. */
  sky?: (dir: d.v3f) => d.v3f;
  /** Constant sky/bounce term multiplied by AO. Must match the resolve's. */
  ambient?: readonly [number, number, number];
  /** Exposure applied before tonemapping. Must match the resolve's. */
  exposure?: number;
  /**
   * Longest path a ray may be treated as taking through a transparent body, in world
   * units. Both the value assumed when there is nothing to measure against and the cap on
   * what is measured.
   *
   * Path length normally comes from the gap between the two G-buffer layers along one ray,
   * which is exactly right when the far side of the body *is* the thing behind it - water
   * in a basin, a pane on a wall. It is an over-estimate when the backdrop is far away: a
   * hand's width of falling water in front of a distant wall measures as metres of water
   * and goes black. Capping is the cheap correction, and it is why this is one number
   * rather than two - set it to the deepest the substance actually gets.
   *
   * The exact answer needs the body's own far side, i.e. a second trace through the
   * transparency layer's field, which this pass deliberately does not have: a field costs
   * a bind group, the composite already uses all four, and a rasterised transparent object
   * has no field to trace at all.
   */
  thickness?: number;
  /**
   * Replaces the composited pixel with one term. Compiled in as a JS constant, so an
   * unused mode costs nothing.
   *
   * `thickness` shows the absorption path length, white at `thickness` world units.
   * `refraction` shows the screen-space offset the refracted lookup used, red where the
   * offset was rejected as an occluder. `transmitted` is what came through the body,
   * `surface` the body's own light - the two halves the composite mixes, which is the
   * fastest way to tell "the transmission is wrong" from "the surface is too bright".
   */
  debug?: 'off' | 'thickness' | 'refraction' | 'transmitted' | 'surface' | 'normal' | 'material';
}

/**
 * Composites the transparency layer over the resolved opaque image: refraction,
 * Beer-Lambert absorption, a Fresnel reflection and the surface's own shading.
 *
 * Why a separate pass rather than a wider G-buffer. A see-through surface needs the
 * *radiance* behind it, not the geometry - and by the time the deferred resolve has run,
 * that radiance already exists, shaded, denoised and temporally filtered, in the history
 * target this frame just wrote. Refraction is then a projection instead of a second trace:
 * bend the view ray at the surface, walk it as far as the body is thick, project that
 * point back to the screen and read the resolved image there. One texture fetch buys what
 * a per-pixel secondary trace through the whole scene would otherwise cost, and it works
 * identically for a surface that was ray-marched and one that was rasterised, because both
 * wrote the same two attachments.
 *
 * What it therefore cannot do, and the reasons are the same reason:
 *
 *   - **One layer.** Two panes of glass in a line show the nearer one over the opaque
 *     image, not through each other. The G-buffer holds one see-through surface per pixel.
 *   - **Nothing hidden refracts into view.** The refracted lookup can only find what the
 *     opaque image already contains, so a bent ray that should reveal something outside
 *     the frame - or behind a foreground object - falls back to the straight lookup rather
 *     than inventing it. That fallback is the visible cost, and it is why the offset is
 *     validated against the opaque layer's depth instead of used blind.
 *   - **A transparent object does not shadow, occlude or bounce light.** It is not in the
 *     field the shadow, AO and reflection terms trace against - deliberately, since clear
 *     water casting a solid shadow is the worse artefact.
 *
 * Blending, not `discard`: a pixel with no see-through surface returns zero coverage and
 * the presented image passes through untouched.
 */
export class TransparentComposite {
  private readonly pipeline: TgpuRenderPipeline;

  constructor(
    root: TgpuRoot,
    camera: TgpuUniform<typeof CameraUniform>,
    light: TgpuUniform<typeof DirLight>,
    palette: MaterialPalette,
    options: CompositeOptions,
  ) {
    const ray = makeCameraRay(camera);
    const samplePalette = makePaletteSampler(palette, options.paletteCount);
    const sky = options.sky ?? defaultSky;
    const ambient = options.ambient ?? [0.25, 0.26, 0.3];
    const ambientVec = d.vec3f(ambient[0], ambient[1], ambient[2]);
    const exposure = options.exposure ?? 1.1;
    const fallbackThickness = options.thickness ?? 1;
    const debug = options.debug ?? 'off';
    const INV_PI = 1 / 3.14159265;

    this.pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({
        in: { pos: d.builtin.position, uv: d.vec2f },
        out: d.vec4f,
      })(({ pos, uv }) => {
        'use gpu';
        const ip = d.vec2i(d.i32(pos.x), d.i32(pos.y));
        const ntT = std.textureLoad(transparentReadLayout.$.normalTT, ip, 0);
        // Negative `w` is the layer's "nothing here" mask, and also its clear value, so
        // an uncovered pixel costs one fetch and zero coverage.
        if (ntT.w < 0) {
          return d.vec4f();
        }

        const c = camera.$;
        const dir = std.normalize(ray(uv));
        const tFront = ntT.w;
        const p = c.pos + dir * tFront;
        // Face the eye. A primary ray can legitimately land on the far side of a thin
        // sheet - a splash, the underside of a wave - and an away-facing normal would
        // refract the wrong way and reflect the ground instead of the sky.
        const n0 = std.normalize(ntT.xyz);
        const n = std.select(std.neg(n0), n0, std.dot(n0, dir) < 0);

        const albT = std.textureLoad(transparentReadLayout.$.albedoT, ip, 0);
        const id = std.round(albT.w * 255);
        const m = samplePalette(id);

        // --- how far the ray travels inside the body --------------------------
        // The opaque layer at this pixel *is* the far side: the distance between the two
        // layers along one ray is the path length through whatever sits between them.
        // That is free, exact for the common cases (a pool in a basin, a pane on a wall),
        // and needs no second trace - the reason absorption can be a length integral here
        // at all.
        const nt = std.textureLoad(gbufferLayout.$.normalT, ip, 0);
        const hasBackdrop = nt.w > 0;
        const thickness = std.select(
          d.f32(fallbackThickness),
          std.clamp(nt.w - tFront, 0, fallbackThickness),
          hasBackdrop,
        );

        // --- refraction, in screen space -------------------------------------
        const eta = 1 / std.max(m.ior, 1e-3);
        const bent = std.refract(dir, n, eta);
        // Total internal reflection makes `refract` return zero; the straight ray is the
        // only sane fallback and is what an `ior` of 1 gives anyway.
        const rd = std.select(dir, std.normalize(bent), std.dot(bent, bent) > 1e-6);
        const exit = p + rd * thickness;
        const clip = c.viewProj * d.vec4f(exit, 1);
        const ndc = clip.xy * (1 / std.max(std.abs(clip.w), 1e-6));
        const uvR = std.clamp(
          d.vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5),
          d.vec2f(0),
          d.vec2f(1),
        );
        // The offset is only usable if it lands on a surface *behind* this one. Two things
        // it therefore rejects, both of which look far worse than not refracting at all:
        // something in front of the glass, smeared into it; and open sky, which is the
        // loud one - a puddle at the edge of a plate would otherwise refract the bright
        // sky just past that edge into itself and read as milk.
        const px = uvR * c.resolution;
        const occl = std.textureLoad(
          gbufferLayout.$.normalT,
          d.vec2i(d.i32(px.x), d.i32(px.y)),
          0,
        );
        const usable = clip.w > 0 && occl.w > tFront;
        const uvB = std.select(uv, uvR, usable);

        // The resolved opaque image, linear and already denoised - this frame's history
        // slot, written by the pass immediately before this one.
        const behind = std.textureSampleLevel(
          gbufferLayout.$.history,
          gbufferLayout.$.samp,
          uvB,
          d.f32(0),
        ).xyz;
        // Against open sky, refract the sky itself rather than the neighbouring pixel:
        // the gradient is smooth, so a screen-space offset would show nothing at all.
        const backdrop = std.select(sky(rd), behind, hasBackdrop);

        // Beer-Lambert, tinted by `1 - albedo`: a blue material absorbs everything but
        // blue, and it does so per unit travelled, so a shallow edge stays clear while
        // the deep middle goes saturated.
        const absorb = std.exp(
          std.neg((d.vec3f(1) - m.albedo) * (std.max(m.absorption, 0) * thickness)),
        );
        const transmitted = backdrop * absorb;

        // --- the surface's own light -----------------------------------------
        const lt = std.textureLoad(lightReadLayout.$.light, ip, 0);
        const shadow = lt.z;
        const ao = lt.w;
        const l = light.$;
        const toLight = std.neg(l.dir);
        const sun = l.color * (l.intensity * shadow);
        const ndl = std.max(std.dot(n, toLight), 0);
        // Diffuse and specular are split rather than taken from one `shadeDirect` call,
        // because only the diffuse half is what transmission replaces. A highlight on
        // water is not weaker for the water being clear - passing zero albedo through the
        // same BRDF is the specular term on its own.
        const spec = shadeDirect(n, std.neg(dir), toLight, d.vec3f(), m.roughness) * sun;
        const diffuse = m.albedo * (INV_PI * ndl) * sun + shadeAmbient(m.albedo, ambientVec, ao);

        const opacity = std.clamp(m.opacity, 0, 1);
        const body = std.mix(transmitted, diffuse, opacity);
        // Fresnel over the top: head-on it is 4% and the surface is a window, at a
        // grazing angle it goes to 1 and the same surface is a mirror.
        const fres = fresnelSchlick(std.max(std.dot(n, std.neg(dir)), 0), 0.04);
        const reflected = sky(std.reflect(dir, n));
        const surface = std.mix(diffuse, reflected, fres) + spec + m.emissive;
        const linear = std.mix(body, reflected, fres) + spec + m.emissive;

        let shown = tonemap(linear * exposure);
        if (debug === 'transmitted') {
          shown = tonemap(transmitted * exposure);
        } else if (debug === 'surface') {
          shown = tonemap(surface * exposure);
        } else if (debug === 'thickness') {
          shown = d.vec3f(thickness / std.max(d.f32(fallbackThickness), 1e-3));
        } else if (debug === 'refraction') {
          shown = std.select(
            d.vec3f(1, 0, 0),
            d.vec3f(std.abs(uvR.x - uv.x) * 20, std.abs(uvR.y - uv.y) * 20, 0),
            usable,
          );
        } else if (debug === 'normal') {
          shown = n * 0.5 + d.vec3f(0.5);
        } else if (debug === 'material') {
          shown = m.albedo;
        }
        return d.vec4f(shown, 1);
      }),
      targets: {
        format: options.presentFormat,
        // Coverage is 0 or 1 - the composite above already did the mixing, so this only
        // decides whether the pixel is replaced at all. Alpha is left as the swapchain
        // had it, since the canvas is configured opaque.
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
        },
      },
    });
  }

  /**
   * Composites over the presented image. `pass` must have the swapchain view as its only
   * attachment, loaded rather than cleared.
   *
   * `gbufferGroup` must be {@link GBuffer.resolvedGroup} - the history slot the resolve
   * pass just wrote, not the one it read.
   */
  draw(
    pass: TgpuRenderCommands,
    gbufferGroup: TgpuBindGroup,
    lightGroup: TgpuBindGroup,
    transparentGroup: TgpuBindGroup,
  ): void {
    this.pipeline.with(pass).with(gbufferGroup).with(lightGroup).with(transparentGroup).draw(3);
  }
}
