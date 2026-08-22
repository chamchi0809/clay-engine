import tgpu, { common, d, std } from 'typegpu';
import type {
  TgpuBindGroup,
  TgpuRenderCommands,
  TgpuRenderPipeline,
  TgpuRoot,
  TgpuUniform,
} from 'typegpu';
import type { TracedField } from '../trace/field.ts';
import {
  DirLight,
  makePaletteSampler,
  makeShading,
  shadeAmbient,
  shadeDirect,
  type MaterialPalette,
  type ShadingOptions,
} from '../trace/shade.ts';
import { CameraUniform, makeCameraRay } from './camera.ts';
import { gbufferLayout, lightReadLayout, lightingInLayout } from './gbuffer.ts';

export interface DeferredOptions extends ShadingOptions {
  paletteCount: number;
  /** Swapchain format of the presented target. */
  presentFormat: GPUTextureFormat;
  /**
   * Also shade the transparency layer's `(shadow, ao)` into the lighting target's `zw`.
   * Compiled in as a JS constant, so a scene with nothing see-through in it pays for
   * neither the extra ray nor the extra cone.
   */
  transparent?: boolean;
  /** Weight of the new sample in the temporal filter. Lower = smoother, laggier. */
  taaAlpha?: number;
  /**
   * Width of the neighbourhood colour box the reprojected history is clamped into, in
   * standard deviations. Smaller = less ghosting, more noise. 0 disables the clamp.
   */
  taaClamp?: number;
  /** Constant sky/bounce term multiplied by AO. */
  ambient?: readonly [number, number, number];
  /** Sky radiance for rays that escaped. Defaults to a two-tone gradient. */
  sky?: (dir: d.v3f) => d.v3f;
  /** Exposure applied before tonemapping. */
  exposure?: number;
  /**
   * Replaces the presented image with one lighting term. Compiled in as a JS constant,
   * so an unused mode costs nothing. `taa` shows where the temporal filter rejected
   * history (red) - the fastest way to tell a noisy estimator from a broken reprojection.
   */
  debug?: 'off' | 'shadow' | 'ao' | 'normal' | 'material' | 'taa';
  /**
   * Radius in pixels of the spatial filter applied to the stochastic terms. 2 means a
   * 5x5 box, 0 disables it. The filter is what makes fast-moving geometry usable: the
   * temporal filter rejects history exactly where the picture changes, and one shadow
   * ray plus one AO cone on their own are salt-and-pepper.
   */
  filterRadius?: number;
}

/** The engine's stand-in environment: a two-tone gradient. Also what glass reflects. */
export const defaultSky = (dir: d.v3f) => {
  'use gpu';
  const up = std.clamp(dir.y * 0.5 + 0.5, 0, 1);
  return std.mix(d.vec3f(0.35, 0.33, 0.4), d.vec3f(0.55, 0.72, 1.0), up);
};

/**
 * Deferred resolve: ray-traced shadows, ray-traced AO, direct lighting, temporal
 * accumulation, tonemap.
 *
 * Claybook had no baked lighting at all (slide 5) - every term here is a distance
 * field trace, one stochastic sample per pixel per frame, denoised over time. That is
 * also why the temporal filter is not optional: a single AO cone and a single jittered
 * shadow ray per pixel are far too noisy to show raw.
 *
 * Writes two targets in one pass: the presented image and the linear history the next
 * frame reprojects into. `history.w` carries the traced distance, which is what makes
 * the reprojection test cheap - compare it against the distance from *last* frame's
 * eye to this frame's world point.
 */
export class DeferredResolve {
  private readonly pipeline: TgpuRenderPipeline;
  private readonly lightPipeline: TgpuRenderPipeline;
  private readonly field: TracedField;

  constructor(
    root: TgpuRoot,
    field: TracedField,
    camera: TgpuUniform<typeof CameraUniform>,
    light: TgpuUniform<typeof DirLight>,
    palette: MaterialPalette,
    options: DeferredOptions,
  ) {
    this.field = field;
    const ray = makeCameraRay(camera);
    const samplePalette = makePaletteSampler(palette, options.paletteCount);
    const shading = makeShading(field, options);
    const sky = options.sky ?? defaultSky;
    const taaAlpha = options.taaAlpha ?? 0.1;
    const ambient = options.ambient ?? [0.25, 0.26, 0.3];
    const exposure = options.exposure ?? 1.1;
    const ambientVec = d.vec3f(ambient[0], ambient[1], ambient[2]);
    const bias = field.voxelWorld(0);
    const debug = options.debug ?? 'off';
    const withTransparent = options.transparent ?? false;
    const filterRadius = Math.max(0, Math.round(options.filterRadius ?? 2));
    const taaClamp = Math.max(0, options.taaClamp ?? 2);

    // --- stochastic terms, own pass ----------------------------------------
    // Split out so the resolve below can read a *neighbourhood* of them. Sky pixels
    // write (1, 1) so a filter tap that lands on one still reads as "lit".
    this.lightPipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({
        in: { pos: d.builtin.position, uv: d.vec2f },
        // vec4f: `xy` is the opaque layer's (shadow, ao), `zw` the transparency layer's.
        out: d.vec4f,
      })(({ pos, uv }) => {
        'use gpu';
        const c = camera.$;
        const l = light.$;
        const toLight = std.neg(l.dir);
        const dir = std.normalize(ray(uv));
        const seed = d.u32(pos.x) * 1973 + d.u32(pos.y) * 9277 + c.frame * 26699;

        // The transparency layer, if the scene has one. Deterministic queries, because
        // the composite that consumes these is downstream of the temporal filter and has
        // nothing to average noise into.
        let shadowT = d.f32(0);
        let aoT = d.f32(0);
        if (withTransparent) {
          const ntT = std.textureSampleLevel(
            lightingInLayout.$.normalTT,
            lightingInLayout.$.samp,
            uv,
            d.f32(0),
          );
          if (ntT.w > 0) {
            const pT = c.pos + dir * ntT.w;
            // Face the eye: the primary ray can land on either side of a thin sheet, and
            // an away-facing normal biases the ray start *into* the surface.
            const nT0 = std.normalize(ntT.xyz);
            const nT = std.select(std.neg(nT0), nT0, std.dot(nT0, dir) < 0);
            shadowT = shading.hardShadow(pT, nT, toLight, c.far);
            aoT = shading.axisAO(pT, nT);
          }
        }

        const nt = std.textureSampleLevel(
          lightingInLayout.$.normalT,
          lightingInLayout.$.samp,
          uv,
          d.f32(0),
        );
        if (nt.w < 0) {
          return d.vec4f(1, 1, shadowT, aoT);
        }
        const p = c.pos + dir * nt.w;
        const n = std.normalize(nt.xyz);
        return d.vec4f(
          shading.softShadow(p, n, toLight, c.far, l.size, seed),
          shading.coneAO(p, n, seed + 7919),
          shadowT,
          aoT,
        );
      }),
      targets: { format: 'rgba16float' },
    });

    this.pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({
        in: { pos: d.builtin.position, uv: d.vec2f },
        out: { color: d.vec4f, history: d.vec4f },
      })(({ pos, uv }) => {
        'use gpu';
        const c = camera.$;
        const dir = std.normalize(ray(uv));
        // `textureSampleLevel`, not `textureSample`: everything below the sky branch
        // runs in non-uniform control flow, where implicit-LOD sampling is illegal.
        const nt = std.textureSampleLevel(
          gbufferLayout.$.normalT,
          gbufferLayout.$.samp,
          uv,
          d.f32(0),
        );

        if (nt.w < 0) {
          const s = sky(dir);
          return {
            color: d.vec4f(tonemap(s * exposure), 1),
            // Negative w marks "no surface", so next frame rejects this pixel.
            history: d.vec4f(s, -1),
          };
        }

        const t = nt.w;
        const n = std.normalize(nt.xyz);
        const p = c.pos + dir * t;
        const alb = std.textureSampleLevel(
          gbufferLayout.$.albedo,
          gbufferLayout.$.samp,
          uv,
          d.f32(0),
        );
        const m = samplePalette(std.round(alb.w * 255));
        const l = light.$;
        const toLight = std.neg(l.dir);

        // --- spatial filter of the stochastic terms ---------------------------
        // A box of taps, each kept only if it sits on the same surface: same distance
        // to within a couple of voxels and a normal pointing the same way. Without
        // those two tests the filter would smear shadow across silhouettes, which is
        // the one artefact worse than the noise it removes.
        const ip = d.vec2i(d.i32(pos.x), d.i32(pos.y));
        let sum = d.vec2f();
        let wsum = d.f32(0);
        // First two moments of the neighbourhood's shaded colour, for the temporal
        // clamp further down. Accumulated in the same loop because every tap already
        // has its normal and its lighting loaded.
        let m1 = d.vec3f();
        let m2 = d.vec3f();
        for (let dy = -filterRadius; dy <= filterRadius; dy++) {
          for (let dx = -filterRadius; dx <= filterRadius; dx++) {
            const q = ip + d.vec2i(d.i32(dx), d.i32(dy));
            const qn = std.textureLoad(gbufferLayout.$.normalT, q, 0);
            // Out-of-bounds loads read zero, which fails this test - no clamping needed.
            const near = std.abs(qn.w - t) < std.max(0.03 * t, bias * 4);
            // 0.6, not 0.85: a hard normal test is what starves this filter exactly
            // where it is needed. A blobby surface - water as a union of spheres, or
            // anything at grazing incidence - turns over more than 30 degrees between
            // adjacent pixels, every tap gets rejected, and the "filtered" value is one
            // raw sample again. 53 degrees still refuses to cross a silhouette, because
            // the distance test above already did.
            if (qn.w > 0 && near && std.dot(std.normalize(qn.xyz), n) > 0.6) {
              const ql = std.textureLoad(lightReadLayout.$.light, q, 0).xy;
              sum = sum + ql;
              wsum = wsum + 1;
              // The clamp box only needs the immediate neighbours; widening the
              // denoiser should not also multiply the shading done for the moments.
              if (taaClamp > 0 && std.abs(dx) <= 1 && std.abs(dy) <= 1) {
                const qa = std.textureLoad(gbufferLayout.$.albedo, q, 0);
                const qm = samplePalette(std.round(qa.w * 255));
                const qlit = shadeDirect(std.normalize(qn.xyz), std.neg(dir), toLight, qa.xyz, qm.roughness)
                    * (l.color * (l.intensity * ql.x))
                  + shadeAmbient(qa.xyz, ambientVec, ql.y)
                  + qm.emissive;
                m1 = m1 + qlit;
                m2 = m2 + qlit * qlit;
              }
            }
          }
        }
        const filtered = sum * (1 / std.max(wsum, 1));
        const shadow = filtered.x;
        const ao = filtered.y;

        const direct = shadeDirect(n, std.neg(dir), toLight, alb.xyz, m.roughness)
          * (l.color * (l.intensity * shadow));
        const lit = direct + shadeAmbient(alb.xyz, ambientVec, ao) + m.emissive;

        // --- temporal accumulation ---------------------------------------------
        const prevClip = c.prevViewProj * d.vec4f(p, 1);
        const prevNdc = prevClip.xy * (1 / std.max(prevClip.w, 1e-6));
        const prevUv = d.vec2f(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
        const hist = std.textureSampleLevel(
          gbufferLayout.$.history,
          gbufferLayout.$.samp,
          prevUv,
          d.f32(0),
        );
        // Valid only if the point was on screen last frame, in front of the old eye,
        // and at the distance the history says it was - that last test is the
        // disocclusion check.
        const onScreen = prevClip.w > 0
          && prevUv.x > 0 && prevUv.x < 1 && prevUv.y > 0 && prevUv.y < 1;
        const expected = std.length(p - c.prevPos);
        const consistent = hist.w > 0 && std.abs(hist.w - expected) < std.max(0.02 * expected, bias * 2);
        const blend = std.select(d.f32(1), taaAlpha, onScreen && consistent);
        // Neighbourhood colour clamp (Salvi's variance form). The reprojection test
        // above only checks *distance*, and a surface that moves while staying the same
        // distance from the eye - flowing water, a rolling body - passes it, so a 10-plus
        // frame history smears visibly behind it. Constraining the history to the colour
        // range this frame's neighbourhood actually produced kills the trail in one or
        // two frames without giving up the accumulation everywhere else.
        const invN = 1 / std.max(wsum, 1);
        const mu = m1 * invN;
        const sd = std.sqrt(std.max(m2 * invN - mu * mu, d.vec3f()));
        // `lit` is folded in because the moments come from unfiltered single-sample
        // lighting: on a lone pixel the deviation is zero and the box would otherwise
        // collapse onto a value the centre pixel never had.
        const lo = std.min(mu - sd * taaClamp, lit);
        const hi = std.max(mu + sd * taaClamp, lit);
        const kept = std.select(hist.xyz, std.clamp(hist.xyz, lo, hi), taaClamp > 0);
        const acc = std.mix(kept, lit, blend);

        // Debug views bypass the tonemap but keep the history write, so toggling one
        // does not reset the accumulation of the others.
        let shown = tonemap(acc * exposure);
        if (debug === 'shadow') {
          shown = d.vec3f(shadow);
        } else if (debug === 'ao') {
          shown = d.vec3f(ao);
        } else if (debug === 'normal') {
          shown = n * 0.5 + d.vec3f(0.5);
        } else if (debug === 'material') {
          shown = alb.xyz;
        } else if (debug === 'taa') {
          shown = std.select(d.vec3f(1, 0, 0), d.vec3f(0.15), onScreen && consistent);
        }

        return {
          color: d.vec4f(shown, 1),
          history: d.vec4f(acc, t),
        };
      }),
      targets: {
        color: { format: options.presentFormat },
        history: { format: 'rgba16float' },
      },
    });
  }

  /** Fills the `(shadow, ao)` target. `pass` must have it as its only attachment. */
  drawLighting(pass: TgpuRenderCommands, lightingInGroup: TgpuBindGroup): void {
    let p = this.lightPipeline.with(pass).with(lightingInGroup);
    for (const g of this.field.groups) {
      p = p.with(g);
    }
    p.draw(3);
  }

  /**
   * `pass` must have `color` bound to the presented view and `history` to the target slot.
   *
   * No field groups here, unlike {@link drawLighting}: this pass traces nothing. It reads
   * the g-buffer, the filtered `(shadow, ao)` and the palette, and every distance-field
   * query was already spent by the pass above. Binding the field anyway cost two of the
   * four bind groups a device guarantees, for shaders that never touch them - which is
   * exactly the budget a second occluder needs.
   */
  draw(pass: TgpuRenderCommands, gbufferGroup: TgpuBindGroup, lightGroup: TgpuBindGroup): void {
    this.pipeline.with(pass).with(gbufferGroup).with(lightGroup).draw(3);
  }
}

/** Narkowicz ACES approximation - one mad and a divide, no LUT. */
export const tonemap = (x: d.v3f) => {
  'use gpu';
  const a = x * (x * 2.51 + d.vec3f(0.03));
  const b = x * (x * 2.43 + d.vec3f(0.59)) + d.vec3f(0.14);
  return std.pow(std.clamp(a / b, d.vec3f(0), d.vec3f(1)), d.vec3f(1 / 2.2));
};
