import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';

/** Pixels per edge of one cone-trace pre-pass tile (Claybook GDC'18 slide 28 used 8x8). */
export const PREPASS_TILE = 8;

/**
 * Which G-buffer layer a pass reads or writes. `opaque` is the one the deferred resolve
 * shades; `transparent` holds the nearest see-through surface in front of it.
 */
export type RenderLayer = 'opaque' | 'transparent';

/** Index of a layer's pre-pass and attachment slots. */
export const layerIndex = (layer: RenderLayer): 0 | 1 => (layer === 'opaque' ? 0 : 1);

/** G-buffer read side: albedo/roughness, normal + ray length, and the TAA history. */
export const gbufferLayout = tgpu.bindGroupLayout({
  albedo: { texture: d.texture2d(d.f32) },
  normalT: { texture: d.texture2d(d.f32) },
  history: { texture: d.texture2d(d.f32) },
  samp: { sampler: 'filtering' },
});

/**
 * Input side of the stochastic lighting pass. Only the normal/depth targets, because a
 * shadow or AO ray does not care what colour the surface is - and because the pass
 * writes {@link lightReadLayout}'s texture, which therefore cannot be in the same group.
 *
 * Both layers, in one pass: the transparency layer needs the same two lighting terms at
 * its own surface, and shading it in the pass that already has the tracer bound is
 * cheaper than a fourth pipeline that binds the field again.
 */
export const lightingInLayout = tgpu.bindGroupLayout({
  normalT: { texture: d.texture2d(d.f32) },
  normalTT: { texture: d.texture2d(d.f32) },
  samp: { sampler: 'filtering' },
});

/**
 * Read side of the noisy lighting buffer: `(shadow, ao)` for the opaque layer in `xy`,
 * the same two for the transparency layer in `zw`.
 */
export const lightReadLayout = tgpu.bindGroupLayout({
  light: { texture: d.texture2d(d.f32) },
});

/**
 * Read side of the transparency layer's G-buffer. Fetched with `textureLoad` rather than
 * sampled: the composite pass only ever wants the surface at *this* pixel, and filtering
 * a material id across a silhouette produces a material that does not exist.
 */
export const transparentReadLayout = tgpu.bindGroupLayout({
  albedoT: { texture: d.texture2d(d.f32) },
  normalTT: { texture: d.texture2d(d.f32) },
});

/** Pre-pass read side, fetched with `textureLoad` - one texel per pixel tile. */
export const prepassReadLayout = tgpu.bindGroupLayout({
  prepass: { texture: d.texture2d(d.f32) },
});

/** Pre-pass write side. */
export const prepassWriteLayout = tgpu.bindGroupLayout({
  out: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
});

function createTargets(root: TgpuRoot, width: number, height: number) {
  const pw = Math.max(1, Math.ceil(width / PREPASS_TILE));
  const ph = Math.max(1, Math.ceil(height / PREPASS_TILE));

  const albedoTex = root
    .createTexture({ size: [width, height], format: 'rgba8unorm' })
    .$usage('render', 'sampled');
  const normalTex = root
    .createTexture({ size: [width, height], format: 'rgba16float' })
    .$usage('render', 'sampled');
  const depthTex = root
    .createTexture({ size: [width, height], format: 'depth24plus' })
    .$usage('render');
  // The transparency layer: the same two channels one surface deeper. A see-through
  // surface needs what is *behind* it, and a single-layer G-buffer cannot say - so the
  // nearest transparent surface gets its own albedo/normal pair, depth-tested against
  // the opaque layer so anything hidden behind a wall is culled for free.
  const albedoTexT = root
    .createTexture({ size: [width, height], format: 'rgba8unorm' })
    .$usage('render', 'sampled');
  const normalTexT = root
    .createTexture({ size: [width, height], format: 'rgba16float' })
    .$usage('render', 'sampled');
  // One pre-pass texture per layer: a pre-pass `t` is only a valid lower bound for the
  // field it was traced against, and the two layers trace different fields.
  const prepassTex = [0, 1].map(() =>
    root.createTexture({ size: [pw, ph], format: 'rgba16float' }).$usage('storage', 'sampled'),
  );
  // Four channels, one pass: the stochastic terms are computed here and spatially
  // filtered during resolve, which is what makes 1 sample per pixel per frame usable
  // on geometry that moves too fast for the temporal filter alone (Claybook GDC'18
  // slide 38 - one ray, then filter). `zw` is the same pair for the transparency layer.
  const lightTex = root
    .createTexture({ size: [width, height], format: 'rgba16float' })
    .$usage('render', 'sampled');
  const historyTex = [0, 1].map(() =>
    root.createTexture({ size: [width, height], format: 'rgba16float' }).$usage('render', 'sampled'),
  );

  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const albedo = albedoTex.createView(d.texture2d(d.f32));
  const normalT = normalTex.createView(d.texture2d(d.f32));
  const albedoT = albedoTexT.createView(d.texture2d(d.f32));
  const normalTT = normalTexT.createView(d.texture2d(d.f32));

  return {
    width,
    height,
    prepassWidth: pw,
    prepassHeight: ph,
    albedoView: albedoTex.createView('render'),
    normalView: normalTex.createView('render'),
    depthView: depthTex.createView('render'),
    albedoViewT: albedoTexT.createView('render'),
    normalViewT: normalTexT.createView('render'),
    lightView: lightTex.createView('render'),
    lightingInGroup: root.createBindGroup(lightingInLayout, {
      normalT,
      normalTT,
      samp: sampler,
    }),
    lightReadGroup: root.createBindGroup(lightReadLayout, {
      light: lightTex.createView(d.texture2d(d.f32)),
    }),
    transparentReadGroup: root.createBindGroup(transparentReadLayout, {
      albedoT,
      normalTT,
    }),
    historyViews: historyTex.map((t) => t.createView('render')),
    readGroups: historyTex.map((t) =>
      root.createBindGroup(gbufferLayout, {
        albedo,
        normalT,
        history: t.createView(d.texture2d(d.f32)),
        samp: sampler,
      }),
    ),
    /** Indexed by {@link RenderLayer}. */
    prepassReadGroups: prepassTex.map((t) =>
      root.createBindGroup(prepassReadLayout, {
        prepass: t.createView(d.texture2d(d.f32)),
      }),
    ),
    prepassWriteGroups: prepassTex.map((t) =>
      root.createBindGroup(prepassWriteLayout, {
        out: t.createView(d.textureStorage2d('rgba16float', 'write-only')),
      }),
    ),
    destroy() {
      albedoTex.destroy();
      normalTex.destroy();
      depthTex.destroy();
      albedoTexT.destroy();
      normalTexT.destroy();
      lightTex.destroy();
      for (const t of prepassTex) {
        t.destroy();
      }
      for (const t of historyTex) {
        t.destroy();
      }
    },
  };
}

export type RenderTargets = ReturnType<typeof createTargets>;

/**
 * Render targets for the deferred SDF path.
 *
 * `normalT.w` doubles as the sky mask - negative means the primary ray escaped, which
 * saves a separate coverage target. The transparency layer reuses the convention: a
 * negative `w` there means "no see-through surface at this pixel", which is also what the
 * attachment clears to, so a pixel the transparent pass never covers needs no mask of its
 * own. History is double-buffered because the temporal
 * filter reads last frame's resolve while writing this frame's.
 */
export class GBuffer {
  readonly root: TgpuRoot;
  /** Ping-pong index of the history slot holding *last* frame's resolve. */
  historyIndex = 0;
  private targets: RenderTargets | null = null;

  constructor(root: TgpuRoot) {
    this.root = root;
  }

  get width(): number {
    return this.targets?.width ?? 0;
  }
  get height(): number {
    return this.targets?.height ?? 0;
  }
  /** Throws before the first {@link resize} - nothing can render without targets. */
  get current(): RenderTargets {
    if (!this.targets) {
      throw new Error('GBuffer: call resize() before rendering');
    }
    return this.targets;
  }

  /** Returns true when the targets were actually rebuilt. */
  resize(width: number, height: number): boolean {
    const w = Math.max(1, Math.ceil(width));
    const h = Math.max(1, Math.ceil(height));
    if (this.targets && this.targets.width === w && this.targets.height === h) {
      return false;
    }
    this.targets?.destroy();
    this.targets = createTargets(this.root, w, h);
    return true;
  }

  /** Attachment for this frame's resolved colour. */
  get historyTarget() {
    return this.current.historyViews[this.historyIndex ^ 1];
  }
  /** Bind group whose `history` texture is last frame's resolved colour. */
  get readGroup() {
    return this.current.readGroups[this.historyIndex];
  }
  /**
   * Bind group whose `history` texture is *this* frame's resolved colour - the slot the
   * resolve pass just wrote. What the transparency composite refracts: the linear,
   * pre-tonemap image of everything opaque, already temporally filtered.
   */
  get resolvedGroup() {
    return this.current.readGroups[this.historyIndex ^ 1];
  }
  /** Swap slots. Call once per frame, after the resolve pass. */
  flip(): void {
    this.historyIndex ^= 1;
  }

  destroy(): void {
    this.targets?.destroy();
    this.targets = null;
  }
}
