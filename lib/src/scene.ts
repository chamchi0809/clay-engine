import tgpu from 'typegpu';
import type { TgpuCommandEncoder, TgpuRenderPass, TgpuRoot, TgpuUniform } from 'typegpu';
import { compileBrushes, type BrushDesc, type BrushValue } from './field/brush.ts';
import { SdfBuilder } from './field/builder.ts';
import { SdfEditor } from './field/modify.ts';
import { SdfVolume, type SdfVolumeOptions } from './field/volume.ts';
import { Camera, type CameraOptions } from './render/camera.ts';
import { DeferredResolve, type DeferredOptions } from './render/deferred.ts';
import { GBuffer } from './render/gbuffer.ts';
import { SdfRaymarcher } from './render/raymarch.ts';
import { volumeField, type TracedField } from './trace/field.ts';
import {
  createPalette,
  DirLight,
  type DirLightValue,
  type MaterialPalette,
  type MaterialSpec,
} from './trace/shade.ts';
import type { TracerOptions } from './trace/march.ts';

export interface SdfSceneOptions {
  canvas: HTMLCanvasElement;
  materials: readonly MaterialSpec[];
  light?: DirLightValue;
  volume?: SdfVolumeOptions;
  camera?: CameraOptions;
  tracer?: TracerOptions;
  shading?: Omit<DeferredOptions, 'paletteCount' | 'presentFormat' | 'transparent'>;
  /** Upper bound on brushes in one full rebuild. */
  maxBrushes?: number;
  /** Device pixel ratio cap. 1 is a good default for a raymarcher. */
  maxPixelRatio?: number;
  /**
   * Replaces the field the renderer traces. Whatever comes back is what gets traced,
   * shadowed and AO'd.
   *
   * It runs after the root, the volume, the camera and the palette exist but before any
   * render pipeline does, which is deliberately the point where a game can construct
   * everything of its own that needs to be *in* the field - a fluid bake, a clay body's
   * bake, a morph target - and hand them all back as one `unionField`. Anything created
   * later cannot be traced, because a pipeline bakes its field into shader code.
   */
  compose?: (ctx: SdfSceneContext) => TracedField;
}

/** What {@link SdfSceneOptions.compose} gets. Everything that exists before pipelines do. */
export interface SdfSceneContext {
  root: TgpuRoot;
  /** The brush-built volume, on its own. */
  world: TracedField;
  /** The scene camera, for rasterised geometry that has to share the G-buffer. */
  camera: Camera;
  palette: MaterialPalette;
  paletteCount: number;
}

/**
 * Convenience wiring of the whole render path over one {@link SdfVolume}: build the
 * field from brushes, edit it at runtime, trace it, shade it, present it.
 *
 * Nothing here is required to use the engine - every part is constructible on its own
 * over any {@link TracedField}. This exists so a game does not have to re-derive the
 * pass order, and so the two extension points a game actually needs (`onSimulate` for
 * compute work, `onDrawGeometry` for rasterised geometry sharing the G-buffer) are in
 * one obvious place.
 */
export class SdfScene {
  readonly root: TgpuRoot;
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly presentFormat: GPUTextureFormat;

  readonly volume: SdfVolume;
  readonly builder: SdfBuilder;
  readonly editor: SdfEditor;
  /** The brush-built volume on its own, before {@link SdfSceneOptions.compose}. */
  readonly worldField: TracedField;
  /** What the renderer actually traces. */
  readonly field: TracedField;
  readonly gbuffer: GBuffer;
  readonly camera: Camera;
  readonly light: TgpuUniform<typeof DirLight>;
  readonly raymarcher: SdfRaymarcher;
  readonly resolve: DeferredResolve;
  /** Exposed so rasterised geometry drawn through `onDrawGeometry` shades identically. */
  readonly palette: MaterialPalette;

  /** Extra compute work, recorded before the frame's render passes. */
  onSimulate: ((encoder: TgpuCommandEncoder) => void) | null = null;
  /** Extra draws into the G-buffer pass; they depth-test against the traced world. */
  onDrawGeometry: ((pass: TgpuRenderPass) => void) | null = null;

  private readonly maxPixelRatio: number;
  private rebuildQueued = false;

  static async create(options: SdfSceneOptions): Promise<SdfScene> {
    const root = await tgpu.init();
    return new SdfScene(root, options);
  }

  constructor(root: TgpuRoot, options: SdfSceneOptions) {
    this.root = root;
    this.canvas = options.canvas;
    this.maxPixelRatio = options.maxPixelRatio ?? 1;
    const context = options.canvas.getContext('webgpu');
    if (!context) {
      throw new Error('SdfScene: canvas has no webgpu context');
    }
    this.context = context;
    this.presentFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: root.device,
      format: this.presentFormat,
      alphaMode: 'opaque',
    });

    this.volume = new SdfVolume(root, options.volume);
    this.builder = new SdfBuilder(this.volume, { maxBrushes: options.maxBrushes });
    this.editor = new SdfEditor(this.volume);
    this.worldField = volumeField(this.volume);
    this.gbuffer = new GBuffer(root);
    this.camera = new Camera(root, options.camera);

    const palette = createPalette(root, options.materials);
    this.palette = palette;
    // Before every pipeline, after everything a game's own field might need.
    this.field = options.compose
      ? options.compose({
          root,
          world: this.worldField,
          camera: this.camera,
          palette,
          paletteCount: options.materials.length,
        })
      : this.worldField;
    this.light = root.createUniform(
      DirLight,
      options.light ?? {
        dir: [-0.45, -0.78, -0.43],
        size: 0.06,
        color: [1.0, 0.95, 0.85],
        intensity: 3.2,
      },
    );

    this.raymarcher = new SdfRaymarcher(
      root,
      this.field,
      this.camera.uniform,
      this.gbuffer,
      palette,
      { ...options.tracer, paletteCount: options.materials.length },
    );
    this.resolve = new DeferredResolve(root, this.field, this.camera.uniform, this.light, palette, {
      ...options.shading,
      paletteCount: options.materials.length,
      presentFormat: this.presentFormat,
    });
  }

  /** Replaces the whole brush list and schedules a full rebuild on the next frame. */
  setBrushes(list: readonly BrushDesc[] | readonly BrushValue[]): void {
    const compiled = (list as readonly BrushDesc[]).every((b) => typeof (b as BrushDesc).kind === 'string')
      ? compileBrushes(list as readonly BrushDesc[])
      : (list as readonly BrushValue[]);
    this.builder.setBrushes(compiled);
    this.rebuildQueued = true;
  }

  /** Queues an incremental edit - sculpting, explosions, fluid erosion. */
  edit(desc: BrushDesc): void {
    this.editor.push(desc);
  }

  /** Matches the drawing buffer to the CSS size. Returns true when it changed. */
  syncSize(): boolean {
    const ratio = Math.min(window.devicePixelRatio || 1, this.maxPixelRatio);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    const changed = this.canvas.width !== w || this.canvas.height !== h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.camera.resize(w, h);
    return this.gbuffer.resize(w, h) || changed;
  }

  /** One full frame: field maintenance, simulation hook, trace, shade, present. */
  render(): void {
    this.syncSize();
    this.camera.update();

    const encoder = this.root['~unstable'].createCommandEncoder();
    const targets = this.gbuffer.current;

    if (this.rebuildQueued || this.editor.pendingCount > 0) {
      const pass = encoder.beginComputePass();
      if (this.rebuildQueued) {
        this.builder.rebuild(pass);
        this.rebuildQueued = false;
      }
      this.editor.flush(pass);
      pass.end();
    }

    this.onSimulate?.(encoder);
    this.raymarcher.prepass(encoder);

    const gpass = encoder.beginRenderPass({
      label: 'sdf-gbuffer',
      colorAttachments: [
        { view: targets.albedoView, clearValue: [0, 0, 0, 0] },
        { view: targets.normalView, clearValue: [0, 0, 0, -1] },
      ],
      depthStencilAttachment: {
        view: targets.depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    this.raymarcher.draw(gpass);
    this.onDrawGeometry?.(gpass);
    gpass.end();

    // Stochastic shadow + AO into their own target, so the resolve can filter them
    // spatially instead of relying on the temporal filter alone.
    const lpass = encoder.beginRenderPass({
      label: 'sdf-lighting',
      colorAttachments: [{ view: targets.lightView, clearValue: [1, 1, 0, 0] }],
    });
    this.resolve.drawLighting(lpass, targets.lightingInGroup);
    lpass.end();

    const rpass = encoder.beginRenderPass({
      label: 'sdf-resolve',
      colorAttachments: [
        { view: this.context },
        { view: this.gbuffer.historyTarget },
      ],
    });
    this.resolve.draw(rpass, this.gbuffer.readGroup, targets.lightReadGroup);
    rpass.end();

    encoder.submit();
    this.gbuffer.flip();
  }

  destroy(): void {
    this.gbuffer.destroy();
    this.editor.destroy();
    this.volume.destroy();
  }
}

