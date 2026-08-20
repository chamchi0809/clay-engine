import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { GBuffer, type RenderLayer } from '../render/gbuffer.ts';
import { SdfRaymarcher } from '../render/raymarch.ts';
import { DeferredResolve, type DeferredOptions } from '../render/deferred.ts';
import { TransparentComposite, type CompositeOptions } from '../render/composite.ts';
import {
  createPalette,
  normalizeMaterial,
  type MaterialPalette,
  type MaterialSpec,
  type MaterialValue,
} from '../trace/shade.ts';
import { analyticField, unionField, type TracedField } from '../trace/field.ts';
import { BrushSet, defaultBrushSet, type CustomBrush } from '../field/brush.ts';
import { BrushAtlas, type BrushAtlasOptions } from '../field/atlas.ts';
import { MeshBaker, type MeshBakerOptions } from '../field/meshbake.ts';
import { normalizeMesh, parseObj, type BakedMesh, type MeshData } from '../shape/mesh.ts';
import type { TracerOptions } from '../trace/march.ts';
import { GameCamera, type CameraSpawnOptions } from './camera.ts';
import { Fluid, type FluidSpawnOptions } from './fluid.ts';
import { Solid, type SolidSpawnOptions } from './solid.ts';
import { SoftBody, type SoftBodySpawnOptions } from './softbody.ts';
import { probeField } from './probe.ts';
import { Sun, type SunSpawnOptions } from './sun.ts';
import type { Entity, EntityContext } from './entity.ts';

export interface GameOptions {
  canvas: HTMLCanvasElement;
  /**
   * The material palette, by name. Shapes refer to these names, so a level never
   * hardcodes a palette index.
   *
   * `albedo` is the only required field. An entry with `opacity` below 1 is see-through,
   * and any object spawned with a single such material draws itself that way without
   * being told - see {@link Entity.transparent}.
   */
  materials: Record<string, MaterialSpec>;
  /**
   * Primitives of your own, by the name `sdf.custom(name)` refers to them by. Each is a
   * distance function plus a conservative bound - see {@link CustomBrush}.
   *
   * Declared here, alongside the materials, and for the same reason: both are compiled
   * into shader code when the game boots. A primitive added later could not be baked into
   * a field whose pipeline already exists, so this is the one and only place to add one.
   */
  brushes?: Record<string, CustomBrush>;
  /**
   * Enables {@link Game.loadMesh}. Present-means-on: `meshes: {}` is enough, and the
   * defaults hold sixteen shapes at 48 voxels each.
   *
   * Off by default because the atlas is a fixed-size 3D texture bound into every pipeline
   * that bakes or edits a field - a few megabytes and one bind group that a game with no
   * baked meshes should not pay for. Like {@link brushes}, it cannot be turned on later:
   * sampling it is compiled into the brush fold.
   */
  meshes?: BrushAtlasOptions & MeshBakerOptions;
  /**
   * The play area. Sets the default extent of a solid's volume and of a fluid's bake,
   * both of which are fixed-size 3D textures and so cannot simply grow.
   */
  bounds?: { size?: number; origin?: readonly [number, number, number] };
  /** Device pixel ratio cap. 1 is the right default for a raymarcher. */
  pixelRatio?: number;
  tracer?: TracerOptions;
  /** Lighting and filter knobs. Everything here has a working default. */
  shading?: Omit<DeferredOptions, 'paletteCount' | 'presentFormat' | 'transparent'>;
  /**
   * Transparency knobs. Only consulted once something in the scene is see-through; sky,
   * ambient and exposure are taken from {@link shading} so the two passes agree.
   */
  transparency?: Omit<CompositeOptions, 'paletteCount' | 'presentFormat' | 'sky' | 'ambient' | 'exposure'>;
}

export interface LoadMeshOptions {
  /**
   * How much of the bake box the shape fills, on its widest axis. Under 1 by necessity - the
   * field just inside the box wall has to be positive or there is no exterior for a ray to
   * approach the surface through - and the default 0.9 spends 10% of the resolution on that.
   *
   * Lower it for a shape that other brushes have to blend smoothly into from far away, since
   * only the field inside the box is real; outside it, the brush reports the distance to the
   * box instead.
   */
  fit?: number;
}

/**
 * A game.
 *
 * Everything in it is a spawned object - the level, the camera, the sun, every body,
 * every drop of liquid. Nothing exists implicitly, and `game.spawn.*` is nothing but
 * sugar over `new Solid(game, opts)`, so adding a new *kind* of object needs no
 * registration: implement {@link Entity}, call `game.attach(this)`, done.
 *
 * The frame is fixed and the game does not drive it: {@link start} runs the loop and
 * calls back once per frame with the elapsed time, which is where input, camera
 * placement and game rules go. Everything GPU-side - the compute passes, the trace, the
 * shading, the temporal filter - is the engine's problem.
 */
export class Game {
  readonly root: TgpuRoot;
  readonly canvas: HTMLCanvasElement;
  readonly bounds: { origin: readonly [number, number, number]; size: number };
  /**
   * The primitives every field in this game is built out of: the builtins plus whatever
   * {@link GameOptions.brushes} declared. Fixed for the game's lifetime.
   */
  readonly brushSet: BrushSet;
  /** The baked-mesh atlas, or null unless {@link GameOptions.meshes} asked for one. */
  readonly meshAtlas: BrushAtlas | null;
  /** Seconds since {@link start}. */
  time = 0;

  private readonly context: GPUCanvasContext;
  private readonly presentFormat: GPUTextureFormat;
  private readonly gbuffer: GBuffer;
  private readonly palette: MaterialPalette;
  private readonly paletteCount: number;
  private readonly materialIds: Map<string, number>;
  private readonly materialValues: readonly MaterialValue[];
  private readonly options: GameOptions;
  private readonly pixelRatio: number;

  private readonly meshBaker: MeshBaker | null;
  private readonly members: Entity[] = [];
  private raymarcher: SdfRaymarcher | null = null;
  private transparentMarcher: SdfRaymarcher | null = null;
  private resolve: DeferredResolve | null = null;
  private composite: TransparentComposite | null = null;
  private dirty = true;
  private running = false;
  private raf = 0;
  private last = 0;

  static async create(options: GameOptions): Promise<Game> {
    const root = await tgpu.init();
    return new Game(root, options);
  }

  constructor(root: TgpuRoot, options: GameOptions) {
    this.root = root;
    this.options = options;
    this.canvas = options.canvas;
    this.pixelRatio = options.pixelRatio ?? 1;
    this.bounds = {
      size: options.bounds?.size ?? 24,
      origin: options.bounds?.origin ?? [-12, -4, -12],
    };
    const context = options.canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Game: this canvas has no WebGPU context. Is WebGPU enabled?');
    }
    this.context = context;
    this.presentFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: root.device, format: this.presentFormat, alphaMode: 'opaque' });

    // Before the brush set, because the set compiles the atlas sample into its fold.
    this.meshAtlas = options.meshes ? new BrushAtlas(root, options.meshes) : null;
    this.meshBaker = this.meshAtlas
      ? new MeshBaker(root, this.meshAtlas, options.meshes ?? {})
      : null;

    const custom = options.brushes ?? {};
    // Reuse the shared set when there is nothing to add, so a game that declares no
    // primitives of its own resolves to the exact same WGSL every other one does.
    this.brushSet = Object.keys(custom).length > 0 || this.meshAtlas
      ? new BrushSet({ custom, atlas: this.meshAtlas })
      : defaultBrushSet;

    const names = Object.keys(options.materials);
    this.materialIds = new Map(names.map((n, i) => [n, i]));
    this.paletteCount = names.length;
    this.materialValues = names.map((n) => normalizeMaterial(options.materials[n]!));
    this.palette = createPalette(root, names.map((n) => options.materials[n]!));
    this.gbuffer = new GBuffer(root);
  }

  /** Sugar. Every one of these is `new X(game, opts)`; use that for your own types. */
  readonly spawn = {
    solid: (o: SolidSpawnOptions): Solid => new Solid(this, o),
    softBody: (o: SoftBodySpawnOptions): SoftBody => new SoftBody(this, o),
    fluid: (o: FluidSpawnOptions = {}): Fluid => new Fluid(this, o),
    camera: (o: CameraSpawnOptions = {}): GameCamera => new GameCamera(this, o),
    sun: (o: SunSpawnOptions = {}): Sun => new Sun(this, o),
  };

  /**
   * Bakes a triangle mesh into a distance field and hands back a brush.
   *
   * `source` is either positions plus indices - straight off a glTF loader or a three.js
   * `BufferGeometry` - or the text of an OBJ file. The mesh does not have to be watertight,
   * consistently wound, or free of self-intersections: the sign comes from a generalised
   * winding number, which is what makes an art asset usable rather than only a CAD solid.
   *
   * The result is an ordinary primitive from there:
   *
   * ```ts
   * const rock = await game.loadMesh(rockObj);
   * level.shape = sdf.union(ground, sdf.mesh(rock).at([3, 0, 0]).material('stone'));
   * level.cut(sdf.mesh(rock).scale(0.3).at(hit).only('clay'));
   * ```
   *
   * Awaited, because the bake has to have landed before a field is built out of it - a brush
   * pointing at a slot nobody has written samples whatever the texture was cleared to.
   */
  async loadMesh(source: MeshData | string, options: LoadMeshOptions = {}): Promise<BakedMesh> {
    if (!this.meshAtlas || !this.meshBaker) {
      throw new Error(
        'Game: baked meshes are off. Pass `meshes: {}` to `Game.create`. It cannot be turned '
          + 'on now, because sampling the atlas is compiled into every field pipeline.',
      );
    }
    const data = typeof source === 'string' ? parseObj(source) : source;
    const normalized = normalizeMesh(data, options.fit);
    const slot = this.meshAtlas.allocate();
    await this.meshBaker.bake(slot, normalized);
    return {
      slot,
      half: normalized.half,
      center: normalized.center,
      triangleCount: normalized.triangleCount,
    };
  }

  /** Everything spawned, in spawn order. */
  get entities(): readonly Entity[] {
    return this.members;
  }

  /**
   * Registers an entity. Called from an entity's constructor; a game never calls this.
   * Marks the render pipelines for rebuild, because they bake the traced field in.
   */
  attach(entity: Entity): void {
    this.members.push(entity);
    this.dirty = true;
  }

  /** Removes and destroys an entity. The pipelines rebuild on the next frame. */
  despawn(entity: Entity): void {
    const i = this.members.indexOf(entity);
    if (i < 0) {
      return;
    }
    this.members.splice(i, 1);
    entity.destroy?.();
    this.dirty = true;
  }

  /** Palette index for a material name. Numbers pass through, `undefined` is slot 0. */
  material(name: string | number | undefined): number {
    if (typeof name === 'number') {
      return name;
    }
    if (name === undefined) {
      return 0;
    }
    const id = this.materialIds.get(name);
    if (id === undefined) {
      throw new Error(
        `Game: no material named "${name}". Declared: ${[...this.materialIds.keys()].join(', ')}`,
      );
    }
    return id;
  }

  /**
   * Opacity of a material, `0..1`. What an entity asks so it can default its own
   * `transparent` flag from the material it was spawned with, instead of a game having to
   * say "this is water" twice.
   */
  materialOpacity(name: string | number | undefined): number {
    return this.materialValues[this.material(name)]?.opacity ?? 1;
  }

  /**
   * Everything solid enough to hit, as one field. `exclude` drops one entity, which is
   * how a simulated body avoids colliding with its own baked surface.
   */
  /**
   * Samples what a collider would see along a line. Diagnostic only - see
   * {@link probeField}.
   */
  probe(
    from: readonly [number, number, number],
    to: readonly [number, number, number],
    count = 64,
  ) {
    return probeField(this.root, this.colliderField(), from, to, count);
  }

  colliderField(exclude?: Entity): TracedField {
    return this.merge(this.members.filter((e) => e !== exclude && e.collidable !== false));
  }

  /** Everything the renderer traces this frame, both layers. */
  get field(): TracedField {
    return this.merge(this.tracedMembers());
  }

  /**
   * The traced field of one layer. The split is what makes transparency possible at all:
   * the opaque layer is what the deferred resolve shades and what every shadow, AO and
   * refraction lookup sees, and the transparent layer is traced separately *after* it, so
   * a see-through surface can be composited over an image that already exists.
   */
  layerField(layer: RenderLayer): TracedField {
    const want = layer === 'transparent';
    return this.merge(this.tracedMembers().filter((e) => (e.transparent === true) === want));
  }

  /** True when anything at all is drawn into the transparency layer. */
  private get hasTransparent(): boolean {
    return this.members.some((e) => e.transparent === true);
  }

  private tracedMembers(): Entity[] {
    return this.members.filter((e) => e.traced !== false);
  }

  private merge(list: readonly Entity[]): TracedField {
    const fields = list.map((e) => e.field).filter((f): f is TracedField => !!f);
    return fields.length > 0 ? fields.reduce(unionField) : EMPTY_FIELD();
  }

  /** The active camera, or null before one is spawned. */
  get camera(): GameCamera | null {
    return this.members.find((e): e is GameCamera => e instanceof GameCamera) ?? null;
  }

  /** The active sun, or null before one is spawned. */
  get sun(): Sun | null {
    return this.members.find((e): e is Sun => e instanceof Sun) ?? null;
  }

  /**
   * Starts the frame loop. `onFrame` runs before the frame is drawn, which is where
   * input, camera placement and game logic belong.
   */
  start(onFrame: (dt: number) => void): void {
    if (this.running) {
      throw new Error('Game: already running');
    }
    this.running = true;
    const loop = (now: number) => {
      if (!this.running) {
        return;
      }
      const dt = this.last > 0 ? Math.min(0.1, (now - this.last) / 1000) : 0;
      this.last = now;
      this.time += dt;
      onFrame(dt);
      this.render();
      for (const e of this.members) {
        e.sync?.(dt);
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  /** One frame. `start` calls this; call it yourself if you own the loop. */
  render(): void {
    const camera = this.camera;
    if (!camera) {
      throw new Error('Game: no camera. `game.spawn.camera()` before rendering.');
    }
    this.resize();
    camera.camera.update();
    if (this.dirty) {
      this.rebuild();
    }
    const targets = this.gbuffer.current;
    const encoder = this.root['~unstable'].createCommandEncoder();

    const cpass = encoder.beginComputePass();
    for (const e of this.members) {
      e.simulate?.(cpass);
    }
    cpass.end();

    this.raymarcher!.prepass(encoder);
    this.transparentMarcher?.prepass(encoder);

    const gpass = encoder.beginRenderPass({
      label: 'game-gbuffer',
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
    this.raymarcher!.draw(gpass);
    for (const e of this.members) {
      if (e.transparent !== true) {
        e.drawGeometry?.(gpass);
      }
    }
    gpass.end();

    // The transparency layer: the same two attachments one surface deeper, sharing the
    // opaque layer's depth buffer. Loading that depth rather than clearing it is what
    // culls every see-through surface hidden behind a wall, for free and before shading.
    if (this.composite) {
      const tpass = encoder.beginRenderPass({
        label: 'game-gbuffer-transparent',
        colorAttachments: [
          { view: targets.albedoViewT, clearValue: [0, 0, 0, 0] },
          { view: targets.normalViewT, clearValue: [0, 0, 0, -1] },
        ],
        depthStencilAttachment: {
          view: targets.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      });
      this.transparentMarcher?.draw(tpass);
      for (const e of this.members) {
        if (e.transparent === true) {
          e.drawGeometry?.(tpass);
        }
      }
      tpass.end();
    }

    // Stochastic shadow + AO into their own target, so the resolve can filter them
    // spatially instead of leaning on the temporal filter alone.
    const lpass = encoder.beginRenderPass({
      label: 'game-lighting',
      colorAttachments: [{ view: targets.lightView, clearValue: [1, 1, 0, 0] }],
    });
    this.resolve!.drawLighting(lpass, targets.lightingInGroup);
    lpass.end();

    const rpass = encoder.beginRenderPass({
      label: 'game-resolve',
      colorAttachments: [{ view: this.context }, { view: this.gbuffer.historyTarget }],
    });
    this.resolve!.draw(rpass, this.gbuffer.readGroup, targets.lightReadGroup);
    rpass.end();

    // Transparency last, over the presented image, refracting the linear copy of it the
    // resolve just wrote. `resolvedGroup` is this frame's history slot, not last frame's.
    if (this.composite) {
      const xpass = encoder.beginRenderPass({
        label: 'game-transparency',
        colorAttachments: [{ view: this.context, loadOp: 'load' }],
      });
      this.composite.draw(
        xpass,
        this.gbuffer.resolvedGroup,
        targets.lightReadGroup,
        targets.transparentReadGroup,
      );
      xpass.end();
    }

    encoder.submit();
    this.gbuffer.flip();
  }

  destroy(): void {
    this.stop();
    for (const e of [...this.members]) {
      e.destroy?.();
    }
    this.members.length = 0;
    this.gbuffer.destroy();
    this.meshAtlas?.destroy();
  }

  /**
   * Recreates everything that bakes the traced field into shader code, then lets every
   * entity build its own pipelines.
   *
   * A render pipeline is compiled against one specific field, so spawning or despawning
   * anything traceable invalidates it. Rebuilding lazily on the first frame after the
   * change is what makes both spawning-after-boot and `despawn` possible at all, and it
   * means a game can build its level in whatever order reads best.
   */
  private rebuild(): void {
    this.dirty = false;
    const camera = this.camera!;
    const sun = this.sun;
    if (!sun) {
      throw new Error('Game: no sun. `game.spawn.sun()` before rendering.');
    }
    const ctx: EntityContext = {
      root: this.root,
      colliders: (exclude) => this.colliderField(exclude),
      scene: this.field,
      camera: camera.camera,
      palette: this.palette,
      paletteCount: this.paletteCount,
      bounds: this.bounds,
      material: (n) => this.material(n),
      materialOpacity: (n) => this.materialOpacity(n),
    };
    for (const e of this.members) {
      e.build?.(ctx);
    }
    // After `build`, because an entity may only have a field once it is built.
    const transparent = this.hasTransparent;
    // Only the opaque layer is shaded, shadowed and AO'd. That is also what a transparent
    // surface refracts and reflects, so keeping it out of its own backdrop is deliberate
    // rather than a shortcut: clear water that casts a solid shadow looks worse than clear
    // water that casts none.
    const field = transparent ? this.layerField('opaque') : this.field;
    this.raymarcher = new SdfRaymarcher(this.root, field, camera.camera.uniform, this.gbuffer, this.palette, {
      ...this.options.tracer,
      paletteCount: this.paletteCount,
    });
    this.resolve = new DeferredResolve(this.root, field, camera.camera.uniform, sun.uniform, this.palette, {
      ...this.options.shading,
      transparent,
      paletteCount: this.paletteCount,
      presentFormat: this.presentFormat,
    });

    // Both null unless something asked to be see-through, so a scene without any
    // transparency runs exactly the passes it used to.
    this.transparentMarcher = null;
    this.composite = null;
    if (transparent) {
      const tracedTransparent = this.tracedMembers().some((e) => e.transparent === true);
      if (tracedTransparent) {
        this.transparentMarcher = new SdfRaymarcher(
          this.root,
          this.layerField('transparent'),
          camera.camera.uniform,
          this.gbuffer,
          this.palette,
          { ...this.options.tracer, paletteCount: this.paletteCount, layer: 'transparent' },
        );
      }
      this.composite = new TransparentComposite(
        this.root,
        camera.camera.uniform,
        sun.uniform,
        this.palette,
        {
          ...this.options.transparency,
          // Shared with the resolve, because a refracted pixel is compared against the
          // pixels around it: a different sky, ambient or exposure here would draw a seam
          // along every silhouette.
          sky: this.options.shading?.sky,
          ambient: this.options.shading?.ambient,
          exposure: this.options.shading?.exposure,
          paletteCount: this.paletteCount,
          presentFormat: this.presentFormat,
        },
      );
    }
  }

  private resize(): void {
    const ratio = Math.min(window.devicePixelRatio || 1, this.pixelRatio);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    this.canvas.width = w;
    this.canvas.height = h;
    this.camera!.camera.resize(w, h);
    this.gbuffer.resize(w, h);
  }
}

/** A field that is nowhere. Lets a game render before it has spawned any geometry. */
const EMPTY_FIELD = (): TracedField =>
  analyticField(() => {
    'use gpu';
    return 1e5;
  }, { band: 1e5, epsilon: 0.01, material: 0 });
