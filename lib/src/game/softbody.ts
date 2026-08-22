import { ParticleMesh } from '../sim/meshdraw.ts';
import { ParticleSet, type BodyTracker } from '../sim/particles.ts';
import { ClaySolver } from '../sim/pbd.ts';
import { SplatField, bodyCloud } from '../sim/splat.ts';
import { ExtractMotion, SurfaceExtractor } from '../sim/extract.ts';
import { SdfBuilder } from '../field/builder.ts';
import { SdfVolume } from '../field/volume.ts';
import { compileShape, shapeBounds, type Shape } from '../shape/sdf.ts';
import { d } from '../gpu.ts';
import { lerpField, offsetField, volumeField, type TracedField } from '../trace/field.ts';
import { GameObject, type EntityContext, type ForceMode } from './entity.ts';
import type { TgpuComputePass, TgpuMutable, TgpuRenderCommands, TgpuUniform } from 'typegpu';
import type { Game } from './game.ts';

export interface SoftBodySpawnOptions {
  /** The rest shape. Its bounds size the extraction box, the collider and the bake. */
  shape: Shape;
  position?: readonly [number, number, number];
  material?: string | number;
  /**
   * How hard the body snaps back to its rest shape, `0..1`, applied per substep.
   * Measured squash from a 2.4-unit drop: 0.85 -> 1.5% (reads as rigid), 0.3 -> 8%,
   * 0.15 -> 16%, 0.05 -> 25% (jelly). Rigid bodies want 1.
   *
   * Below about 0.2 a body resting under gravity visibly sags, because shape matching is
   * the only thing holding it up.
   */
  stiffness?: number;
  /**
   * Fraction of a dent that becomes permanent per *second*, `0..1`. 0 is rubber - it
   * always springs back. Anything above 0 is clay: dents stay.
   *
   * Per second, not per substep, because the substep rate is an implementation detail:
   * at 180 substeps a second, a per-substep 0.05 forgets the rest shape inside two
   * frames and the body collapses into a bowl under its own weight.
   */
  plasticity?: number;
  /**
   * World-space distance between extracted particles. This resolves *features*, not
   * extent, so it has to suit the thinnest part of the thinnest shape the body will ever
   * be: a rod of radius 0.48 sampled every 0.17 is three particles across and looks it.
   * Defaults to the spawn shape's radius / 8.5.
   */
  spacing?: number;
  /**
   * World half-extent the body may ever occupy, over every shape it can morph into. The
   * extraction box is fixed at construction, so it cannot be derived from the spawn
   * shape alone - a sphere that morphs into a long rod needs the rod's reach. Defaults
   * to 1.4x the spawn shape's radius; {@link morph} throws if a shape exceeds it.
   */
  reach?: number;
  /** Voxels per axis of this body's own collider bake. Detail, not extent. */
  bakeResolution?: number;
  gravity?: number;
  /** Tangential motion removed on contact, `0..1`. */
  friction?: number;
  /**
   * Draw the body see-through. Defaults to whether its material has an `opacity` below 1.
   *
   * The body rasterises rather than traces, and transparency does not care: it moves the
   * one indirect draw into the transparency layer's pass, against the same depth buffer,
   * and the composite reads the same two attachments either way.
   */
  transparent?: boolean;
  label?: string;
}

/** Seconds a {@link SoftBody.morph} takes. */
const MORPH_TIME = 0.22;
const FRAME = 1 / 60;
/** Solver substeps per frame. */
const SUBSTEPS = 3;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/**
 * A deformable body: a cloud of particles that remembers a rest shape, pulled back
 * toward it by shape matching (Muller et al. 2005, and Claybook GDC'18 slides 49-51).
 *
 * The particles *are* the mesh vertices, so deformation is free on the render side and
 * the body rasterises as triangles in one indirect draw - which is what Claybook shipped
 * and why it could afford thousands of them. Its own signed distance field is baked
 * separately, at collider resolution, because that is all anything else needs it for.
 *
 * Two knobs cover the whole material range: `stiffness` decides whether it is rigid or
 * floppy, `plasticity` whether dents heal. Clay is soft and plastic, rubber soft and
 * elastic, a stone rigid and elastic.
 */
export class SoftBody extends GameObject {
  readonly set: ParticleSet;
  readonly body: number;
  /** Conservative radius of the rest shape. Sizes everything else. */
  readonly radius: number;
  /**
   * This body as a distance field, re-baked from its particles every frame - a collider
   * and an occluder, not a picture. {@link traced} is false because the body draws itself
   * as triangles.
   */
  readonly field: TracedField;
  /**
   * The particles *are* the mesh vertices, so the body rasterises in one indirect draw
   * and deformation costs nothing on the render side. Tracing it instead would need a
   * render-resolution bake every frame plus an extra texture sample on every primary ray
   * in the scene - which is why Claybook rasterised its clay too (GDC'18 slides 45-47)
   * and only ray-traced the fluid.
   */
  readonly traced = false;
  /**
   * The bake is a collider anyway, so shadows and AO come almost free - and without them
   * a body is lit as if nothing else in the scene existed: no shadow on the floor, no
   * darkening where it meets a wall, no occlusion of its own concavities. Cheap because
   * shadow and AO rays are the two that were already allowed to be coarse (slide 35),
   * which is exactly what a collider-resolution bake is.
   */
  readonly occluder = true;
  /**
   * Decided at construction, because it selects which G-buffer pass this body's draw is
   * recorded into and the game reads it before `build` runs.
   */
  readonly transparent: boolean;

  private readonly tracker: BodyTracker;
  private readonly box: number;
  private readonly morphVolumes: [SdfVolume, SdfVolume];
  private readonly morphBuilders: [SdfBuilder, SdfBuilder];
  private readonly blend: TgpuUniform<d.F32>;
  /** GPU-side translation and linear velocity of the body-local morph volumes. */
  private readonly motion: TgpuMutable<typeof ExtractMotion>;
  private readonly surface: SplatField;
  private readonly solverOptions: { gravity: number; friction: number };
  private readonly resolution: number;
  private readonly reach: number;
  private readonly bakeRes: number;
  private readonly materialId: number;

  private extractor!: SurfaceExtractor;
  private solver!: ClaySolver;
  private mesh!: ParticleMesh;

  private restShape: Shape;
  /** Which of the ping-pong pair holds the *current* shape. */
  private live = 0;
  private morphT = 1;
  /** Which of the morph volumes still needs its shape baked. */
  private bakeDirty: [boolean, boolean] = [true, true];
  private extractQueued = true;
  /** False for a spawn/teleport, true when re-extracting the live particle cloud. */
  private preserveMotion = false;
  /** Accumulated this frame, cleared after the step. */
  private accel: [number, number, number] = [0, 0, 0];
  private impulse: [number, number, number] = [0, 0, 0];

  constructor(game: Game, options: SoftBodySpawnOptions) {
    super(game);
    const root = game.root;
    const bounds = shapeBounds(options.shape, game.brushSet);
    this.restShape = options.shape;
    this.radius = bounds.radius;
    this.reach = options.reach ?? bounds.radius * 1.4;
    const spacing = options.spacing ?? bounds.radius / 8.5;
    this.bakeRes = options.bakeResolution ?? 32;
    this.materialId = game.material(options.material);
    this.transparent = options.transparent ?? game.materialOpacity(options.material) < 1;
    // A cell of margin on every side, or surface nets clips the body flat where it
    // touches the boundary. Cost is O(res^3) in the extraction dispatch, so the cell
    // count is capped rather than letting a fine spacing over a wide reach explode.
    this.box = this.reach * 2 + spacing * 2;
    this.resolution = Math.min(64, Math.max(8, Math.ceil(this.box / spacing)));
    this.solverOptions = {
      gravity: options.gravity ?? -18,
      friction: options.friction ?? 0.45,
    };

    // The morph pair. A shape is baked into a body-local volume rather than compiled
    // into the shader, which is what lets `morph()` take an arbitrary runtime shape
    // instead of one from a list declared up front.
    const label = options.label ?? 'softBody';
    const volumeOptions = {
      // Half a particle spacing per voxel, rounded up to a power of two. The mesh takes
      // its normals from this volume's gradient, so a voxel the size of the particle
      // spacing shows up as visible lattice-aligned banding across the body.
      resolution: Math.min(64, 1 << Math.ceil(Math.log2((this.box / spacing) * 2))),
      worldSize: this.box,
      origin: [-this.box / 2, -this.box / 2, -this.box / 2] as [number, number, number],
      band: 4,
      // The extracted mesh shades with this volume's gradient, so the sawtooth matters
      // here in a way it never does for world geometry.
      gradientVoxels: 2,
    };
    this.morphVolumes = [
      new SdfVolume(root, { ...volumeOptions, label: `${label}ShapeA` }),
      new SdfVolume(root, { ...volumeOptions, label: `${label}ShapeB` }),
    ];
    this.morphBuilders = [
      new SdfBuilder(this.morphVolumes[0], { brushSet: game.brushSet }),
      new SdfBuilder(this.morphVolumes[1], { brushSet: game.brushSet }),
    ];
    this.blend = root.createUniform(d.f32, 0);
    const spawn = options.position ?? [0, 0, 0];
    this.motion = root.createMutable(ExtractMotion, {
      center: [spawn[0], spawn[1], spawn[2]],
      _padA: 0,
      velocity: [0, 0, 0],
      _padB: 0,
    });
    const blend = this.blend;
    const motion = this.motion;
    // The shape volumes are baked about the origin once and then *moved* by GPU-side
    // state, rather than rebaked wherever the body happens to be.
    const morphField = offsetField(
      lerpField(
        volumeField(this.morphVolumes[0]),
        volumeField(this.morphVolumes[1]),
        () => {
          'use gpu';
          return blend.$;
        },
      ),
      () => {
        'use gpu';
        // A copy, not the reference: TypeGPU cannot return a storage reference.
        return d.vec3f(motion.$.center);
      },
    );
    this.setShapeBrushes(0, this.restShape);
    this.setShapeBrushes(1, this.restShape);

    this.set = new ParticleSet(root, { capacity: 4096, maxBodies: 1, label });
    this.body = this.set.addBody({
      stiffness: options.stiffness ?? 0.55,
      // Per-second -> per-substep. `(1 - p)^h` is the exact conversion for a per-step
      // exponential decay, so the same number means the same clay at any substep rate.
      plasticity: 1 - (1 - clamp01(options.plasticity ?? 0.35)) ** (FRAME / SUBSTEPS),
      material: this.materialId,
    });
    this.tracker = this.set.track(this.body);
    if (options.position) {
      this.tracker.reset(options.position);
    }

    // Three load-bearing constraints on the bake, all of them geometric:
    //  - splat radius > half the worst gap between extracted particles, or the shell
    //    has holes (surface nets emits one particle per straddling cell, so the worst
    //    case is a body-diagonal step);
    //  - voxel < splat radius, or the splats never cross zero at a voxel centre and the
    //    body simply is not there;
    //  - band * voxel > the radius any collider tests with, or the saturated reading
    //    outside the band *is* a false contact. See `SplatField`.
    // The radius used to be a fourth constraint pulling the other way, because it also
    // set how far outside the particles the shell landed - and the particles are the mesh
    // vertices, so the occluder outgrew the thing being drawn. `bodyCloud` supplies
    // normals, so the shell now passes through the particles and the radius only has to
    // satisfy the three above.
    const cell = this.box / this.resolution;
    this.surface = new SplatField(root, bodyCloud(this.set, this.body), {
      radius: cell * 1.15,
      material: this.materialId,
      resolution: this.bakeRes,
      band: 4,
      worldSize: this.box,
      label: `${label}Bake`,
    });
    this.field = this.surface.field;
    this.morphSource = morphField;
  }

  private readonly morphSource: TracedField;

  build(ctx: EntityContext): void {
    // ponytail: built once. Rebuilding would reallocate the shell buffers and drop the
    // particles mid-flight; the cost is that a solid spawned later is not collided
    // against until the body is respawned.
    if (this.extractor) {
      return;
    }
    this.extractor = new SurfaceExtractor(this.set, this.morphSource, {
      resolution: this.resolution,
      body: this.body,
      base: 0,
      capacity: 3000,
      motion: this.motion,
      label: 'softBodyShell',
    });
    // Everything hittable except this body itself - a body that collides with its own
    // baked surface locks solid instantly.
    this.solver = new ClaySolver(this.set, ctx.colliders(this), {
      gravity: [0, this.solverOptions.gravity, 0],
      dt: FRAME,
      substeps: SUBSTEPS,
      // Roughly half the particle spacing, so the shell rests on the surface instead of
      // sinking into it.
      radius: (this.box / this.resolution) * 0.6,
      friction: this.solverOptions.friction,
      // Air drag, per substep. 0.992 is 0.24 of the velocity surviving a second, which is
      // enough to eat two thirds of a jump; 0.997 is 0.58 and still damps the modes the
      // predictor needs damped.
      damping: 0.997,
      maxSpeed: 26,
    });
    this.mesh = new ParticleMesh(ctx.root, this.extractor, ctx.camera.uniform, ctx.palette, {
      paletteCount: ctx.paletteCount,
      material: this.materialId,
    });
  }

  /** Centre of mass. A GPU readback, so it lands a few frames late and steps. */
  get position(): readonly [number, number, number] {
    return this.tracker.raw;
  }
  /**
   * Orientation shape matching fitted to the particle cloud, xyzw unit quaternion.
   *
   * This is what lets a game stamp the body's own shape into the world - a footprint, a
   * dent, the groove something leaves behind - instead of approximating it with a sphere:
   * `world.cut(shape.turn(body.rotation).at(body.position))`.
   */
  get rotation(): readonly [number, number, number, number] {
    return this.tracker.rot;
  }
  /**
   * Centre of mass, exponentially smoothed and advanced every frame. Point a camera at
   * this; pointing one at {@link position} makes it shake at the readback rate.
   */
  get smoothPosition(): readonly [number, number, number] {
    return this.tracker.pos;
  }
  /** Metres per second, differenced over the real interval between readbacks. */
  get velocity(): readonly [number, number, number] {
    return this.tracker.vel;
  }
  get morphing(): boolean {
    return this.morphT < 1;
  }
  get shape(): Shape {
    return this.restShape;
  }

  /**
   * Adds a force for this frame. Accumulates, and clears after the step - so a force
   * that stops being applied stops acting, which is what every physics engine does and
   * the opposite of what a persistent uniform would do.
   */
  addForce(v: readonly [number, number, number], mode: ForceMode = 'acceleration'): void {
    const target = mode === 'velocity' ? this.impulse : this.accel;
    target[0] += v[0];
    target[1] += v[1];
    target[2] += v[2];
  }

  /**
   * Hard-sets the velocity of the whole body, as the impulse that gets it there. It
   * reads {@link velocity}, which is a readback and therefore a few frames stale, so
   * this lands close rather than exact - fine for a jump, wrong for a rail.
   */
  setVelocity(v: readonly [number, number, number]): void {
    const cur = this.velocity;
    this.addForce([v[0] - cur[0], v[1] - cur[1], v[2] - cur[2]], 'velocity');
  }

  /** Moves the body without simulating the trip, and rebuilds it from its rest shape. */
  setPosition(p: readonly [number, number, number]): void {
    this.tracker.reset(p);
    this.motion.write({
      center: [p[0], p[1], p[2]],
      _padA: 0,
      velocity: [0, 0, 0],
      _padB: 0,
    });
    // Finish any morph in flight *including its blend*: leaving `blend` where the
    // animation stopped leaves the body stuck at whatever shape it was passing through.
    this.morphT = 1;
    this.writeBlend(1);
    this.extractQueued = true;
    this.preserveMotion = false;
  }

  /**
   * Reshapes the body over {@link MORPH_TIME} seconds. The argument is an ordinary
   * shape expression built at runtime - nothing is declared in advance, because the
   * shape is baked into a volume rather than compiled into a shader.
   */
  morph(shape: Shape): void {
    const reach = shapeBounds(shape, this.game.brushSet).radius;
    if (reach > this.reach) {
      throw new Error(
        `SoftBody.morph: shape reaches ${reach.toFixed(2)} but the body was built for `
          + `${this.reach.toFixed(2)}. Pass a larger \`reach\` when spawning it.`,
      );
    }
    const next = this.live ^ 1;
    this.setShapeBrushes(next, shape);
    this.bakeDirty[next] = true;
    this.live = next;
    this.restShape = shape;
    this.morphT = 0;
  }

  simulate(pass: TgpuComputePass): void {
    for (let i = 0; i < 2; i++) {
      if (this.bakeDirty[i]) {
        this.morphBuilders[i]!.rebuild(pass);
        this.bakeDirty[i] = false;
      }
    }
    if (this.morphT < 1) {
      this.morphT = Math.min(1, this.morphT + FRAME / MORPH_TIME);
      this.extractQueued = true;
      this.writeBlend(this.morphT);
    }
    if (this.extractQueued) {
      const half = this.box * 0.5;
      this.extractor.setRegion(
        [-half, -half, -half],
        this.box,
        [0, 0, 0],
        this.preserveMotion,
      );
      this.extractor.extract(pass);
      this.extractQueued = false;
      this.preserveMotion = true;
    }
    this.solver.setForce(this.body, this.accel, this.impulse);
    this.accel = [0, 0, 0];
    this.impulse = [0, 0, 0];
    this.solver.step(pass);
    // The bake follows the body, so 32^3 over a body-sized box is a fine voxel for a
    // fraction of the memory a world-sized volume at the same detail would need.
    this.surface.setCenter(this.position);
    this.surface.bake(pass);
  }

  drawGeometry(pass: TgpuRenderCommands): void {
    this.mesh.draw(pass);
  }

  sync(dt: number): void {
    this.tracker.sync(dt);
  }

  destroy(): void {
    this.morphVolumes[0].destroy();
    this.morphVolumes[1].destroy();
    this.surface.destroy();
  }

  /**
   * `t` is morph progress; the uniform is the mix weight of volume 1, so which end it
   * runs toward depends on which slot is live. Smoothstepped, so it does not jerk.
   */
  private writeBlend(t: number): void {
    const k = t * t * (3 - 2 * t);
    this.blend.write(this.live === 1 ? k : 1 - k);
  }

  /** The morph volumes are body-local, so the shape is baked about the origin. */
  private setShapeBrushes(slot: number, shape: Shape): void {
    const builder = this.morphBuilders[slot]!;
    builder.setBrushes(builder.brushSet.compile(compileShape(shape, (n) => this.game.material(n))));
  }
}
