import { FluidSim } from '../sim/fluid.ts';
import { GameObject, type EntityContext } from './entity.ts';
import type { TgpuComputePass } from 'typegpu';
import type { TracedField } from '../trace/field.ts';
import type { Game } from './game.ts';

export interface FluidSpawnOptions {
  material?: string | number;
  /** Particle capacity. Emission wraps, so a tap can run forever. */
  capacity?: number;
  /** Rest spacing between particles. Sets the mass, the rest density and the look. */
  spacing?: number;
  /**
   * Voxels per axis of the surface bake. Match the solid's, or the drops read as cubes
   * and the whole scene traces at the coarser field's step slack.
   */
  bakeResolution?: number;
  /** XSPH velocity blend per substep, `0..1`. Claybook's clay-like water wants ~0.4. */
  viscosity?: number;
  gravity?: readonly [number, number, number];
  /**
   * Draw the liquid see-through. Defaults to whether its material has an `opacity` below
   * 1, which is the whole configuration a game needs: declare water as water in the
   * palette and it renders as water.
   */
  transparent?: boolean;
  label?: string;
}

/** One place the fluid is grinding against something solid, with its impact speed. */
export interface FluidContact {
  position: [number, number, number];
  speed: number;
}

/**
 * Simulated liquid: weakly compressible SPH that collides against a distance field and
 * renders itself as one.
 *
 * Unlike a soft body it has no rest shape and no stable topology, which is exactly why
 * it is baked into a volume and ray-traced rather than meshed - there is no vertex to
 * keep. That is the split Claybook shipped too (GDC'18 slides 55-62): clay as triangles,
 * fluid as an SDF regenerated every frame.
 */
export class Fluid extends GameObject {
  /** A fluid is not something to stand on, and must not collide with its own bake. */
  readonly collidable = false;
  /**
   * Decided at construction, not at build: which pass a fluid is drawn in is baked into
   * the pipelines, and the flag has to be readable before `build` runs.
   */
  readonly transparent: boolean;

  private readonly options: FluidSpawnOptions;
  private sim: FluidSim | null = null;

  constructor(game: Game, options: FluidSpawnOptions = {}) {
    super(game);
    this.options = options;
    this.transparent = options.transparent ?? game.materialOpacity(options.material) < 1;
  }

  /**
   * Null until the game builds the fluid, because a fluid is the one thing that has to
   * know what else exists first: its collider is everything else in the scene, and a
   * field cannot be swapped into a pipeline after the fact.
   */
  get field(): TracedField | null {
    return this.sim?.field ?? null;
  }

  get count(): number {
    return this.sim?.particleCount ?? 0;
  }

  /** Emits `n` drops from a point, jittered inside `radius`. */
  emit(
    n: number,
    at: readonly [number, number, number],
    velocity: readonly [number, number, number] = [0, -5, 0],
    radius?: number,
  ): void {
    this.sim?.spawn(n, at, velocity, radius);
  }

  /**
   * Where the fluid is scraping against geometry, newest frame available. Filter by
   * speed and turn the fast ones into subtractive edits and you have erosion.
   *
   * A readback, so it resolves a few frames late. Awaiting it every frame is fine; the
   * sim does not wait for it.
   */
  async contacts(): Promise<FluidContact[]> {
    const hits = (await this.sim?.readContacts()) ?? [];
    return hits.map(([x, y, z, speed]) => ({ position: [x, y, z], speed }));
  }

  build(ctx: EntityContext): void {
    // ponytail: built once - a rebuild would drop every particle in flight. So a solid
    // spawned after the first frame is not collided against.
    if (this.sim) {
      return;
    }
    const o = this.options;
    this.sim = new FluidSim(ctx.root, ctx.colliders(this), {
      capacity: o.capacity ?? 6000,
      spacing: o.spacing ?? 0.34,
      material: ctx.material(o.material),
      origin: ctx.bounds.origin,
      worldSize: ctx.bounds.size,
      bakeResolution: o.bakeResolution ?? 128,
      bakeBand: 2,
      viscosity: o.viscosity ?? 0.4,
      gravity: o.gravity,
      label: o.label ?? 'fluid',
    });
  }

  simulate(pass: TgpuComputePass): void {
    this.sim?.step(pass);
  }

  destroy(): void {
    this.sim?.destroy();
  }
}
