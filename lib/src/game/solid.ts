import { SdfBuilder } from '../field/builder.ts';
import { SdfEditor } from '../field/modify.ts';
import { SdfVolume } from '../field/volume.ts';
import { compileShape, type Shape } from '../shape/sdf.ts';
import { volumeField, type TracedField } from '../trace/field.ts';
import { GameObject } from './entity.ts';
import type { TgpuComputePass } from 'typegpu';
import type { EntityContext } from './entity.ts';
import type { Game } from './game.ts';

export interface SolidSpawnOptions {
  /** The whole shape, as one expression. */
  shape: Shape;
  /**
   * Voxels per axis. Cost is memory-cubic, so this is the one knob worth thinking about:
   * 128 over a 24-unit world is a 0.19 voxel, which is roughly Claybook's.
   */
  resolution?: number;
  /** World edge length covered. Defaults to the game's bounds. */
  size?: number;
  /** Minimum corner. Defaults to the game's bounds. */
  origin?: readonly [number, number, number];
  /**
   * Stored distance band half-width, in voxels. 4 was Claybook's; below 2 the tracer
   * loses the slack it needs to take big steps at coarse mips.
   */
  band?: number;
  /**
   * Draw the whole solid see-through. Defaults to false, and unlike a body or a fluid it
   * is not inferred: a level is one field over many materials, so there is no single
   * material to read an answer off. Set it for a solid that really is one substance - a
   * pane, an ice sheet, a jar - and give that substance an `opacity` below 1.
   */
  transparent?: boolean;
  label?: string;
}

/**
 * Static geometry: the level.
 *
 * One object with one shape, because that is what it *is* once it is on the GPU. Brushes
 * are fused into a single distance field during the bake, so an individual brush has no
 * identity afterwards - there is nothing to hand back a handle to. Change the level by
 * assigning a new {@link shape}, or carve it at runtime with {@link add}/{@link cut}.
 */
export class Solid extends GameObject {
  readonly volume: SdfVolume;
  readonly field: TracedField;
  readonly transparent: boolean;

  private readonly builder: SdfBuilder;
  private readonly editor: SdfEditor;
  private readonly resolveMaterial: (name: string) => number;
  private current: Shape;
  private rebuildQueued = true;

  constructor(game: Game, options: SolidSpawnOptions) {
    super(game);
    this.volume = new SdfVolume(game.root, {
      resolution: options.resolution ?? 128,
      worldSize: options.size ?? game.bounds.size,
      origin: options.origin ?? game.bounds.origin,
      band: options.band ?? 4,
      label: options.label ?? 'solid',
    });
    this.field = volumeField(this.volume);
    const brushSet = game.brushSet;
    this.builder = new SdfBuilder(this.volume, { brushSet });
    this.editor = new SdfEditor(this.volume, { brushSet });
    this.resolveMaterial = (n) => game.material(n);
    this.transparent = options.transparent ?? false;
    this.current = options.shape;
  }

  /** The shape currently baked. Assigning schedules a full rebuild next frame. */
  get shape(): Shape {
    return this.current;
  }
  set shape(next: Shape) {
    this.current = next;
    this.rebuildQueued = true;
  }

  /**
   * Fuses `shape` into the solid, re-baking only the region it touches. This is the
   * sculpting half of Claybook: a sphere per frame is affordable, a hundred is not.
   */
  add(shape: Shape): void {
    for (const brush of compileShape(shape, this.resolveMaterial)) {
      this.editor.push(brush);
    }
  }

  /** Subtracts `shape`. Digging, erosion, explosions. */
  cut(shape: Shape): void {
    for (const brush of compileShape(shape, this.resolveMaterial)) {
      this.editor.push({ ...brush, op: brush.op === 'paint' ? 'paint' : 'cut' });
    }
  }

  build(_ctx: EntityContext): void {}

  simulate(pass: TgpuComputePass): void {
    if (this.rebuildQueued) {
      this.builder.setBrushes(
        this.builder.brushSet.compile(compileShape(this.current, this.resolveMaterial)),
      );
      this.builder.rebuild(pass);
      this.rebuildQueued = false;
    }
    this.editor.flush(pass);
  }

  destroy(): void {
    this.editor.destroy();
    this.volume.destroy();
  }
}
