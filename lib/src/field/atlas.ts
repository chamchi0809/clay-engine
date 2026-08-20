import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot, TgpuTexture, TgpuTextureView, SampledFlag, StorageFlag } from 'typegpu';
import type { BrushVolumeSource } from './brush.ts';
import type { FieldGroups } from '../trace/field.ts';

const sampled3dSchema = d.texture3d(d.f32);
const storage3dSchema = d.textureStorage3d('rgba16float', 'write-only');

export type AtlasTexture = TgpuTexture<{
  size: [number, number, number];
  format: 'rgba16float';
  dimension: '3d';
}> & SampledFlag & StorageFlag;

export interface BrushAtlasOptions {
  /**
   * Voxels per axis of one slot. Cost is cubic and shared by every slot, so this is the
   * knob: 32 is a silhouette, 48 holds a recognisable rock, 64 holds a face.
   */
  resolution?: number;
  /** How many shapes can be baked. Allocated up front - the texture cannot grow. */
  slots?: number;
  label?: string;
}

/**
 * A grid of baked signed distance fields, one per slot, in a single 3D texture.
 *
 * This is the escape hatch Claybook actually shipped: brushes it had no closed form for
 * were baked offline into small volume textures and sampled as ordinary primitives
 * (GDC'18 slide 9). The engine evaluates its analytic kinds analytically instead, so this
 * exists only for the shapes that have no closed form - a mesh, most of all.
 *
 * One texture rather than one per shape, because the brush fold is a single shader: a
 * separate texture per mesh would mean a separate binding per mesh, and the number of
 * bindings is fixed when the pipeline is compiled. A slot index in the brush struct costs
 * nothing and is unbounded.
 *
 * Slots are stacked along Z, and every sample is inset by half a texel before it is mapped
 * into its slot's span - without that, a linear fetch at a slot's Z border would blend in
 * the shape stacked behind it.
 *
 * Each slot stores the field of a cube of half-extent 1, divided by that half-extent: a
 * *normalised* distance. That is what lets one bake serve any size and any scale - the
 * brush multiplies back by its own half-extent - and it is why the bake never needs to
 * know how big the shape will be in the world.
 */
export class BrushAtlas implements BrushVolumeSource {
  readonly root: TgpuRoot;
  readonly resolution: number;
  readonly slots: number;
  readonly texture: AtlasTexture;
  readonly sampledView: TgpuTextureView<typeof sampled3dSchema>;
  /** The whole atlas as a write target. What the baker stores into. */
  readonly storageView: TgpuTextureView<typeof storage3dSchema>;
  readonly sampler: ReturnType<TgpuRoot['createSampler']>;
  readonly layout: ReturnType<typeof makeAtlasReadLayout>;
  readonly groups: FieldGroups;
  /** Bind the atlas as a write target. For the baker, not for a field. */
  readonly writeGroup: ReturnType<TgpuRoot['createBindGroup']>;
  readonly sampleSlot: (slot: number, q: d.v3f) => number;

  private used = 0;

  constructor(root: TgpuRoot, options: BrushAtlasOptions = {}) {
    const resolution = options.resolution ?? 48;
    const slots = options.slots ?? 16;
    const depth = resolution * slots;
    const limit = root.device.limits.maxTextureDimension3D;
    if (depth > limit) {
      throw new Error(
        `BrushAtlas: ${slots} slots of ${resolution} stack to ${depth} along Z, past this `
          + `device's 3D texture limit of ${limit}. Lower \`resolution\` or \`slots\`.`,
      );
    }

    this.root = root;
    this.resolution = resolution;
    this.slots = slots;

    // No mip chain. The world volume's coarse levels point-sample the atlas, exactly as
    // they point-sample an analytic brush, and a point-sampled distance is still a valid
    // step bound; what mips would buy is anti-aliasing of a thin feature, which the world
    // volume's own trilinear reconstruction does not have either.
    this.texture = root
      .createTexture({
        size: [resolution, resolution, depth],
        format: 'rgba16float',
        dimension: '3d',
      })
      .$usage('sampled', 'storage')
      .$name(options.label ?? 'brushAtlas') as AtlasTexture;

    this.sampledView = this.texture.createView(sampled3dSchema);
    this.storageView = this.texture.createView(storage3dSchema);
    this.sampler = root.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.layout = makeAtlasReadLayout();
    const layout = this.layout;
    this.groups = [
      root.createBindGroup(layout, { tex: this.sampledView, samp: this.sampler }),
    ];
    this.writeGroup = root.createBindGroup(atlasWriteLayout, { out: this.storageView });

    // Half a texel of the *slot*, in slot-local units, and the slot's share of Z.
    const inset = 0.5 / resolution;
    const span = 1 / slots;
    this.sampleSlot = (slot: number, q: d.v3f) => {
      'use gpu';
      const local = std.clamp(
        q * 0.5 + d.vec3f(0.5),
        d.vec3f(inset),
        d.vec3f(1 - inset),
      );
      return std.textureSampleLevel(
        layout.$.tex,
        layout.$.samp,
        d.vec3f(local.x, local.y, (d.f32(slot) + local.z) * span),
        0,
      ).x;
    };
  }

  /**
   * Claims the next free slot.
   *
   * Slots are never returned. A baked shape is an asset - it is loaded once and referred
   * to by however many brushes want it - so a free list would be machinery for a case that
   * does not arise. Raise `slots` if a game really does load shapes for a level at a time.
   */
  allocate(): number {
    if (this.used >= this.slots) {
      throw new Error(
        `BrushAtlas: all ${this.slots} slots are taken. Pass a larger \`slots\` when the `
          + 'game is created.',
      );
    }
    return this.used++;
  }

  /** Slots claimed so far. */
  get allocated(): number {
    return this.used;
  }

  destroy(): void {
    this.texture.destroy();
  }
}

/** Write target for the baker: the whole atlas as a storage texture. */
export const atlasWriteLayout = tgpu.bindGroupLayout({
  out: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
});

/**
 * Per-instance, like a volume's read layout: a shader that samples two atlases would
 * otherwise bind both to the same `@group`.
 */
export function makeAtlasReadLayout() {
  return tgpu.bindGroupLayout({
    tex: { texture: d.texture3d(d.f32) },
    samp: { sampler: 'filtering' },
  });
}
