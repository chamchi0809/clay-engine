import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot, TgpuTexture, TgpuTextureView, TgpuUniform, SampledFlag, StorageFlag } from 'typegpu';
import * as sdf from '@typegpu/sdf';

/** Voxels per tile edge. Every sparse dispatch works on one 8^3 tile per workgroup. */
export const TILE = 8;
/** Threads per tile workgroup. */
export const TILE_THREADS = TILE * TILE * TILE;

export const VolumeParams = d.struct({
  /** World position of the volume's minimum corner. */
  origin: d.vec3f,
  /** World units per voxel at mip 0. */
  voxelSize: d.f32,
  /** World size of the volume (cubic). */
  extent: d.vec3f,
  /** Half-width of the stored distance band, in voxels of the sampled mip. */
  band: d.f32,
  center: d.vec3f,
  maxMip: d.f32,
  halfExtent: d.vec3f,
  resolution: d.f32,
});

/** Per-mip constants for the generation passes. One buffer per level, written once. */
export const LevelParams = d.struct({
  mip: d.u32,
  /** Voxel resolution of this mip. */
  mipRes: d.u32,
  /** Tile resolution of this mip (`mipRes / TILE`). */
  tileRes: d.u32,
  /** Where this mip's tile records start inside the shared grid buffers. */
  cellOffset: d.u32,
});

export interface SdfVolumeOptions {
  /** Mip-0 voxel resolution per axis. Must be a multiple of `TILE * 2^(mipLevels-1)`. */
  resolution?: number;
  /** World-space edge length covered by the volume. */
  worldSize?: number;
  /** World position of the minimum corner. */
  origin?: readonly [number, number, number];
  /** Number of mip levels; each halves the resolution. */
  mipLevels?: number;
  /** Stored band half-width in voxels (Claybook used 4). */
  band?: number;
  /**
   * Central-difference stencil for {@link SdfVolume.gradient}, in voxels. Trilinear
   * reconstruction has a discontinuous gradient at every cell face, so a one-voxel
   * stencil on a curved surface returns a lattice-aligned sawtooth - harmless for
   * stepping, but visible as banding wherever the normal is *shaded*. Widening it
   * averages the sawtooth away at the cost of blunting genuine creases. Defaults to 1.
   */
  gradientVoxels?: number;
  label?: string;
}

/** `createView` returns a union of both view kinds; these narrow it back per binding. */
const sampled3dSchema = d.texture3d(d.f32);
const storage3dSchema = d.textureStorage3d('rgba16float', 'write-only');
export type SampledView = TgpuTextureView<typeof sampled3dSchema>;
export type StorageView = TgpuTextureView<typeof storage3dSchema>;

export type SdfTexture = TgpuTexture<{
  size: [number, number, number];
  format: 'rgba16float';
  dimension: '3d';
  mipLevelCount: number;
}> & SampledFlag & StorageFlag;

/**
 * A mip-mapped signed distance volume.
 *
 * Layout follows Claybook (GDC'18 slides 7-8, 20): a band-limited field stored per
 * voxel, `+-band` voxels wide, normalised to `[-1, 1]`, with the world-space band
 * width doubling per mip level. That is what makes the hierarchical sphere-tracing
 * loop work: a saturated sample (`|v| == 1`) means "at least `band` voxels of empty
 * space at this level", so the tracer can hop to a coarser mip and take much longer
 * steps.
 *
 * Claybook packed this into 8-bit snorm. WebGPU's core storage formats do not include
 * a filterable single-channel 8/16-bit format, so we use `rgba16float` and spend the
 * extra channels: `.r` = normalised distance, `.g` = material id.
 */
export class SdfVolume {
  readonly root: TgpuRoot;
  readonly resolution: number;
  readonly worldSize: number;
  readonly voxelSize: number;
  readonly band: number;
  readonly mipLevels: number;
  /** Lower corner of the covered box. Mutable via {@link setOrigin}. */
  origin: readonly [number, number, number];
  /** Mip-0 tile resolution. */
  readonly tileRes: number;
  /** Tile resolution per mip level. */
  readonly tileResPerMip: number[];
  /** Start index of each mip's tile records inside the shared grid buffers. */
  readonly cellOffsets: number[];
  /** Total tile records across all mips. */
  readonly totalCells: number;

  readonly texture: SdfTexture;
  /** Sampled view covering every mip - for rendering and simulation queries. */
  readonly sampledView: SampledView;
  /** Per-mip write-only storage views - generation targets. */
  readonly storageViews: StorageView[];
  /** Per-mip single-level sampled views - safe to read while writing another level. */
  readonly levelViews: SampledView[];
  readonly sampler: ReturnType<TgpuRoot['createSampler']>;
  readonly params: TgpuUniform<typeof VolumeParams>;
  readonly levelParams: TgpuUniform<typeof LevelParams>[];

  /** Bind group layout for reading the field (sampler + all mips + params). */
  readonly layout: ReturnType<typeof makeVolumeReadLayout>;
  readonly bindGroup: ReturnType<TgpuRoot['createBindGroup']>;
  /** One bind group per mip exposing that level as a write-only storage texture. */
  readonly writeGroups: ReturnType<TgpuRoot['createBindGroup']>[];
  /** One bind group per mip exposing that level as a plain sampled texture. */
  readonly levelGroups: ReturnType<TgpuRoot['createBindGroup']>[];

  /** `(p, mip) -> raw texel`, normalised distance in `.r`, material in `.g`. */
  readonly sampleRaw: (p: d.v3f, mip: number) => d.v4f;
  /** World-space band half-width at a mip level. */
  readonly bandWorld: (mip: number) => number;
  /** Conservative world-space distance to the surface. Falls back to the volume bounds outside. */
  readonly sampleDist: (p: d.v3f, mip: number) => number;
  /** `(distance, material)` in world units. */
  readonly sampleField: (p: d.v3f, mip: number) => d.v2f;
  /** Normalised gradient of the field at mip 0. */
  readonly gradient: (p: d.v3f) => d.v3f;

  constructor(root: TgpuRoot, options: SdfVolumeOptions = {}) {
    const resolution = options.resolution ?? 128;
    const mipLevels = options.mipLevels ?? Math.max(1, Math.log2(resolution / TILE) | 0);
    const worldSize = options.worldSize ?? 16;
    const band = options.band ?? 4;
    const origin = options.origin ?? [0, 0, 0];

    if (resolution % (TILE * 2 ** (mipLevels - 1)) !== 0) {
      throw new Error(
        `SdfVolume: resolution ${resolution} must be a multiple of ${TILE * 2 ** (mipLevels - 1)} for ${mipLevels} mip levels`,
      );
    }

    this.root = root;
    this.resolution = resolution;
    this.worldSize = worldSize;
    this.voxelSize = worldSize / resolution;
    this.band = band;
    this.mipLevels = mipLevels;
    this.origin = origin;
    this.tileRes = resolution / TILE;
    this.tileResPerMip = [];
    this.cellOffsets = [];
    let offset = 0;
    for (let m = 0; m < mipLevels; m++) {
      const tr = this.tileRes >> m;
      this.tileResPerMip.push(tr);
      this.cellOffsets.push(offset);
      offset += tr * tr * tr;
    }
    this.totalCells = offset;

    this.texture = root
      .createTexture({
        size: [resolution, resolution, resolution],
        format: 'rgba16float',
        dimension: '3d',
        mipLevelCount: mipLevels,
      })
      .$usage('sampled', 'storage')
      .$name(options.label ?? 'sdfVolume') as SdfTexture;

    this.sampledView = this.texture.createView(sampled3dSchema) as SampledView;
    this.storageViews = [];
    this.levelViews = [];
    for (let m = 0; m < mipLevels; m++) {
      this.storageViews.push(
        this.texture.createView(storage3dSchema, {
          baseMipLevel: m,
          mipLevelCount: 1,
        }) as StorageView,
      );
      this.levelViews.push(
        this.texture.createView(sampled3dSchema, {
          baseMipLevel: m,
          mipLevelCount: 1,
        }) as SampledView,
      );
    }

    // Nearest mip filtering: the tracer hops between whole levels, so blending two
    // levels would double the texel fetches for nothing.
    this.sampler = root.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    const half = worldSize / 2;
    this.params = root.createUniform(VolumeParams, {
      origin: [origin[0], origin[1], origin[2]],
      voxelSize: this.voxelSize,
      extent: [worldSize, worldSize, worldSize],
      band,
      center: [origin[0] + half, origin[1] + half, origin[2] + half],
      maxMip: mipLevels - 1,
      halfExtent: [half, half, half],
      resolution,
    });

    this.levelParams = [];
    for (let m = 0; m < mipLevels; m++) {
      this.levelParams.push(
        root.createUniform(LevelParams, {
          mip: m,
          mipRes: resolution >> m,
          tileRes: this.tileResPerMip[m],
          cellOffset: this.cellOffsets[m],
        }),
      );
    }

    // Each volume gets its own layout object so a shader can sample two different
    // volumes (e.g. clay world + fluid) without the two colliding on one @group.
    this.layout = makeVolumeReadLayout();
    this.bindGroup = root.createBindGroup(this.layout, {
      tex: this.sampledView,
      samp: this.sampler,
      params: this.params,
    });

    this.writeGroups = this.storageViews.map((out) =>
      root.createBindGroup(volumeWriteLayout, { out }),
    );
    this.levelGroups = this.levelViews.map((src) =>
      root.createBindGroup(volumeLevelLayout, { src, srcSamp: this.sampler }),
    );

    const layout = this.layout;

    this.sampleRaw = (p: d.v3f, mip: number) => {
      'use gpu';
      const uv = (p - layout.$.params.origin) / layout.$.params.extent;
      // Callers pass integer literals for `mip`; WGSL needs an f32 level.
      return std.textureSampleLevel(layout.$.tex, layout.$.samp, uv, d.f32(mip));
    };

    this.bandWorld = (mip: number) => {
      'use gpu';
      return layout.$.params.band * layout.$.params.voxelSize * std.exp2(d.f32(mip));
    };

    const sampleRaw = this.sampleRaw;
    const bandWorld = this.bandWorld;

    this.sampleDist = (p: d.v3f, mip: number) => {
      'use gpu';
      // Outside the volume the distance to the volume box is a valid lower bound and
      // avoids sampling clamped border voxels.
      const dBox = sdf.sdBox3d(p - layout.$.params.center, layout.$.params.halfExtent);
      if (dBox > 0) {
        return dBox;
      }
      return sampleRaw(p, mip).x * bandWorld(mip);
    };

    this.sampleField = (p: d.v3f, mip: number) => {
      'use gpu';
      const dBox = sdf.sdBox3d(p - layout.$.params.center, layout.$.params.halfExtent);
      if (dBox > 0) {
        return d.vec2f(dBox, 0);
      }
      const s = sampleRaw(p, mip);
      return d.vec2f(s.x * bandWorld(mip), s.y);
    };

    const sampleDist = this.sampleDist;
    const gradScale = options.gradientVoxels ?? 1;
    this.gradient = (p: d.v3f) => {
      'use gpu';
      const h = layout.$.params.voxelSize * gradScale;
      const dx = d.vec3f(h, 0, 0);
      const dy = d.vec3f(0, h, 0);
      const dz = d.vec3f(0, 0, h);
      const g = d.vec3f(
        sampleDist(p + dx, 0) - sampleDist(p - dx, 0),
        sampleDist(p + dy, 0) - sampleDist(p - dy, 0),
        sampleDist(p + dz, 0) - sampleDist(p - dz, 0),
      );
      const len = std.length(g);
      return std.select(d.vec3f(0, 1, 0), g * (1 / len), len > 1e-8);
    };
  }

  /** World-space size of one voxel at a mip level. */
  /**
   * Moves the covered box. Everything shader-side reads the origin from the uniform, so
   * a volume can follow a moving object: a body-sized volume at a fine voxel is orders
   * of magnitude cheaper than a world-sized one at the same detail, and outside the box
   * a volume field reports the distance to the box, which is still a valid bound.
   *
   * Whatever filled the volume is *not* re-baked - the caller owns that. For a splat
   * this is free (it rebuilds every frame anyway); for a brush-built volume it means a
   * full rebuild.
   */
  setOrigin(origin: readonly [number, number, number]): void {
    this.origin = [origin[0], origin[1], origin[2]];
    const half = this.worldSize / 2;
    this.params.writePartial({
      origin: d.vec3f(origin[0], origin[1], origin[2]),
      center: d.vec3f(origin[0] + half, origin[1] + half, origin[2] + half),
    });
  }

  voxelSizeAt(mip: number): number {
    return this.voxelSize * 2 ** mip;
  }

  destroy(): void {
    this.texture.destroy();
  }
}

/** Write target for generation passes: one mip level as a storage texture. */
export const volumeWriteLayout = tgpu.bindGroupLayout({
  out: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
});

/**
 * A single mip level as a sampled texture. Separate from {@link volumeWriteLayout} so
 * a pass can read level `m-1` while writing level `m` - different subresources, which
 * WebGPU allows, unlike aliasing one view.
 */
export const volumeLevelLayout = tgpu.bindGroupLayout({
  src: { texture: d.texture3d(d.f32) },
  srcSamp: { sampler: 'filtering' },
});

export function makeVolumeReadLayout() {
  return tgpu.bindGroupLayout({
    tex: { texture: d.texture3d(d.f32) },
    samp: { sampler: 'filtering' },
    params: { uniform: VolumeParams },
  });
}
