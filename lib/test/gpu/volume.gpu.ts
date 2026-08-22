import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { SdfVolume } from '../../src/field/volume.ts';
import { adapterLimits, hasWebGPU } from './harness.mjs';

/**
 * The volume's shape, as opposed to its contents - which `meshbake.gpu.ts` covers.
 *
 * A resolution is a number a caller picks by eye, from how fine they want the field and how
 * much texture they are willing to spend, and the only constraint they should have to hold in
 * their head is that it tiles. Everything else - how deep the mip chain goes, how many tile
 * records that comes to - is the volume's business. These pin that down, because getting it
 * wrong is a constructor throw in the user's face at the worst possible moment: after the
 * device exists, during a level load, with nothing on screen to explain it.
 */
let root: TgpuRoot | null = null;
let available = false;

before(async () => {
  available = await hasWebGPU();
  if (available) {
    root = await tgpu.init({ device: { requiredLimits: await adapterLimits() } });
  }
});

test('any multiple of the tile size is a legal resolution', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }

  // 160 is the case that used to throw. Its tile grid is 20 cubed, which halves twice and
  // then hits 5 - so three levels, even though log2(20) is nearly 4.4 and the old default
  // rounded it down to 4 and then demanded a multiple of 64.
  const odd = new SdfVolume(root, { resolution: 160, worldSize: 12 });
  assert.equal(odd.mipLevels, 3, '160 carries three levels');
  assert.deepEqual(odd.tileResPerMip, [20, 10, 5], 'and its tile grid halves twice');
  assert.equal(odd.totalCells, 20 ** 3 + 10 ** 3 + 5 ** 3);

  // 40 halves not at all: 5 tiles, and 5 is odd. One level, no chain.
  const flat = new SdfVolume(root, { resolution: 40, worldSize: 4 });
  assert.equal(flat.mipLevels, 1, 'an odd tile grid gets a single level');
  assert.deepEqual(flat.tileResPerMip, [5]);

  // The powers of two are the ones every other test and demo already runs on, so the
  // derivation has to leave them exactly where they were.
  for (const [resolution, levels] of [[64, 3], [128, 4], [256, 5]] as const) {
    const pow = new SdfVolume(root, { resolution });
    assert.equal(pow.mipLevels, levels, `${resolution} still gets ${levels} levels`);
  }
});

test('a resolution that cannot carry the mip chain asked for is refused', async (t) => {
  if (!available || !root) {
    return t.skip('no WebGPU adapter');
  }

  // Spelling out `mipLevels` is a claim about the chain, and a claim can be wrong. 160 can
  // do three levels; asking it for four is asking for a 2.5-tile mip.
  assert.throws(
    () => new SdfVolume(root!, { resolution: 160, mipLevels: 4 }),
    /must be a multiple of 64/,
  );
  // And a resolution that does not tile at all fails whatever the chain.
  assert.throws(() => new SdfVolume(root!, { resolution: 100 }), /multiple of 8/);
});
