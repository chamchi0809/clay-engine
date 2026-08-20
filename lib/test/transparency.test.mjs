import { test } from 'node:test';
import assert from 'node:assert/strict';

/**
 * JS mirrors of the arithmetic in `trace/shade.ts` and `render/composite.ts`. The
 * transparency path is mostly texture plumbing, which fails loudly, plus four small
 * formulas that fail quietly: a Fresnel term that is subtly wrong still looks like glass,
 * an absorption that is not a length integral still looks tinted, a path length that is
 * not capped only misbehaves in front of a distant wall, and a material whose defaults
 * are filled wrong turns every opaque surface see-through.
 */

// --- Fresnel (Schlick) ----------------------------------------------------
const fresnel = (cos, f0) => {
  const c = Math.min(1, Math.max(0, 1 - cos));
  return f0 + (1 - f0) * c ** 5;
};

test('fresnel is f0 head-on and total at grazing', () => {
  assert.equal(fresnel(1, 0.04), 0.04);
  assert.equal(fresnel(0, 0.04), 1);
});

test('fresnel is monotonic, and near f0 over most of the hemisphere', () => {
  // The point of the fifth power: a dielectric is a window until it is nearly edge-on.
  // A linear ramp would put ~50% reflection at 60 degrees and turn every flat pool into
  // a mirror.
  let prev = -1;
  for (let i = 0; i <= 10; i++) {
    const v = fresnel(i / 10, 0.04);
    assert.ok(v <= prev || prev < 0, `not monotonic at cos=${i / 10}`);
    prev = v;
  }
  assert.ok(fresnel(Math.cos(Math.PI / 3), 0.04) < 0.08, '60 degrees should still be a window');
  assert.ok(fresnel(Math.cos((85 * Math.PI) / 180), 0.04) > 0.5, '85 degrees should be a mirror');
});

// --- Beer-Lambert absorption ----------------------------------------------
// `exp(-(1 - albedo) * absorption * thickness)`, per channel. The tint is `1 - albedo`
// so a blue material is the one that lets blue through.
const absorb = (albedo, absorption, thickness) =>
  albedo.map((a) => Math.exp(-(1 - a) * absorption * thickness));

test('absorption leaves the material colour and eats its complement', () => {
  const water = [0.24, 0.5, 0.78];
  const [r, g, b] = absorb(water, 1.4, 2);
  assert.ok(b > g && g > r, `blue must survive deepest, got ${[r, g, b]}`);
  assert.ok(r < 0.2, 'red should be mostly gone through 2 units of water');
});

test('absorption is a length integral, not a tint: it compounds with distance', () => {
  const water = [0.24, 0.5, 0.78];
  const one = absorb(water, 1.4, 1);
  const two = absorb(water, 1.4, 2);
  // exp(-k*2) == exp(-k)^2 is the property that makes a shallow edge clear and a deep
  // middle saturated. A flat per-material tint would give the same colour at both depths.
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(two[i] - one[i] ** 2) < 1e-12);
  }
  assert.ok(two[0] < one[0]);
});

test('zero absorption is perfectly clear glass at any thickness', () => {
  for (const t of [0, 0.5, 40]) {
    assert.deepEqual(absorb([0.24, 0.5, 0.78], 0, t), [1, 1, 1]);
  }
});

// --- path length between the two G-buffer layers --------------------------
// `hasBackdrop ? clamp(tBackdrop - tFront, 0, cap) : cap`.
const pathLength = (tFront, tBackdrop, cap) =>
  tBackdrop > 0 ? Math.min(Math.max(tBackdrop - tFront, 0), cap) : cap;

test('path length is the gap between the layers when there is a backdrop', () => {
  assert.ok(Math.abs(pathLength(10, 10.4, 1.2) - 0.4) < 1e-9);
});

test('path length falls back to the cap against open sky', () => {
  // A negative backdrop distance is the sky mask, not a distance.
  assert.equal(pathLength(10, -1, 1.2), 1.2);
});

test('the cap is what stops a thin sheet in front of a far wall reading as solid', () => {
  // Without it, 0.3 units of falling water 30 units in front of a wall absorbs as if it
  // were 30 units thick and goes black.
  assert.equal(pathLength(10, 40, 1.2), 1.2);
  const capped = absorb([0.24, 0.5, 0.78], 1.4, pathLength(10, 40, 1.2));
  const raw = absorb([0.24, 0.5, 0.78], 1.4, 30);
  assert.ok(capped[2] > 0.4, 'capped water still transmits');
  assert.ok(raw[2] < 1e-3, 'uncapped water is black - the bug the cap exists for');
});

test('a backdrop nearer than the surface is zero path, not negative', () => {
  // Depth-testing should have culled this pixel, but a coincident surface can still
  // land here, and a negative thickness would *amplify* the backdrop through exp().
  assert.equal(pathLength(10, 9.5, 1.2), 0);
});

// --- material defaults ----------------------------------------------------
const normalizeMaterial = (spec) => ({
  albedo: [spec.albedo[0], spec.albedo[1], spec.albedo[2]],
  roughness: spec.roughness ?? 0.7,
  emissive: [...(spec.emissive ?? [0, 0, 0])],
  metallic: spec.metallic ?? 0,
  opacity: spec.opacity ?? 1,
  ior: spec.ior ?? 1.33,
  absorption: spec.absorption ?? 0.5,
});

test('a material that says nothing about transparency is opaque', () => {
  // The whole palette defaults through here, so an `opacity` default of anything but 1
  // would put every surface in the scene into the transparency layer at once.
  const m = normalizeMaterial({ albedo: [0.78, 0.44, 0.36] });
  assert.equal(m.opacity, 1);
});

test('an explicit zero survives the defaults', () => {
  // `?? 0` is the reason these are nullish coalescings rather than `||`: a fully
  // transmissive material asks for opacity 0, and `||` would hand it back 1.
  const m = normalizeMaterial({ albedo: [0, 0, 0], opacity: 0, roughness: 0, absorption: 0 });
  assert.equal(m.opacity, 0);
  assert.equal(m.roughness, 0);
  assert.equal(m.absorption, 0);
});

// --- the composite mix ----------------------------------------------------
test('opacity 1 composites to exactly the opaque shading', () => {
  // The fast path is a separate pass, but the formula still has to agree at the boundary,
  // or a material animated from 0.99 to 1 pops.
  const mix = (a, b, t) => a * (1 - t) + b * t;
  const transmitted = 0.9;
  const diffuse = 0.2;
  assert.equal(mix(transmitted, diffuse, 1), diffuse);
  assert.equal(mix(transmitted, diffuse, 0), transmitted);
});
