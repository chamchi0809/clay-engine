/**
 * What every demo on this page is handed, and what it has to be.
 *
 * The four demos were four separate Vite apps until they were not. They each want a
 * canvas, a status line and a panel of key bindings, and keeping one copy of that here is
 * the entire reason the shell exists - a demo file should be about the engine, not about
 * `document.getElementById`.
 */

export interface DemoContext {
  canvas: HTMLCanvasElement;
  /** The bottom-left line. Frame counters and load progress go here. */
  status(text: string): void;
  /** The top-left panel, as label/value rows. Called with `[]` it empties. */
  hud(rows: readonly (readonly [string, string])[]): void;
  /**
   * `#demo=clay&page=2` - the location hash, minus the `demo` key the shell itself reads.
   * Diagnosing a raymarcher without a way to freeze a term or retune a knob is guesswork,
   * so every demo is free to define its own.
   */
  flags: URLSearchParams;
}

/**
 * A demo's whole contract: set the scene up, start its own frame loop, never return.
 *
 * There is no teardown, because there is no switching: choosing another demo reloads the
 * page. A `Game` owns a GPU device, a swap chain and half a dozen pipelines, and a
 * half-released one of those is a far worse bug than a reload is a slow transition.
 */
export type DemoStart = (ctx: DemoContext) => Promise<void>;

export interface DemoEntry {
  /** URL slug, and the value of `#demo=`. */
  id: string;
  name: string;
  /** One line, shown under the picker. */
  blurb: string;
  /** Loaded on demand, so picking one demo does not compile the other three. */
  load: () => Promise<DemoStart>;
}

export const demos: readonly DemoEntry[] = [
  {
    id: 'clay',
    name: 'Claybook clone',
    blurb: 'A rolling ball of clay, three levels, water that pools and erodes.',
    load: () => import('./demos/clay.ts').then((m) => m.start),
  },
  {
    id: 'geometry',
    name: 'Geometry catalogue',
    blurb: 'Every three.js geometry at once, baked, next to the analytic primitives it overlaps with.',
    load: () => import('./demos/geometry.ts').then((m) => m.start),
  },
  {
    id: 'brushes',
    name: 'Custom brushes',
    blurb: 'A hand-written primitive and three baked meshes, used as ordinary brushes.',
    load: () => import('./demos/brushes.ts').then((m) => m.start),
  },
  {
    id: 'analytic',
    name: 'Analytic field',
    blurb: 'The same renderer with no clay, no volume and no simulation at all.',
    load: () => import('./demos/analytic.ts').then((m) => m.start),
  },
];
