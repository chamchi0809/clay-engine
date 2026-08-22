import { demos, type DemoContext } from './shell.ts';

/**
 * The page. Picks a demo out of the hash, mounts it, and stays out of its way.
 */
const canvas = document.getElementById('view') as HTMLCanvasElement;
const statusBox = document.getElementById('status') as HTMLElement;
const keysBox = document.getElementById('keys') as HTMLElement;
const blurbBox = document.getElementById('blurb') as HTMLElement;
const picker = document.getElementById('picker') as HTMLSelectElement;
const errorBox = document.getElementById('error') as HTMLElement;

const flags = new URLSearchParams(location.hash.slice(1));
const wanted = flags.get('demo');
const entry = demos.find((e) => e.id === wanted) ?? demos[0]!;

picker.replaceChildren(
  ...demos.map((e) => {
    const option = document.createElement('option');
    option.value = e.id;
    option.textContent = e.name;
    option.selected = e === entry;
    return option;
  }),
);
blurbBox.textContent = entry.blurb;

// A reload rather than a swap: see `DemoStart`. The demo's own flags are dropped along
// the way, since they mean nothing to the next one.
picker.addEventListener('change', () => {
  location.hash = `demo=${picker.value}`;
  location.reload();
});

const ctx: DemoContext = {
  canvas,
  status: (text) => {
    statusBox.textContent = text;
  },
  hud: (rows) => {
    keysBox.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  },
  flags,
};

ctx.status('loading…');
entry
  .load()
  .then((start) => start(ctx))
  .catch((e: unknown) => {
    errorBox.style.display = 'grid';
    errorBox.textContent = String(e instanceof Error ? (e.stack ?? e.message) : e);
    ctx.status('failed');
    throw e;
  });
