import { readdir, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { rolldown } from 'rolldown';
import typegpu from 'unplugin-typegpu/rollup';

/**
 * Runs the GPU tests.
 *
 * `'use gpu'` closures are not runnable JavaScript - the TypeGPU plugin rewrites each one
 * at build time into an AST the resolver turns into WGSL, and without that pass a shader
 * function throws the moment it is resolved. That is why the engine ships a Vite plugin and
 * why `node --test` cannot simply import a `.ts` file here.
 *
 * So the tests are bundled the same way a game is, with the same plugin, and then run. The
 * bundler is Rolldown because Vite 8 already brings it; `packages: 'external'` keeps it to
 * the engine's own sources, which are the only files that need the transform.
 */
const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, '.out');

const entries = (await readdir(here))
  .filter((f) => f.endsWith('.gpu.ts'))
  .map((f) => join(here, f));

if (entries.length === 0) {
  console.log('test/gpu: no *.gpu.ts entries');
  process.exit(0);
}

await rm(outDir, { recursive: true, force: true });

const bundle = await rolldown({
  input: entries,
  plugins: [typegpu({ include: [/\.m?[jt]sx?$/] })],
  platform: 'node',
  // Only the engine's sources need the transform; everything installed ships pre-built.
  // Matched as a predicate rather than a regex because Rolldown also offers up already
  // resolved ids, and a Windows absolute path (`C:\...`) reads as a bare specifier to any
  // "does not start with a dot" test - which quietly externalised the whole engine.
  external: (id) => !/^[.]{1,2}[\\/]/.test(id) && !/^([A-Za-z]:|[\\/])/.test(id),
});
await bundle.write({ dir: outDir, format: 'esm', entryFileNames: '[name].mjs' });
await bundle.close();

const built = (await readdir(outDir)).filter((f) => f.endsWith('.mjs'));
let failed = 0;
for (const file of built) {
  const code = await new Promise((resolve) => {
    // `--test-force-exit`: the Dawn binding keeps a handle open for the lifetime of the
    // instance, so a run that passed every assertion still hangs at exit.
    spawn(process.execPath, ['--test', '--test-force-exit', join(outDir, file)], {
      stdio: 'inherit',
    }).on('close', resolve);
  });
  if (code !== 0) {
    failed++;
  }
}
process.exit(failed > 0 ? 1 : 0);
