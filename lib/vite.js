import typegpu from 'unplugin-typegpu/vite';

/**
 * The Vite plugin a consumer of this engine must install.
 *
 * The engine's extension mechanism is "write a `'use gpu'` closure and hand it to
 * `analyticField`", which means the *game's* source needs the TypeGPU transform, not
 * just the engine's. Shipping the plugin from here keeps that one dependency instead of
 * a build detail every consumer has to rediscover.
 *
 * Plain `.js`, not `.ts`: Vite loads its own config through Node, which will not import
 * a bare `.ts` file out of a linked package.
 *
 * @returns {import('vite').Plugin}
 */
export function clayPlugin() {
  // Every .ts/.js, not a src glob: the engine is consumed as source through the
  // workspace link, so its files must be transformed as well.
  return typegpu({ include: [/\.m?[jt]sx?$/] });
}
