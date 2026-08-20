import { defineConfig } from 'vite';
import { clayPlugin } from '@clay/engine/vite';

export default defineConfig({
  // The plugin is what makes a custom brush possible at all: it rewrites every `'use gpu'`
  // closure at build time into an AST the resolver turns into WGSL.
  plugins: [clayPlugin()],
  server: { port: 4175, strictPort: true },
});
