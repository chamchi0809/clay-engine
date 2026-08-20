import { defineConfig } from 'vite';
import { clayPlugin } from '@clay/engine/vite';

export default defineConfig({
  // The plugin comes from the engine: the game's own `'use gpu'` closures need the
  // TypeGPU transform too, so it is the engine's dependency, not the game's.
  plugins: [clayPlugin()],
  // strictPort: silently landing on a different port than the README claims is how
  // you end up testing somebody else's app.
  server: { port: 4173, strictPort: true },
});
