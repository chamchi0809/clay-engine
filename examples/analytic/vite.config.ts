import { defineConfig } from 'vite';
import { clayPlugin } from '@clay/engine/vite';

export default defineConfig({
  plugins: [clayPlugin()],
  // strictPort: silently landing on a different port than the README claims is how
  // you end up testing somebody else's app.
  server: { port: 4174, strictPort: true },
});
