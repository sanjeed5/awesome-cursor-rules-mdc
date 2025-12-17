---
description: This guide provides definitive best practices for developing high-performance, maintainable applications with Vite, focusing on optimal configuration, code structure, and testing.
globs: **/*.{js,jsx}
---
# vite Best Practices

Vite is the modern standard for frontend tooling. Adhere to these principles to leverage its full potential, ensuring blazing-fast development and optimized production builds.

## 1. Code Organization and Structure

### Keep `vite.config.js` Minimal
Vite's philosophy is a lean core. Avoid over-configuring. Only add plugins or options when absolutely necessary.

❌ **BAD** - Overly complex `vite.config.js`
```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import legacy from '@vitejs/plugin-legacy';
import { visualizer } from 'rollup-plugin-visualizer';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    legacy({ targets: ['defaults', 'not IE 11'] }),
    visualizer({ filename: './dist/stats.html' }),
    VitePWA({ registerType: 'autoUpdate' }),
    // ... many more plugins
  ],
  resolve: {
    alias: {
      '@': '/src',
      '~': '/node_modules',
    },
    extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json', '.vue'],
  },
  build: {
    target: 'es2015',
    minify: 'terser',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
});
```

✅ **GOOD** - Lean and focused `vite.config.js`
```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Only add resolve.alias if absolutely needed for complex paths.
  // Avoid resolve.extensions unless you have specific non-standard file types.
  // Vite's defaults are usually sufficient.
});
```

### Use Explicit File Extensions
Relying on `resolve.extensions` for implicit imports forces Vite to perform multiple filesystem checks, slowing down resolution. Be explicit.

❌ **BAD** - Implicit import
```javascript
// src/components/MyComponent.jsx
import { util } from '../utils'; // Vite has to guess .js, .ts, .jsx etc.
```

✅ **GOOD** - Explicit import
```javascript
// src/components/MyComponent.jsx
import { util } from '../utils/index.js'; // Or .ts, .jsx, etc.
```

### Avoid Barrel Files
Barrel files (e.g., `index.js` re-exporting many modules) force Vite to fetch and transform all re-exported files, even if only one API is used. This hurts initial page load performance.

❌ **BAD** - Barrel file (`src/utils/index.js`)
```javascript
// src/utils/index.js
export * from './color.js';
export * from './dom.js';
export * from './slash.js';

// src/app.js
import { slash } from './utils'; // Loads color.js, dom.js, and slash.js
```

✅ **GOOD** - Direct imports
```javascript
// src/app.js
import { slash } from './utils/slash.js'; // Only loads slash.js
```

## 2. Common Patterns and Anti-patterns

### Embrace Native ES Modules
Vite is built on native ES Modules. Always write your client-side code using `import`/`export` syntax.

❌ **BAD** - CommonJS in client-side code
```javascript
// main.js
const myModule = require('./my-module'); // Will fail in browser
```

✅ **GOOD** - Native ES Modules
```javascript
// main.js
import myModule from './my-module.js';
```

### Use `import.meta.env` for Environment Variables
Vite injects environment variables via `import.meta.env`. This is the correct way to access them in client-side code. `process.env` is for Node.js environments.

❌ **BAD** - Using `process.env` in client code
```javascript
// app.js
console.log(process.env.VITE_API_URL); // `process` is not defined in browser
```

✅ **GOOD** - Using `import.meta.env`
```javascript
// app.js
console.log(import.meta.env.VITE_API_URL); // Correctly accesses Vite env vars
```

### Optimize with Dynamic Imports
For large components or libraries, use dynamic imports to load them only when needed, reducing initial bundle size and improving load times.

❌ **BAD** - Eagerly loading large component
```javascript
// App.jsx
import LargeComponent from './LargeComponent'; // Always bundled
function App() {
  return <LargeComponent />;
}
```

✅ **GOOD** - Dynamically importing
```javascript
// App.jsx (React example)
import { lazy, Suspense } from 'react';
const LargeComponent = lazy(() => import('./LargeComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <LargeComponent />
    </Suspense>
  );
}
```

## 3. Performance Considerations

### Audit Custom Plugins
Community plugins can introduce performance bottlenecks. Profile them using Vite's debug flags.

❌ **BAD** - Blindly adding plugins
```javascript
// vite.config.js
import { defineConfig } from 'vite';
import someHeavyPlugin from 'some-heavy-plugin'; // No profiling done

export default defineConfig({
  plugins: [someHeavyPlugin()],
});
```

✅ **GOOD** - Profiling plugins
```bash
# Run Vite with debug flags to identify slow plugins
vite --debug plugin-transform
```
Use `vite-plugin-inspect` to visualize the transform pipeline.

### Optimize Browser Setup
Browser extensions and disabled cache settings can severely impact dev server performance.

❌ **BAD** - Developing with "Disable Cache" enabled in dev tools.
```
// Browser Dev Tools -> Network tab -> "Disable Cache" checked
```

✅ **GOOD** - Use a clean browser profile or incognito mode.
Ensure "Disable Cache" is **unchecked** in dev tools.

### Warm Up Critical Files
For complex applications, pre-warming frequently used files can prevent request waterfalls.

❌ **BAD** - Relying solely on on-demand transformation for critical paths.
```javascript
// No explicit warmup configured
```

✅ **GOOD** - Use `server.warmup` in `vite.config.js`
```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    warmup: {
      clientFiles: ['./src/main.js', './src/App.jsx'],
      // Or use patterns: ['**/*.vue', '**/*.jsx']
    },
  },
});
```

## 4. Common Pitfalls and Gotchas

### Incorrect Base Path for Deployment
When deploying to a sub-path (e.g., `yourdomain.com/my-app/`), ensure `base` is correctly configured.

❌ **BAD** - Hardcoding absolute paths or missing `base`
```javascript
// vite.config.js
// Default base: '/'
// Assets might break when deployed to a sub-path
```

✅ **GOOD** - Configure `base` for sub-path deployments
```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/my-app/', // For deploying to https://yourdomain.com/my-app/
  // Or use './' for relative paths if the base is unknown at build time
  // base: './',
});
```
Access the base path in your code via `import.meta.env.BASE_URL`.

### Mismanaging `NODE_ENV` with API Usage
When using Vite's JS API (`createServer`, `build`) in the same Node.js process, ensure `process.env.NODE_ENV` or the `mode` config option is consistent to prevent conflicts.

❌ **BAD** - Conflicting `NODE_ENV`
```javascript
// script.js
process.env.NODE_ENV = 'production';
await createServer(); // Might behave unexpectedly
```

✅ **GOOD** - Explicitly set `mode` or spawn child processes
```javascript
// script.js
import { createServer } from 'vite';
// Option 1: Explicitly set mode
const devServer = await createServer({ mode: 'development' });
await devServer.listen();

// Option 2: Spawn child processes for separate contexts
// (e.g., one for dev server, one for build)
```

## 5. Testing Approaches

### Standardize on Vitest
Vitest is the official testing framework for Vite projects, offering seamless integration with Vite's configuration and plugin ecosystem.

❌ **BAD** - Using a separate test runner (e.g., Jest) that requires its own complex configuration.
```json
// package.json
"scripts": {
  "test": "jest" // Requires separate Babel/Webpack config
}
```

✅ **GOOD** - Integrate Vitest directly into `vite.config.ts`
```typescript
// vite.config.ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true, // For global APIs like `describe`, `it`, `expect`
    environment: 'jsdom', // Or 'node'
    setupFiles: './src/setupTests.js', // Global setup for tests
  },
});
```