---
description: Definitive guidelines for configuring and using esbuild to achieve ultra-fast, optimized, and production-ready JavaScript/TypeScript builds.
globs: **/*.{js,jsx}
---
# esbuild Best Practices

esbuild is the definitive choice for speed-first JavaScript/TypeScript bundling. Follow these rules to maximize its performance, maintainability, and production readiness.

## 1. Centralize Configuration

Always manage esbuild options in a dedicated `.mjs` file for clarity, reproducibility, and IDE linting. Avoid scattering CLI flags across `package.json` scripts.

❌ BAD
```json
// package.json
"scripts": {
  "build": "esbuild src/index.js --bundle --minify --outfile=dist/bundle.js"
}
```

✅ GOOD
```javascript
// esbuild.config.mjs
import * as esbuild from 'esbuild';

const isProduction = process.env.NODE_ENV === 'production';

esbuild.build({
  entryPoints: ['src/index.js'],
  bundle: true,
  outfile: 'dist/bundle.js',
  minify: isProduction,
  sourcemap: !isProduction ? 'inline' : 'external', // External sourcemaps for production debugging
  define: {
    'process.env.NODE_ENV': JSON.stringify(isProduction ? 'production' : 'development'),
  },
}).catch(() => process.exit(1));
```

## 2. Optimize for Target Environments

Specify the `target` to ensure esbuild only transpiles what's necessary, reducing bundle size and build time.

❌ BAD
```javascript
// esbuild.config.mjs
esbuild.build({
  // ...
  // No target specified, defaults to 'esnext' which might over-transpile
});
```

✅ GOOD
```javascript
// esbuild.config.mjs
esbuild.build({
  // ...
  target: ['es2022', 'chrome90', 'firefox90'], // Target modern browsers/Node.js versions
});
```

## 3. Leverage Incremental Builds for Development

For rapid development feedback, enable `incremental` builds or use `watch` mode.

❌ BAD
```javascript
// esbuild.config.mjs (development)
esbuild.build({
  entryPoints: ['src/index.js'],
  bundle: true,
  outfile: 'dist/bundle.js',
}).catch(() => process.exit(1)); // Full rebuild on every run
```

✅ GOOD
```javascript
// esbuild.config.mjs (development)
import * as esbuild from 'esbuild';

async function devBuild() {
  const ctx = await esbuild.context({
    entryPoints: ['src/index.js'],
    bundle: true,
    outdir: 'dist',
    sourcemap: true,
  });
  await ctx.watch(); // Near-instant rebuilds on file changes
  console.log('Watching for changes...');
}
devBuild();
```

## 4. Implement Code Splitting & Hashing

For larger applications, use code splitting with ESM format and cache-friendly chunk naming.

❌ BAD
```javascript
// esbuild.config.mjs
esbuild.build({
  entryPoints: ['src/app.js', 'src/admin.js'], // Multiple entries, but no splitting
  bundle: true,
  outfile: 'dist/bundle.js', // Single output file, no code splitting
});
```

✅ GOOD
```javascript
// esbuild.config.mjs
esbuild.build({
  entryPoints: ['src/app.js', 'src/admin.js'],
  bundle: true,
  splitting: true, // Enable code splitting
  format: 'esm', // Required for splitting
  outdir: 'dist',
  chunkNames: 'chunks/[name]-[hash]', // Cache-friendly chunk names
  assetNames: 'assets/[name]-[hash]', // For non-JS assets
});
```

## 5. Manage External Dependencies

When building libraries, declare common dependencies as `external` to prevent duplication and allow consumers to dedupe.

❌ BAD
```javascript
// esbuild.config.mjs (for a library)
esbuild.build({
  entryPoints: ['src/library.js'],
  bundle: true, // Bundles React into the library
  outfile: 'dist/library.js',
});
```

✅ GOOD
```javascript
// esbuild.config.mjs (for a library)
esbuild.build({
  entryPoints: ['src/library.js'],
  bundle: true,
  outfile: 'dist/library.js',
  external: ['react', 'react-dom'], // Exclude common dependencies
});
```

## 6. Use Path Aliases

Improve module resolution readability and prevent deep relative imports. Configure `tsconfig.json` and pass it to esbuild.

❌ BAD
```javascript
// src/components/Button.js
import { helperFunction } from '../../../../utils/helpers';
```

✅ GOOD
```javascript
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "~/*": ["src/*"]
    }
  }
}
// esbuild.config.mjs
esbuild.build({
  // ...
  tsconfig: 'tsconfig.json', // esbuild respects paths from tsconfig
});
// src/components/Button.js
import { helperFunction } from '~/utils/helpers';
```

## 7. Enable Strict TypeScript Integration

Always integrate your `tsconfig.json` to leverage strict type checking and ensure esbuild correctly processes TypeScript.

❌ BAD
```javascript
// esbuild.config.mjs
esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  loader: { '.ts': 'tsx' }, // Manual loader, ignores tsconfig.json
});
```

✅ GOOD
```javascript
// esbuild.config.mjs
esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  tsconfig: 'tsconfig.json', // Automatically uses tsconfig for loader and options
});
```

## 8. Optimize Production Builds

Ensure `minify: true`, `sourcemap: 'external'`, and `define` for dead code elimination.

❌ BAD
```javascript
// esbuild.config.mjs (production)
esbuild.build({
  // ...
  minify: false, // Larger bundles
  sourcemap: true, // Inline sourcemaps in production
  // Missing process.env.NODE_ENV definition
});
```

✅ GOOD
```javascript
// esbuild.config.mjs (production)
esbuild.build({
  // ...
  minify: true, // Smallest bundles
  sourcemap: 'external', // Separate sourcemap for debugging, not exposed to users
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'), // Enables dead code elimination
  },
});
```

## 9. Testing Integration

esbuild focuses on bundling, not testing. Ensure your build configuration doesn't interfere with test environments (e.g., by conditionally defining `process.env.NODE_ENV`). For testing, use dedicated test runners (e.g., Vitest, Jest) that can either process raw source files or use a separate, test-specific esbuild configuration.

```javascript
// esbuild.config.test.mjs (Example of a test-specific config)
import * as esbuild from 'esbuild';

esbuild.build({
  entryPoints: ['src/**/*.test.ts'], // Target test files
  bundle: true,
  outdir: 'test-dist',
  platform: 'node', // Or 'browser' depending on test environment
  define: {
    'process.env.NODE_ENV': JSON.stringify('test'), // Specific env for tests
  },
  // ... other test-specific configurations
}).catch(() => process.exit(1));
```