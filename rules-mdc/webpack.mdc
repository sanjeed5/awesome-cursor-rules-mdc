---
description: This guide provides opinionated, actionable best practices for configuring webpack, focusing on performance, maintainability, and modern development workflows in 2025.
globs: **/*.{js,jsx}
---
# webpack Best Practices

webpack remains the cornerstone of modern JavaScript application builds. Adhering to these definitive guidelines ensures your projects are performant, maintainable, and aligned with current best practices.

## 1. Configuration Structure

Always separate your webpack configurations by environment. Use `webpack-merge` to combine a common base with environment-specific overrides.

❌ BAD: Monolithic `webpack.config.js` with conditional logic.
```javascript
// webpack.config.js
const isProduction = process.env.NODE_ENV === 'production';

module.exports = {
  mode: isProduction ? 'production' : 'development',
  // ... lots of if/else logic
};
```

✅ GOOD: Modular configs with `webpack-merge`.
```javascript
// webpack.common.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    clean: true, // Always clean dist folder
  },
  plugins: [
    new HtmlWebpackPlugin({ template: './src/index.html' }),
  ],
  module: {
    rules: [
      {
        test: /\.js$/,
        include: path.resolve(__dirname, 'src'), // Target only your source
        loader: 'babel-loader',
      },
    ],
  },
};

// webpack.dev.js
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  mode: 'development',
  devtool: 'eval-cheap-module-source-map', // Fast source maps for dev
  output: {
    filename: '[name].bundle.js',
  },
  devServer: {
    static: './dist',
    hot: true, // Enable HMR
  },
});

// webpack.prod.js
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');
const MiniCssExtractPlugin = require('mini-css-extract-plugin'); // For production CSS

module.exports = merge(common, {
  mode: 'production',
  devtool: 'source-map', // High-quality source maps for production debugging
  output: {
    filename: '[name].[contenthash].bundle.js', // Cache busting
  },
  plugins: [
    new MiniCssExtractPlugin({
      filename: '[name].[contenthash].css',
    }),
  ],
  optimization: {
    minimize: true, // Enable minification
    splitChunks: {
      chunks: 'all', // Aggressively split chunks
    },
  },
});
```

## 2. Performance Optimizations

Prioritize build speed and bundle size from the start.

### A. Caching

Enable persistent caching for faster incremental builds.

❌ BAD: No caching, slow rebuilds.
```javascript
module.exports = { /* ... */ };
```

✅ GOOD: Filesystem caching.
```javascript
module.exports = {
  cache: {
    type: 'filesystem',
    buildDependencies: {
      config: [__filename], // Invalidate cache if config changes
    },
  },
  // ...
};
```

### B. Loaders Scope

Apply loaders to the absolute minimum set of files. Exclude `node_modules` where possible.

❌ BAD: Applying Babel to `node_modules`.
```javascript
module.exports = {
  module: {
    rules: [{ test: /\.js$/, loader: 'babel-loader' }],
  },
};
```

✅ GOOD: Explicitly `include` your source, `exclude` `node_modules`.
```javascript
const path = require('path');
module.exports = {
  module: {
    rules: [
      {
        test: /\.js$/,
        include: path.resolve(__dirname, 'src'),
        exclude: /node_modules/, // Crucial for performance
        loader: 'babel-loader',
      },
    ],
  },
};
```

### C. Code Splitting

Leverage `optimization.splitChunks` and dynamic `import()` for smaller, on-demand bundles.

❌ BAD: Single large bundle for the entire application.
```javascript
// src/index.js
import { largeModuleA } from './largeModuleA';
import { largeModuleB } from './largeModuleB';
// ...
```

✅ GOOD: Dynamic imports with `splitChunks`.
```javascript
// webpack.prod.js (part of optimization)
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all', // Automatically split vendor and common code
      minSize: 20000, // Minimum size for a chunk to be generated
      maxInitialRequests: 30, // Max number of parallel requests on initial load
      maxAsyncRequests: 30, // Max number of parallel requests for on-demand loading
      cacheGroups: {
        vendors: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: -10,
          reuseExistingChunk: true,
        },
        default: {
          minChunks: 2,
          priority: -20,
          reuseExistingChunk: true,
        },
      },
    },
    runtimeChunk: 'single', // Separate runtime manifest
  },
  // ...
};

// src/index.js
const loadLargeModuleA = () => import('./largeModuleA'); // Dynamic import
const loadLargeModuleB = () => import('./largeModuleB');
// ... use them when needed, e.g., on button click
```

## 3. Essential Plugins

Use these plugins for robust and optimized builds.

### A. `HtmlWebpackPlugin`

Always generate your `index.html` and inject bundles automatically.

❌ BAD: Manually updating `<script>` tags in `index.html`.
```html
<!-- index.html -->
<script src="./dist/main.bundle.js"></script>
```

✅ GOOD: Let `HtmlWebpackPlugin` handle it.
```javascript
// webpack.common.js
const HtmlWebpackPlugin = require('html-webpack-plugin');
module.exports = {
  plugins: [
    new HtmlWebpackPlugin({
      template: './public/index.html', // Your template HTML
      title: 'My App',
    }),
  ],
  // ...
};
```

### B. `DefinePlugin`

Inject environment variables safely and enable dead-code elimination. `mode` handles `process.env.NODE_ENV` automatically. For custom variables, use `DefinePlugin`.

❌ BAD: Hardcoding API keys or relying on client-side `process.env`.
```javascript
// src/api.js
const API_URL = 'https://dev.api.example.com'; // Changes for prod
```

✅ GOOD: Injecting via `DefinePlugin`.
```javascript
// webpack.common.js
const webpack = require('webpack');
module.exports = {
  plugins: [
    new webpack.DefinePlugin({
      'process.env.API_URL': JSON.stringify(process.env.API_URL || 'http://localhost:3000/api'),
    }),
  ],
  // ...
};

// src/api.js
const API_URL = process.env.API_URL; // Injected at build time
```

## 4. Modern Tooling & Maintenance

Stay current with webpack and Node.js versions.

### A. Keep Dependencies Updated

Regularly update `webpack`, `webpack-cli`, `webpack-dev-server`, and Node.js for performance gains and security fixes.

### B. `output.clean`

Always use `output.clean: true` to prevent stale files in your `dist` directory.

❌ BAD: Manually deleting `dist` or having old files linger.
```javascript
// package.json
"scripts": {
  "build": "rm -rf dist && webpack --config webpack.prod.js"
}
```

✅ GOOD: Built-in `clean` option.
```javascript
// webpack.common.js
module.exports = {
  output: {
    clean: true, // Automatically cleans the output directory before emit
    // ...
  },
  // ...
};
```