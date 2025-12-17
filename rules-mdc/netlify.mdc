---
description: This guide provides opinionated, actionable best practices for building, deploying, and maintaining high-performance, secure, and scalable Jamstack applications on Netlify. It focuses on modern workflows, code quality, and leveraging Netlify's edge-first architecture.
globs: **/*
---
# netlify Best Practices

Netlify is the definitive platform for the Jamstack. Our workflow prioritizes pre-built, edge-first deployments, ensuring maximum performance, security, and developer experience. These guidelines enforce a consistent, high-quality approach to building for the modern web on Netlify.

## 1. Code Organization and Build Configuration

Always centralize your build logic and deployment settings in `netlify.toml`. This ensures consistency and reproducibility across environments and team members.

### ✅ GOOD: Standardized `netlify.toml`

```toml
# netlify.toml
[build]
  command = "npm run build" # Your project's build command
  publish = "dist"          # Directory containing your built static assets
  functions = "netlify/functions" # Directory for Netlify Edge/Serverless Functions

# Essential redirects for SPAs or SEO
[[redirects]]
  from = "/old-path"
  to = "/new-path"
  status = 301

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200 # For Single Page Applications (SPAs)

# Security headers for all responses
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "no-referrer-when-downgrade"
    Content-Security-Policy = "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';"
    Strict-Transport-Security = "max-age=63072000; includeSubDomains; preload"
```

### ❌ BAD: Relying on UI settings or missing critical configurations

```toml
# netlify.toml (incomplete)
[build]
  command = "npm run build"
  publish = "build"
# Missing redirects, functions path, and crucial security headers.
# This leads to inconsistent deployments, SEO issues, and security vulnerabilities.
```

## 2. Environment Variables and Secrets Management

Never hardcode sensitive information. Use Netlify's encrypted environment variables for all secrets. Differentiate between build-time and runtime variables.

### ✅ GOOD: Secure Environment Variables

```javascript
// netlify/functions/api.js
import fetch from 'node-fetch';

export async function handler(event, context) {
  const API_KEY = process.env.MY_EXTERNAL_API_KEY; // Fetched securely at runtime
  if (!API_KEY) {
    return { statusCode: 500, body: 'API Key not configured.' };
  }
  // ... use API_KEY
}
```

```bash
# In Netlify UI or netlify.toml (for build-time only, generally prefer UI for secrets)
# netlify.toml
[build.environment]
  # Only for variables needed during the build process, e.g., for a static site generator
  GATSBY_API_URL = "https://api.example.com/public"
```

### ❌ BAD: Hardcoding secrets or exposing them in client-side bundles

```javascript
// src/components/MyComponent.js
const API_KEY = "sk_YOUR_SECRET_KEY_HERE"; // NEVER DO THIS!
// This key will be bundled with your client-side code and exposed.
```

## 3. Performance Optimization

Leverage Netlify's edge network and build process for maximum speed.

### ✅ GOOD: Image Optimization with Build Plugins

Use Netlify Build Plugins for automated image optimization.

```toml
# netlify.toml
[[plugins]]
  package = "@netlify/plugin-nextjs" # Example for Next.js, includes image optimization
  # Or a dedicated image optimization plugin:
  # package = "@netlify/plugin-image-optim"
```

### ❌ BAD: Serving unoptimized, large images

```html
<!-- src/index.html -->
<img src="/images/hero-lg.jpg" alt="Hero" />
<!-- This image is likely too large, not responsive, and not optimized for web. -->
```

## 4. Edge Functions for Dynamic Logic

Move lightweight server-side logic to the edge with Netlify Edge Functions for faster response times and reduced latency.

### ✅ GOOD: Edge Function for A/B Testing or Geo-targeting

```typescript
// netlify/edge-functions/geo-redirect.ts
import type { Context } from "https://edge.netlify.com/";

export default async (request: Request, context: Context) => {
  const country = context.geo?.country?.name || "Unknown";

  if (country === "Germany") {
    return Response.redirect(new URL("/de", request.url));
  }
  // Continue to the original page
  return context.next();
};
```

### ❌ BAD: Relying on traditional serverless functions for every request

```javascript
// netlify/functions/slow-geo-check.js (traditional serverless function)
exports.handler = async (event, context) => {
  // This will incur higher latency than an Edge Function for simple routing logic.
  const ip = event.headers['x-forwarded-for'];
  // ... complex IP-to-geo lookup (slow)
  return { statusCode: 200, body: `Hello from ${country}` };
};
```

## 5. Testing and CI/CD Hygiene

Integrate automated testing and quality checks into your Netlify build pipeline.

### ✅ GOOD: Automated Linting, Type Checking, and E2E Tests

Ensure your `package.json` scripts support these, and Netlify's build command triggers them.

```json
// package.json
{
  "scripts": {
    "build": "npm run lint && npm run typecheck && next build",
    "lint": "eslint . --ext .js,.jsx,.ts,.tsx",
    "typecheck": "tsc --noEmit",
    "test:e2e": "cypress run"
  }
}
```

```toml
# netlify.toml
[build]
  command = "npm run build" # This will run lint and typecheck before the actual build
  publish = "out"

[context.deploy-preview]
  command = "npm run build && npm run test:e2e" # Run E2E tests on deploy previews
```

### ❌ BAD: Skipping quality checks in CI

```toml
# netlify.toml
[build]
  command = "next build" # No linting, type checking, or tests before deployment.
  publish = "out"
# This allows broken code to reach deploy previews or even production.
```

## 6. Accessibility

Accessibility is a non-negotiable part of modern web development. Integrate tools to enforce it.

### ✅ GOOD: Automated Accessibility Checks

Use tools like Lighthouse CI or Pa11y in your build process.

```json
// package.json
{
  "scripts": {
    "build": "next build",
    "audit:a11y": "lighthouse ci --config=./.lighthouseci.json"
  }
}
```

```toml
# netlify.toml
[build]
  command = "npm run build"
  publish = "out"

[context.production]
  command = "npm run build && npm run audit:a11y" # Run a11y audit on production builds
```

### ❌ BAD: Ignoring accessibility until manual review

```html
<!-- src/components/ImageGallery.js -->
<img src="image.jpg" />
<!-- Missing `alt` attribute, leading to accessibility issues. -->
```

## 7. Decoupled Architecture and API-First Development

Embrace the Jamstack philosophy: frontend decoupled from backend. Interact with services via APIs.

### ✅ GOOD: Consuming external APIs

```javascript
// src/utils/dataService.js
export async function fetchProducts() {
  const response = await fetch('/.netlify/functions/getProducts'); // Call a Netlify Function acting as an API gateway
  if (!response.ok) {
    throw new Error('Failed to fetch products');
  }
  return response.json();
}
```

### ❌ BAD: Tightly coupled frontend and backend

```javascript
// src/server.js (monolithic approach)
// This file would not run on Netlify's static hosting.
// Attempting to deploy a traditional server-side app directly to Netlify will fail.
const express = require('express');
const app = express();
// ... server-side logic
```