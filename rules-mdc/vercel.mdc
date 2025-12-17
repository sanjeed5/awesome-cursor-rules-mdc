---
description: This guide outlines definitive best practices for developing and deploying applications on Vercel, ensuring optimal performance, security, and cost-efficiency.
globs: **/*
---
# vercel Best Practices

Vercel is the definitive platform for deploying modern web applications, especially those built with Next.js. To maximize its potential, adhere to these opinionated guidelines for performance, security, and maintainability.

## 1. Code Organization & Structure

Organize your project to leverage Vercel's serverless functions and build optimizations.

### 1.1. Serverless Functions in `api/`

All serverless functions must reside within the `api/` directory (or `app/api` for App Router). This is Vercel's convention for automatic API route detection.

❌ **BAD: Mixing serverless logic outside `api/`**
```javascript
// pages/products.js
export default function Products() {
  // ... client-side logic
}

// In some_utility.js
export async function fetchProductsData() {
  // This function contains server-side logic and heavy dependencies
  // but is not in a dedicated serverless function file.
  const data = await expensiveDatabaseCall();
  return data;
}
```

✅ **GOOD: Centralized serverless functions**
```javascript
// pages/products.js (or app/products/page.tsx)
import useSWR from 'swr';

export default function Products() {
  const { data, error } = useSWR('/api/products', fetcher);
  // ... client-side rendering
}

// api/products.js (or app/api/products/route.ts)
import { NextResponse } from 'next/server';
import { expensiveDatabaseCall } from '@/lib/db'; // Separate DB logic

export const runtime = 'edge'; // Use Edge Runtime for speed

export async function GET() {
  try {
    const data = await expensiveDatabaseCall();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch products' }, { status: 500 });
  }
}
```

### 1.2. Monorepo Caching with Turborepo

For monorepos, always integrate Turborepo to enable intelligent caching and prevent redundant builds. Vercel automatically detects and optimizes for Turborepo.

❌ **BAD: Monorepo without Turborepo caching**
```bash
# In a large monorepo, every Vercel build will re-build all packages
# even if only one has changed.
npm run build # Rebuilds everything
```

✅ **GOOD: Turborepo for efficient builds**
```json
// package.json (root)
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": ["apps/*", "packages/*"],
  "scripts": {
    "build": "turbo run build",
    "dev": "turbo run dev --parallel"
  },
  "devDependencies": {
    "turbo": "latest"
  }
}
```

## 2. Common Patterns & Anti-patterns

Leverage Vercel's strengths, particularly Edge Functions and Fluid Compute, while avoiding common performance traps.

### 2.1. Prefer Edge Runtime & Fluid Compute

For serverless functions, prioritize the Edge Runtime and ensure Fluid Compute is enabled for automatic cold start optimizations and cost efficiency.

❌ **BAD: Default Node.js runtime for latency-sensitive functions**
```javascript
// api/heavy-computation.js
// This will run in a traditional Node.js Lambda, potentially incurring cold starts.
export default async function handler(req, res) {
  // ... heavy logic
}
```

✅ **GOOD: Edge Runtime with Fluid Compute**
```javascript
// api/fast-response.js
export const runtime = 'edge'; // Explicitly set Edge Runtime
export const preferredRegion = 'iad1'; // Choose region closest to users/data

export default async function handler(req) {
  // Fluid Compute automatically handles cold starts and resource scaling.
  // ... lightweight, latency-sensitive logic
  return new Response('Hello from the Edge!');
}
```

### 2.2. Static Exports for Purely Static Sites

If your Next.js application is purely static (no server-side rendering or API routes), use `output: 'export'` for maximum performance and cost savings.

❌ **BAD: Deploying a static site without `output: 'export'`**
```javascript
// next.config.js
const nextConfig = {}; // Default configuration
module.exports = nextConfig; // Still deploys as a hybrid app, incurring serverless overhead
```

✅ **GOOD: Static export for static sites**
```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // Generates static HTML, CSS, JS
  images: {
    unoptimized: true, // Required for static export if using Next/image
  },
  trailingSlash: true, // Consistent URLs for static exports
};
module.exports = nextConfig;
```

## 3. Performance Considerations

Vercel excels at performance. Ensure your code is optimized to take full advantage.

### 3.1. Minimize Serverless Function Bundle Size

Large dependencies in serverless functions directly increase cold start times. Keep functions lean.

❌ **BAD: Importing large libraries into serverless functions**
```javascript
// api/data-processor.js
import { giantLibrary } from 'giant-library'; // ~5MB dependency

export default async function handler(req, res) {
  const result = giantLibrary.process(req.body);
  res.status(200).json(result);
}
```

✅ **GOOD: Keep functions small and use dynamic imports**
```javascript
// api/data-processor.js
// Only import what's absolutely necessary.
// Consider moving heavy processing to a dedicated service or a different function.

export default async function handler(req, res) {
  // If a large dependency is unavoidable, consider dynamic import for divergent paths
  if (req.query.type === 'special') {
    const { giantLibrary } = await import('giant-library');
    const result = giantLibrary.process(req.body);
    return res.status(200).json(result);
  }
  // ... lighter logic
  return res.status(200).json({ message: 'Processed lightly' });
}
```

### 3.2. Implement Caching Headers

Utilize `Cache-Control` headers for API routes and static assets to leverage Vercel's global CDN.

❌ **BAD: No caching headers for static data**
```javascript
// api/cached-data.js
export default async function handler(req, res) {
  const data = await fetchData();
  res.status(200).json(data); // Default no-cache
}
```

✅ **GOOD: Aggressive caching for static/immutable data**
```javascript
// api/cached-data.js
export default async function handler(req, res) {
  const data = await fetchData();
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate'); // Cache for 1 hour at edge
  res.status(200).json(data);
}
```

### 3.3. Optimize Images with Next.js Image Component

Always use the Next.js `Image` component for image optimization. Vercel handles the resizing and serving via its CDN. Be aware of the image optimization pricing for older teams.

❌ **BAD: Standard `<img>` tags for dynamic images**
```html
<img src="/my-image.jpg" alt="My Image" width="800" height="600" />
```

✅ **GOOD: Next.js Image component for optimized images**
```jsx
import Image from 'next/image';

<Image
  src="/my-image.jpg"
  alt="My Image"
  width={800}
  height={600}
  quality={80} // Adjust quality as needed
  priority // For LCP images
/>
```

## 4. Common Pitfalls & Gotchas

Avoid common mistakes that lead to unexpected behavior or security vulnerabilities.

### 4.1. Environment Variables Scoping

Always scope environment variables correctly (Production, Preview, Development) in the Vercel dashboard. Never commit sensitive keys to your repository.

❌ **BAD: Hardcoding API keys or incorrect scoping**
```javascript
// api/external-service.js
const API_KEY = 'sk_hardcoded_key'; // NEVER do this!
// ... or setting a variable only for Production when needed in Preview
```

✅ **GOOD: Use Vercel's UI for environment variables**
```javascript
// api/external-service.js
// Access via process.env
const API_KEY = process.env.EXTERNAL_SERVICE_API_KEY;

// In Vercel Dashboard: Project Settings -> Environment Variables
// Add EXTERNAL_SERVICE_API_KEY for Development, Preview, and Production environments.
```

### 4.2. Missing Security Headers & CSP

Implement robust security headers, especially a Content Security Policy (CSP), to protect against XSS and other attacks.

❌ **BAD: Relying on default browser security**
```javascript
// next.config.js
// No security headers configured.
const nextConfig = {};
module.exports = nextConfig;
```

✅ **GOOD: Configure security headers in `next.config.js`**
```javascript
// next.config.js
const nextConfig = {
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';",
          },
        ],
      },
    ];
  },
};
module.exports = nextConfig;
```

## 5. Deployment Workflow & CI/CD

Streamline your deployment process using Vercel's native Git integration and CLI.

### 5.1. Leverage Preview Deployments

Always use Vercel's automatic Preview Deployments for every pull request. This enables rapid feedback and testing before merging to `main`.

❌ **BAD: Merging directly to `main` without previewing**
```bash
git push origin main # Directly pushes to production, risking bugs
```

✅ **GOOD: Review changes on a preview deployment**
```bash
# 1. Create a feature branch
git checkout -b feature/new-feature
# 2. Push to GitHub, Vercel creates a preview deployment
git push origin feature/new-feature
# 3. Open a Pull Request, share the Vercel preview URL with teammates for review.
# 4. Once approved and tested on preview, merge PR to `main`.
#    Vercel automatically deploys to production.
```

### 5.2. Use `vercel --prod` for Manual Production Deploys

While Git integration is primary, use `vercel --prod` via CLI for controlled production deployments, especially in CI/CD pipelines.

❌ **BAD: Deploying to production via `vercel` without `--prod`**
```bash
vercel # Deploys to a preview URL, not production
```

✅ **GOOD: Explicit production deployment**
```bash
# In your CI/CD pipeline after tests pass:
vercel --prod --token=$VERCEL_TOKEN # Deploys the current branch to production
```