---
description: This guide provides opinionated, actionable best practices for developing high-performance, maintainable, and secure applications using Turbopack as the default bundler in Next.js 16+.
globs: **/*.{js,jsx}
---
# turbopack Best Practices

Turbopack is the default, Rust-based incremental bundler for Next.js 16+, offering unparalleled speed through function-level caching, lazy bundling, and a unified build graph. To fully leverage its power, adhere to these modern best practices.

## 1. Code Organization and Structure

Turbopack thrives on a well-structured project. Standardize your directory layout to maximize its incremental compilation benefits.

### 1.1 Standardized Directory Structure
Always place application source code under `src/`. This provides a clear boundary and helps Turbopack understand your project's scope.

❌ BAD: Scattered files
```jsx
// project-root/
// ├── index.js
// ├── components/Button.jsx
// └── api/users.js
```

✅ GOOD: Centralized `src/`
```jsx
// project-root/
// └── src/
//     ├── app/          // Next.js App Router routes
//     ├── components/   // Reusable UI components
//     ├── lib/          // Utility functions and shared logic
//     ├── services/     // Data fetching and external API calls
//     ├── styles/       // Global and modular styles
//     └── types/        // TypeScript type definitions
```

### 1.2 TypeScript First
Turbopack has built-in, highly optimized support for TypeScript and JSX/TSX via SWC. Always use TypeScript to catch errors early and enhance developer experience.

❌ BAD: Plain JavaScript
```javascript
// src/lib/math.js
export function add(a, b) {
  return a + b;
}
```

✅ GOOD: Type-safe TypeScript
```typescript
// src/lib/math.ts
export function add(a: number, b: number): number {
  return a + b;
}
```

### 1.3 Component Granularity
Design components to be small, pure, and focused on a single responsibility. This maximizes Turbopack's function-level caching and improves Fast Refresh times.

❌ BAD: Large, monolithic component
```jsx
// src/components/UserProfilePage.tsx
'use client';
import { useState, useEffect } from 'react';
import { fetchUserData, updateUserProfile } from '@/services/api';

export default function UserProfilePage({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  // ... lots of state and logic for profile, orders, settings, etc.
  useEffect(() => { /* fetch data */ }, [userId]);
  const handleSubmit = () => { /* update profile */ };
  return (
    // ... massive JSX structure
  );
}
```

✅ GOOD: Small, composable components
```jsx
// src/app/dashboard/profile/page.tsx (Server Component)
import { Suspense } from 'react';
import UserProfileForm from '@/components/UserProfileForm';
import UserOrdersList from '@/components/UserOrdersList';
import { fetchUserData } from '@/services/api';

export default async function ProfilePage() {
  const user = await fetchUserData(); // Server-side data fetch
  return (
    <div className="container">
      <h1>Welcome, {user.name}</h1>
      <Suspense fallback={<p>Loading profile...</p>}>
        <UserProfileForm initialData={user} />
      </Suspense>
      <Suspense fallback={<p>Loading orders...</p>}>
        <UserOrdersList userId={user.id} />
      </Suspense>
    </div>
  );
}

// src/components/UserProfileForm.tsx (Client Component)
'use client';
import { useState } from 'react';
import { updateUserProfile } from '@/services/api';

export default function UserProfileForm({ initialData }) {
  const [name, setName] = useState(initialData.name);
  const [email, setEmail] = useState(initialData.email);

  const handleSubmit = async (e) => {
    e.preventDefault();
    await updateUserProfile({ name, email });
    alert('Profile updated!');
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
      <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <button type="submit">Save</button>
    </form>
  );
}
```

## 2. Common Patterns and Anti-patterns

Leverage Next.js 16 features that align with Turbopack's strengths, and avoid patterns that hinder its optimizations.

### 2.1 Embrace Next.js App Router and Server Components
Next.js 16 defaults to the App Router and Server Components. This is the optimal architecture for Turbopack, as it enables the unified graph and efficient server/client bundling.

❌ BAD: Sticking to Pages Router for new features
```jsx
// src/pages/dashboard.jsx
// ... uses getServerSideProps for data fetching
```

✅ GOOD: Utilize App Router with Server Components
```jsx
// src/app/dashboard/page.tsx
// This is a Server Component by default
import { fetchDashboardData } from '@/services/api';

export default async function DashboardPage() {
  const data = await fetchDashboardData();
  return (
    <div>
      <h1>Dashboard</h1>
      <p>Data: {data.summary}</p>
    </div>
  );
}
```

### 2.2 Explicit Runtime and Caching Directives
Guide Turbopack's incremental computation and caching with explicit flags.

*   `export const runtime = 'edge'` for Edge Functions.
*   `export const dynamic = 'force-static'` for full static rendering.
*   `export const revalidate = 60` for Incremental Static Regeneration (ISR).
*   `"use cache"` for fine-grained component caching (Next.js 16+).

❌ BAD: Implicit behavior, relying on defaults for critical paths
```typescript
// src/app/api/hello/route.ts (runs on Node.js by default)
export async function GET() { /* ... */ }
```

✅ GOOD: Explicitly define runtime for performance-critical APIs
```typescript
// src/app/api/hello/route.ts
export const runtime = 'edge'; // Run this API route on the Edge
export async function GET() {
  return new Response('Hello from the Edge!');
}
```
And for component caching:
```jsx
// src/components/CachedHeader.tsx
"use cache"; // Cache this component's output
export default function CachedHeader() {
  // This component will be cached by Next.js 16's Cache Components feature
  return <header>My Cached App Header</header>;
}
```

### 2.3 Intentional Code Splitting
Turbopack supports lazy bundling. Use `next/dynamic` or standard dynamic `import()` for large, non-critical components or libraries to reduce initial bundle size.

❌ BAD: Importing heavy components eagerly
```jsx
// src/components/Dashboard.tsx
import Chart from 'heavy-chart-library'; // Imported even if chart is not always visible
export default function Dashboard() {
  return <Chart data={...} />;
}
```

✅ GOOD: Dynamically import components
```jsx
// src/components/Dashboard.tsx
import dynamic from 'next/dynamic';

const Chart = dynamic(() => import('heavy-chart-library'), {
  ssr: false, // Only load on client-side
  loading: () => <p>Loading chart...</p>,
});

export default function Dashboard() {
  return (
    <div>
      <h1>Dashboard Overview</h1>
      <Chart data={...} />
    </div>
  );
}
```

## 3. Performance Considerations

Turbopack is built for speed. Align your code with its incremental nature.

### 3.1 Maximize Function-Level Caching
Design modules with small, pure functions. Turbopack caches results at this granular level, significantly speeding up rebuilds when only small parts of the codebase change.

❌ BAD: Functions with side effects or large dependencies
```typescript
// src/lib/dataProcessor.ts
let globalConfig = {}; // Mutated elsewhere
export function processData(data) {
  // Depends on and modifies globalConfig
  // ... complex logic ...
  return processed;
}
```

✅ GOOD: Pure, isolated functions
```typescript
// src/lib/dataProcessor.ts
export function processData(data: any[], config: any): any[] {
  // Pure function, depends only on its inputs
  // ... complex logic ...
  return processed;
}
```

### 3.2 Scope CSS and Assets
Import CSS and other assets directly within the components that use them. This allows Turbopack to effectively lazy bundle and optimize asset loading. Use CSS Modules for component-scoped styles.

❌ BAD: Global CSS imports for component-specific styles
```jsx
// src/app/layout.tsx
import '../styles/button.css'; // Global import for a specific button style

export default function RootLayout({ children }) { /* ... */ }
```

✅ GOOD: CSS Modules for component-specific styles
```css
/* src/components/MyButton.module.css */
.primary {
  background-color: blue;
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
}
```
```jsx
// src/components/MyButton.tsx
import styles from './MyButton.module.css';

export default function MyButton() {
  return <button className={styles.primary}>Click me</button>;
}
```

### 3.3 Leverage Next.js 16 Cache Components
Enable `cacheComponents: true` in `next.config.ts` and use the `"use cache"` directive to cache component outputs. This is a powerful feature for instant navigation with Partial Pre-rendering (PPR).

```typescript
// next.config.ts
const nextConfig = {
  cacheComponents: true, // Enable Next.js 16 Cache Components
};
export default nextConfig;
```
Then, in your components:
```jsx
// src/components/ProductCard.tsx
"use cache"; // This component's output will be cached
import Image from 'next/image';

export default function ProductCard({ product }) {
  return (
    <div className="product-card">
      <Image src={product.image} alt={product.name} width={200} height={200} />
      <h3>{product.name}</h3>
      <p>${product.price}</p>
    </div>
  );
}
```

## 4. Common Pitfalls and Gotchas

Be aware of these common issues when working with Turbopack.

### 4.1 Avoid Direct Webpack Configuration
Turbopack aims to abstract away Webpack. Do not attempt to configure Webpack directly when using Turbopack, as it will be ignored or lead to unexpected behavior. Configure Turbopack via `next.config.js` options where available.

❌ BAD: Attempting to modify Webpack config directly
```javascript
// next.config.js
module.exports = {
  webpack: (config, { isServer }) => {
    // This will be ignored by Turbopack
    config.plugins.push(new MyWebpackPlugin());
    return config;
  },
};
```

✅ GOOD: Use Next.js 16's `next.config.js` options for Turbopack
```javascript
// next.config.js
const nextConfig = {
  // Turbopack-specific configurations are exposed here
  // e.g., for path aliases or root directory adjustments
  turbopack: {
    resolveAlias: {
      '@my-custom-alias': '/path/to/custom/module',
    },
    root: '../../', // For monorepos with linked dependencies outside project root
  },
};
module.exports = nextConfig;
```

### 4.2 Filesystem Root for Monorepos
If you use linked dependencies (e.g., `npm link`, `yarn link`, `pnpm link`) that reside outside your project's root, Turbopack will not resolve them by default. Adjust the `turbopack.root` option in `next.config.js`.

❌ BAD: Linked dependencies not resolving
```
// project-root/
// └── src/
//     └── app/page.tsx
//
// sibling-package/ (linked via npm link)
// └── index.ts
```
```typescript
// src/app/page.tsx
import { someFunction } from 'sibling-package'; // Fails to resolve
```

✅ GOOD: Configure `turbopack.root`
```javascript
// next.config.js (assuming project-root and sibling-package are in the same parent directory)
const nextConfig = {
  turbopack: {
    root: '../../', // Adjusts the root to include parent directory
  },
};
module.exports = nextConfig;
```

### 4.3 Sass Functions
Turbopack's Rust-based architecture does not support custom Sass functions (`sassOptions.functions`) that rely on JavaScript execution. If you need this, you *must* opt out of Turbopack for that specific project using `--webpack`.

❌ BAD: Custom Sass functions with Turbopack
```javascript
// next.config.js
module.exports = {
  sassOptions: {
    functions: {
      'my-custom-func($value)': (value) => { /* JS logic */ },
    },
  },
};
```

✅ GOOD: Avoid custom Sass functions or use Webpack explicitly
```bash
# If you absolutely need custom Sass functions
next dev --webpack
```
Otherwise, refactor your Sass to avoid custom JS functions.

## 5. Testing Approaches

Integrate Turbopack into your testing pipeline to ensure compiled output behaves as expected.

### 5.1 Unit and Integration Tests
Run your tests against the compiled output. Turbopack provides built-in source map support for debugging, making it easier to pinpoint issues in your original source code.

```json
// package.json
{
  "scripts": {
    "test": "jest --passWithNoTests",
    "test:watch": "jest --watch",
    "build": "next build"
  }
}
```
Ensure your testing setup (e.g., Jest, Vitest) is configured to handle TypeScript and JSX/TSX, which Turbopack processes.

### 5.2 End-to-End (E2E) Testing
Always run E2E tests against the Turbopack-bundled production build (`next build` then `next start`). This validates the entire application flow, including how assets are loaded and code is split by Turbopack.

```json
// package.json
{
  "scripts": {
    "build": "next build",
    "start": "next start",
    "e2e": "playwright test"
  }
}
```
Run `npm run build && npm run start` in a CI environment, then execute your Playwright or Cypress tests against the running application.

By adhering to these guidelines, your team will build high-performance, maintainable, and secure Next.js applications that fully leverage the power of Turbopack.