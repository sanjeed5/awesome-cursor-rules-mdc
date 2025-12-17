---
description: This guide defines definitive best practices for Next.js applications, focusing on the `app` directory, Server Components, performance, and maintainability. Follow these rules to build robust, scalable, and performant Next.js projects.
globs: **/*.{js,jsx,ts,tsx}
---
# next-js Best Practices

This document outlines the definitive best practices for developing Next.js applications. Adhering to these guidelines ensures consistent, performant, and maintainable code, leveraging Next.js's strengths for modern web development.

## 1. Code Organization and Structure

Always use the `app/` directory for new projects. Organize code by feature, not by type, to improve discoverability and cohesion.

### ✅ GOOD: Feature-Driven `app/` Directory

Group all related files for a feature (components, pages, layouts, hooks, types) within a single directory.

```tsx
// app/dashboard/page.tsx
export default function DashboardPage() { /* ... */ }

// app/dashboard/layout.tsx
export default function DashboardLayout({ children }) { /* ... */ }

// app/dashboard/components/DashboardOverview.tsx
export function DashboardOverview() { /* ... */ }

// app/dashboard/hooks/useDashboardData.ts
export function useDashboardData() { /* ... */ }
```

### ❌ BAD: Type-Driven `app/` Directory

Avoid scattering files of the same feature across different top-level type directories.

```tsx
// app/dashboard/page.tsx
// components/dashboard/DashboardOverview.tsx // Separated
// hooks/useDashboardData.ts // Separated
```

### Core Directories

*   **`app/`**: All route-related files (`page.tsx`, `layout.tsx`, `loading.tsx`, `error.tsx`, `route.ts`).
*   **`components/`**: Reusable UI components that are *not* directly tied to a specific route.
*   **`lib/`**: Backend-agnostic utility functions, data access layers, and third-party integrations.
*   **`hooks/`**: Custom React hooks for reusable logic.
*   **`types/`**: Global TypeScript type definitions and interfaces.
*   **`public/`**: Static assets (images, fonts) that are served directly.

## 2. Component Architecture: Server Components First

Prioritize Server Components for all UI rendering. Use Client Components only when interactivity (state, effects, event handlers) is strictly required.

### ✅ GOOD: Server Component by Default

Server Components reduce client-side JavaScript bundles, improve initial page load, and enhance security by keeping sensitive logic on the server.

```tsx
// app/products/[id]/page.tsx (Server Component by default)
import { getProductDetails } from '@/lib/api';

export default async function ProductPage({ params }) {
  const product = await getProductDetails(params.id);
  return (
    <div>
      <h1>{product.name}</h1>
      {/* ... more server-rendered UI */}
      <AddToCartButton productId={product.id} /> {/* Client Component */}
    </div>
  );
}
```

### ✅ GOOD: "use client" Boundary as Low as Possible

Place the `"use client"` directive at the lowest possible point in your component tree. This ensures that only the interactive parts are client-rendered, keeping parent components as Server Components.

```tsx
// components/AddToCartButton.tsx
'use client'; // Only this component and its children are client-side

import { useState } from 'react';

export function AddToCartButton({ productId }) {
  const [quantity, setQuantity] = useState(1);
  // ... interactive logic
  return <button onClick={() => alert(`Added ${quantity} of ${productId}`)}>Add to Cart</button>;
}
```

### ❌ BAD: Overuse of "use client"

Don't mark entire feature folders or layouts as client components if only a small part needs interactivity. This unnecessarily increases client bundle size.

```tsx
// app/products/[id]/page.tsx (BAD: entire page marked client)
'use client'; // This makes the whole page a client component
import { useState, useEffect } from 'react'; // Even if only a small part needs it

export default function ProductPage({ params }) {
  // ...
  return <button>Add to Cart</button>;
}
```

## 3. Data Fetching

Fetch data directly in Server Components using `fetch` or a dedicated data access layer. Use Route Handlers for client-side mutations or when exposing a specific API endpoint.

### ✅ GOOD: Server Component Data Fetching

Directly fetch data in Server Components. `fetch` requests are automatically memoized and cached by Next.js.

```tsx
// app/dashboard/page.tsx
import { getUserProfile, getRecentOrders } from '@/lib/api';

export default async function DashboardPage() {
  // Data fetches in parallel
  const [user, orders] = await Promise.all([
    getUserProfile(),
    getRecentOrders(),
  ]);

  return (
    <div>
      <h2>Welcome, {user.name}</h2>
      <OrderList orders={orders} />
    </div>
  );
}
```

### ✅ GOOD: Route Handlers for Client-Side Mutations

Use `route.ts` for API endpoints that handle client-side data mutations (e.g., form submissions, API calls from client components).

```tsx
// app/api/cart/route.ts
import { NextResponse } from 'next/server';
import { addToCart } from '@/lib/cart';

export async function POST(request: Request) {
  const { productId, quantity } = await request.json();
  await addToCart(productId, quantity);
  return NextResponse.json({ success: true });
}
```

### ✅ GOOD: Streaming with `loading.tsx` and `Suspense`

Improve perceived performance by showing instant loading states for slow data fetches.

```tsx
// app/dashboard/loading.tsx
export default function Loading() {
  return <div>Loading dashboard...</div>;
}

// app/dashboard/page.tsx (assuming some slow component)
import { Suspense } from 'react';
import { SlowComponent } from './components/SlowComponent';

export default async function DashboardPage() {
  return (
    <main>
      <h1>Dashboard</h1>
      <Suspense fallback={<div>Loading slow data...</div>}>
        <SlowComponent />
      </Suspense>
    </main>
  );
}
```

## 4. Performance Considerations

Leverage Next.js's built-in optimizations for images, fonts, and code splitting.

### ✅ GOOD: `next/image` for Images

Always use `next/image` for local and remote images. It provides automatic optimization, lazy loading, and responsive sizing.

```tsx
import Image from 'next/image';
import profilePic from '@/public/profile.jpg';

export function UserAvatar() {
  return (
    <Image
      src={profilePic}
      alt="User Profile"
      width={100}
      height={100}
      placeholder="blur"
    />
  );
}
```

### ❌ BAD: Native `<img>` Tag

Avoid the native `<img>` tag as it bypasses Next.js's image optimizations.

```tsx
// ❌ BAD
export function UserAvatar() {
  return <img src="/profile.jpg" alt="User Profile" width="100" height="100" />;
}
```

### ✅ GOOD: `next/font` for Fonts

Use `next/font` to optimize font loading, eliminate external network requests, and prevent layout shift.

```tsx
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body>{children}</body>
    </html>
  );
}
```

### ✅ GOOD: Dynamic Imports for Heavy Components

Lazily load heavy client-side components or third-party libraries using `next/dynamic`.

```tsx
import dynamic from 'next/dynamic';

const HeavyChart = dynamic(() => import('./components/HeavyChart'), {
  loading: () => <p>Loading chart...</p>,
  ssr: false, // Only load on client if not needed for initial render
});

export function DashboardCharts() {
  return (
    <div>
      <HeavyChart />
    </div>
  );
}
```

## 5. State Management

Keep state local where possible. For global state, use React Context for simple cases or lightweight libraries like Zustand/Jotai for more complex needs.

### ✅ GOOD: Local State with `useState`

For component-specific, ephemeral state.

```tsx
'use client';
import { useState } from 'react';

export function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;
}
```

### ✅ GOOD: Global State with Context or Zustand/Jotai

For application-wide state that needs to be shared across many components. Prefer Zustand or Jotai over Redux for most Next.js projects due to their simplicity and performance.

```tsx
// lib/store.ts (using Zustand)
import { create } from 'zustand';

interface BearState {
  bears: number;
  increasePopulation: () => void;
}

export const useBearStore = create<BearState>((set) => ({
  bears: 0,
  increasePopulation: () => set((state) => ({ bears: state.bears + 1 })),
}));
```

## 6. Error Handling

Implement robust error handling using Next.js's dedicated error files.

### ✅ GOOD: Route-Level `error.tsx`

Catch errors within a specific route segment, providing localized fallback UI.

```tsx
// app/dashboard/error.tsx
'use client'; // Error boundaries must be client components

import { useEffect } from 'react';

export default function Error({ error, reset }) {
  useEffect(() => {
    console.error(error); // Log the error to an error reporting service
  }, [error]);

  return (
    <div>
      <h2>Something went wrong in the dashboard!</h2>
      <button onClick={() => reset()}>Try again</button>
    </div>
  );
}
```

### ✅ GOOD: Global `global-error.tsx`

Catch uncaught errors across your entire application, providing a consistent fallback.

```tsx
// app/global-error.tsx
'use client';

export default function GlobalError({ error, reset }) {
  return (
    <html>
      <body>
        <h2>Something went wrong globally!</h2>
        <button onClick={() => reset()}>Try again</button>
      </body>
    </html>
  );
}
```

### ✅ GOOD: `not-found.tsx` for 404s

Create a custom 404 page for unmatched routes.

```tsx
// app/not-found.tsx
export default function NotFound() {
  return (
    <div>
      <h1>404 - Page Not Found</h1>
      <p>The page you are looking for does not exist.</p>
    </div>
  );
}
```

## 7. ESLint and Type Checking

Enforce code quality and catch common issues early with ESLint and TypeScript.

### ✅ GOOD: Use `eslint-config-next/core-web-vitals`

This configuration elevates performance-related warnings to errors, ensuring your application meets Core Web Vitals standards. Always combine with `eslint-config-next/typescript` for TypeScript projects.

```javascript
// eslint.config.mjs
import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTs from 'eslint-config-next/typescript'; // For TypeScript projects

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs, // Include for TypeScript
  {
    rules: {
      // Custom overrides or additional rules here
      '@next/next/no-img-element': 'error', // Enforce next/image
      // ...
    },
  },
  globalIgnores([
    '.next/**',
    'out/**',
    'build/**',
    'next-env.d.ts',
  ]),
]);

export default eslintConfig;
```

## 8. Common Pitfalls and Anti-patterns

Avoid these common mistakes to maintain a high-quality Next.js application.

### ❌ BAD: Using `<a>` for Internal Navigation

This bypasses Next.js's automatic prefetching and client-side navigation.

```tsx
// ❌ BAD
<a href="/dashboard">Go to Dashboard</a>
```

### ✅ GOOD: Use `<Link>` for Internal Navigation

Enables client-side navigation and prefetching for a smoother user experience.

```tsx
import Link from 'next/link';

// ✅ GOOD
<Link href="/dashboard">Go to Dashboard</Link>
```

### ❌ BAD: Async Client Components

Client Components cannot be `async`. If you need to fetch data on the client, use `useEffect` or a client-side data fetching library.

```tsx
// components/MyClientComponent.tsx
'use client';

// ❌ BAD: Client Components cannot be async
export default async function MyClientComponent() {
  // const data = await fetch('/api/data');
  return <div>Client UI</div>;
}
```

### ✅ GOOD: Client-Side Data Fetching in Client Components

Use `useEffect` or a dedicated client-side library (like SWR or React Query) for data fetching in Client Components.

```tsx
// components/MyClientComponent.tsx
'use client';
import { useState, useEffect } from 'react';

export default function MyClientComponent() {
  const [data, setData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      const res = await fetch('/api/data');
      const json = await res.json();
      setData(json);
    }
    fetchData();
  }, []);

  return <div>{data ? `Data: ${data.message}` : 'Loading...'}</div>;
}
```