---
description: Definitive guidelines for building scalable, performant, and maintainable SvelteKit applications using modern best practices, TypeScript, and Svelte 5 runes.
globs: **/*.{js,jsx,ts,tsx,svelte}
---
# SvelteKit Best Practices

This guide outlines the definitive best practices for developing SvelteKit applications. Adhere to these principles for consistent, high-quality, and performant code.

## 1. Code Organization and Structure

Maintain a clean, scalable project structure.

*   **`src/routes/`**: All page and API endpoints. Co-locate `+page.svelte`, `+page.ts`, `+layout.svelte`, `+error.svelte`, `+server.ts` files.
*   **`src/lib/`**: Reusable utilities, components, and client-side logic. Imported via `$lib`.
*   **`src/lib/server/`**: Server-only modules. Imported via `$lib/server`. SvelteKit prevents client-side import.
*   **`src/params/`**: Custom route matchers.
*   **`static/`**: Static assets (e.g., `favicon.png`, `robots.txt`).
*   **`tests/`**: End-to-end (E2E) tests (e.g., Playwright). Unit tests can be co-located with `src/`.

**Rule**: Use `+` prefixed filenames for SvelteKit's routing and special files.

❌ BAD:
```
src/routes/users/index.svelte
src/routes/api/users.ts
```

✅ GOOD:
```svelte
src/routes/users/+page.svelte
src/routes/api/users/+server.ts
```

## 2. Component Architecture

Design components for reusability and clarity.

**Rule**: Use `lang="ts"` for TypeScript in Svelte components and leverage `$props()` for type-safe props.

```svelte
<!-- src/lib/components/Button.svelte -->
<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { Snippet } from 'svelte';

  interface Props extends HTMLButtonAttributes {
    variant?: 'primary' | 'secondary';
    children: Snippet;
  }

  let { variant = 'primary', children, ...rest }: Props = $props();
</script>

<button class:primary={variant === 'primary'} class:secondary={variant === 'secondary'} {...rest}>
  {@render children()}
</button>

<style>
  button { /* ... base styles ... */ }
  .primary { /* ... primary styles ... */ }
  .secondary { /* ... secondary styles ... */ }
</style>
```

## 3. State Management

Prioritize Svelte 5 runes for local reactivity and classes for complex, encapsulated state.

**Rule**: Use `$state()` for local, reactive component state. For complex logic or shared state, encapsulate it within a class.

❌ BAD (Old Svelte 4 store for local state):
```svelte
<script lang="ts">
  import { writable } from 'svelte/store';
  const count = writable(0);
</script>

<button on:click={() => count.update(n => n + 1)}>Count: {$count}</button>
```

✅ GOOD (Svelte 5 runes for local state):
```svelte
<script lang="ts">
  let count = $state(0);
</script>

<button on:click={() => count++}>Count: {count}</button>
```

✅ GOOD (Class for complex/shared state):
```typescript
// src/lib/stores/counter.ts
class CounterStore {
  count = $state(0);
  increment() {
    this.count++;
  }
  reset() {
    this.count = 0;
  }
}
export const counter = new CounterStore(); // Singleton instance
```
```svelte
<!-- src/routes/+page.svelte -->
<script lang="ts">
  import { counter } from '$lib/stores/counter';
</script>

<button on:click={() => counter.increment()}>Count: {counter.count}</button>
```

## 4. Routing and Data Fetching

Leverage SvelteKit's file-system routing and `load` functions.

**Rule**: Use `+page.ts` for universal `load` functions (SSR + CSR) and `+page.server.ts` for server-only `load` functions or form actions. Always type `load` functions with `$types`.

```typescript
// src/routes/blog/[slug]/+page.ts
import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
  const res = await fetch(`/api/blog/${params.slug}`);
  if (!res.ok) {
    error(res.status, 'Could not fetch blog post.');
  }
  const post = await res.json();
  return { post };
};
```
```svelte
<!-- src/routes/blog/[slug]/+page.svelte -->
<script lang="ts">
  import type { PageProps } from './$types';
  let { data }: PageProps = $props();
</script>

<h1>{data.post.title}</h1>
<p>{data.post.content}</p>
```

**Rule**: For API endpoints, use `+server.ts` files.

```typescript
// src/routes/api/blog/[slug]/+server.ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params }) => {
  // In a real app, fetch from DB
  if (params.slug === 'hello-world') {
    return json({ title: 'Hello World', content: 'Welcome to our blog!' });
  }
  error(404, 'Not Found');
};
```

## 5. Performance Considerations

Optimize for speed and efficiency.

**Rule**: Prerender static routes by exporting `export const prerender = true` in `+page.ts` or `+layout.ts`. Use `@sveltejs/enhanced-img` for image optimization.

```typescript
// src/routes/about/+page.ts
export const prerender = true; // This page will be statically generated at build time
```

**Rule**: Deploy to platforms like Vercel for zero-config serverless functions and automatic optimizations.

## 6. Accessibility

Build inclusive user interfaces.

**Rule**: Always use semantic HTML, ensure keyboard navigability, and test with screen readers.

❌ BAD:
```svelte
<div on:click={doSomething}>Click me</div>
```

✅ GOOD:
```svelte
<button on:click={doSomething}>Click me</button>
```

## 7. Testing Approaches

Implement a robust testing strategy.

**Rule**: Use Playwright for end-to-end tests in `tests/` and Vitest for unit tests, co-located with the code they test. Integrate `svelte-check`, ESLint, and Prettier into your CI pipeline.

```typescript
// src/lib/utils/math.test.ts (Vitest unit test)
import { expect, test } from 'vitest';
import { add } from './math';

test('add function sums numbers correctly', () => {
  expect(add(1, 2)).toBe(3);
});
```

## 8. Common Pitfalls and Gotchas

Avoid common mistakes that lead to bugs or performance issues.

**Rule**: Do not import `$lib/server` modules into client-side code. SvelteKit will prevent this at build time, but catching it early saves time.

❌ BAD (in `+page.ts` or `+page.svelte`):
```typescript
import { serverOnlyFunction } from '$lib/server/utils'; // This will fail
```

✅ GOOD (in `+page.server.ts` or `+server.ts`):
```typescript
import { serverOnlyFunction } from '$lib/server/utils'; // Correct usage
```

**Rule**: Configure `tsconfig.json` correctly, extending SvelteKit's generated config. Specifically, ensure `verbatimModuleSyntax: true` and `isolatedModules: true`.

```json
// tsconfig.json
{
  "extends": "./.svelte-kit/tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "bundler",
    "module": "ESNext",
    "resolveJsonModule": true,
    "allowJs": true,
    "checkJs": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vitest/globals"] // If using Vitest
  },
  "include": ["src/**/*.d.ts", "src/**/*.ts", "src/**/*.js", "src/**/*.svelte"],
  "exclude": ["node_modules"]
}
```