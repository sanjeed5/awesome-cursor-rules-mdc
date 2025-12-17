---
description: Definitive guide for writing clean, performant, and maintainable Svelte 5 applications using the runes API and modern TypeScript practices.
globs: **/*.{svelte,js,ts}
---
# Svelte Best Practices (Svelte 5 + TypeScript)

This guide outlines the definitive best practices for developing Svelte 5 applications with TypeScript. We prioritize explicit reactivity, strong typing, and compiler-driven performance.

## 1. Code Organization and Structure

Always structure your Svelte components for clarity and maintainability.

### 1.1. HTML-First Component Design

Keep your component logic, markup, and styles co-located within the `.svelte` file. This is Svelte's core philosophy.

**❌ BAD: Over-abstracting into separate JS files for simple logic**
```svelte
<!-- MyComponent.svelte -->
<script lang="ts">
  import { calculateValue } from './utils'; // Unnecessary abstraction
  let count = $state(0);
  let computed = $derived(calculateValue(count));
</script>
```

**✅ GOOD: Keep related logic within the component**
```svelte
<!-- MyComponent.svelte -->
<script lang="ts">
  let count = $state(0);
  let computed = $derived(count * 2 + 1); // Simple logic stays here
</script>
```

### 1.2. TypeScript Everywhere

Use `lang="ts"` in all `<script>` tags. Configure `svelte-check` in CI and use the VS Code extension for real-time feedback.

**❌ BAD: Missing `lang="ts"`**
```svelte
<script>
  let name: string = 'world'; // No type checking
</script>
```

**✅ GOOD: Explicitly enable TypeScript**
```svelte
<script lang="ts">
  let name: string = 'world'; // Full type safety
</script>
```

## 2. Component Architecture (Svelte 5 Runes)

Svelte 5's runes API makes reactivity explicit and powerful. Embrace it fully.

### 2.1. Explicit Reactive State with `$state`

Declare all reactive state using `$state`. This makes your component's reactivity clear and refactor-friendly.

**❌ BAD: Implicit reactivity (Svelte 4 style)**
```svelte
<script lang="ts">
  let count = 0; // Not reactive outside top-level
  function increment() { count++; }
</script>
```

**✅ GOOD: Explicit `$state` for all reactive variables**
```svelte
<script lang="ts">
  let count = $state(0); // Clearly reactive
  function increment() { count++; }
</script>
```

### 2.2. Derived State with `$derived`

Use `$derived` for any value that is a pure computation of other reactive state.

**❌ BAD: `$effect` or `$:` for derived values**
```svelte
<script lang="ts">
  let count = $state(0);
  $effect(() => { // Incorrectly using effect for derivation
    console.log('Count changed:', count);
    // This is a side effect, not a derivation
  });
  $: doubled = count * 2; // Svelte 4 style, less explicit
</script>
```

**✅ GOOD: `$derived` for pure computations**
```svelte
<script lang="ts">
  let count = $state(0);
  let doubled = $derived(count * 2); // Clearly a derived value
</script>
```

### 2.3. Side Effects with `$effect`

Reserve `$effect` for actual side effects (e.g., DOM manipulation, API calls, logging).

**❌ BAD: Mixing derivations and effects**
```svelte
<script lang="ts">
  let count = $state(0);
  $: if (count > 5) { // Svelte 4 style, combines derivation and effect
    alert('Count is too high!');
  }
</script>
```

**✅ GOOD: Clear separation of concerns**
```svelte
<script lang="ts">
  let count = $state(0);
  $effect(() => { // Clearly a side effect
    if (count > 5) {
      alert('Count is too high!');
    }
  });
</script>
```

### 2.4. Component Properties with `$props`

Declare all component properties using the `$props()` rune and destructuring. This provides powerful type safety and flexibility.

**❌ BAD: `export let` (Svelte 4 style)**
```svelte
<script lang="ts">
  export let name: string;
  export let age: number = 30;
</script>
```

**✅ GOOD: `$props()` for properties**
```svelte
<script lang="ts">
  interface Props {
    name: string;
    age?: number; // Optional prop
  }
  let { name, age = 30 }: Props = $props();
</script>
```

### 2.5. Event Handling

Remove the `on:` prefix for event handlers. They are now standard properties.

**❌ BAD: `on:click`**
```svelte
<button on:click={() => alert('Clicked!')}>Click me</button>
```

**✅ GOOD: `onclick` (standard HTML attribute)**
```svelte
<button onclick={() => alert('Clicked!')}>Click me</button>
```

### 2.6. Slots and Snippets

Use `{@render}` for rendering slots and snippets, providing type safety and better control over content.

**❌ BAD: Untyped slots**
```svelte
<!-- MyComponent.svelte -->
<slot />
```

**✅ GOOD: Typed snippets for predictable content**
```svelte
<!-- MyComponent.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte';
  let { header }: { header: Snippet } = $props();
</script>
<div>
  {@render header()}
  <slot />
</div>
```

## 3. State Management

Prioritize local component state with `$state`. Use Svelte stores for truly global or complex cross-component state.

### 3.1. Local State First

Most state should live within the component using `$state`.

**❌ BAD: Over-reliance on global stores for simple component state**
```svelte
// store.ts
import { writable } from 'svelte/store';
export const localCounter = writable(0); // Unnecessary global
```

**✅ GOOD: Use `$state` for component-local reactivity**
```svelte
<script lang="ts">
  let count = $state(0); // Local to this component
</script>
```

### 3.2. Svelte Stores for Global State

For state shared across many components or complex application-wide state, use Svelte's built-in stores.

```ts
// src/lib/stores/auth.ts
import { writable } from 'svelte/store';
export const isAuthenticated = writable(false);
export const user = writable<{ id: string; name: string } | null>(null);
```
```svelte
<!-- AuthStatus.svelte -->
<script lang="ts">
  import { isAuthenticated, user } from '$lib/stores/auth';
</script>
{#if $isAuthenticated}
  <p>Welcome, {$user?.name}!</p>
{:else}
  <p>Please log in.</p>
{/if}
```

## 4. Performance Considerations

SvelteKit handles many optimizations automatically. Supplement these with mindful coding.

### 4.1. `{#each}` with Keys

Always provide a unique `key` to `{#each}` blocks for efficient DOM updates.

**❌ BAD: Missing `key`**
```svelte
{#each items as item}
  <ItemComponent {item} />
{/each}
```

**✅ GOOD: Use a unique `key`**
```svelte
{#each items as item (item.id)}
  <ItemComponent {item} />
{/each}
```

### 4.2. Dynamic Imports for Lazy Loading

Lazy-load components or modules that aren't immediately needed to reduce initial bundle size.

```svelte
<script lang="ts">
  let showModal = $state(false);
  async function openModal() {
    const { default: Modal } = await import('./Modal.svelte');
    // Render Modal component
    showModal = true;
  }
</script>
{#if showModal}
  <Modal />
{/if}
<button onclick={openModal}>Open Modal</button>
```

## 5. Common Pitfalls and Gotchas

Avoid these common mistakes to ensure robust and predictable Svelte applications.

### 5.1. Misunderstanding `$effect` Dependencies

`$effect` runs when its *dependencies* change. Be explicit about what it depends on.

**❌ BAD: Implicit dependencies or missing cleanup**
```svelte
<script lang="ts">
  let count = $state(0);
  $effect(() => {
    // This effect might not re-run if `count` isn't directly used
    // or if `doSomething` has internal state changes not tracked by Svelte.
    doSomething(count);
  });
</script>
```

**✅ GOOD: Explicit dependencies and cleanup**
```svelte
<script lang="ts">
  let count = $state(0);
  $effect(() => {
    const timer = setInterval(() => {
      console.log('Count is', count); // `count` is a dependency
    }, 1000);
    return () => clearInterval(timer); // Cleanup function
  });
</script>
```

### 5.2. Forgetting `generics` Attribute

When creating generic components, remember to add the `generics` attribute to the `<script>` tag.

**❌ BAD: Untyped generic props**
```svelte
<script lang="ts">
  interface Props { items: any[]; } // Loses type safety
  let { items }: Props = $props();
</script>
```

**✅ GOOD: Explicit generics for type safety**
```svelte
<script lang="ts" generics="Item extends { id: string }">
  interface Props { items: Item[]; }
  let { items }: Props = $props();
</script>
```

## 6. Accessibility

Build accessible Svelte applications by default.

### 6.1. Semantic HTML

Always prefer semantic HTML elements over generic `div`s or `span`s.

**❌ BAD: Non-semantic button**
```svelte
<div onclick={handleClick} role="button" tabindex="0">Click me</div>
```

**✅ GOOD: Semantic HTML element**
```svelte
<button onclick={handleClick}>Click me</button>
```

### 6.2. ARIA Attributes

Use ARIA attributes when semantic HTML isn't sufficient, but only when necessary.

```svelte
<label for="username">Username</label>
<input id="username" type="text" aria-describedby="username-hint" />
<p id="username-hint">Your username must be unique.</p>
```

## 7. Testing Approaches

Implement a robust testing strategy for Svelte applications.

### 7.1. Unit Testing with Vitest and Svelte Testing Library

Use `vitest` for your test runner and `@testing-library/svelte` for component testing.

```ts
// src/lib/components/Counter.test.ts
import { render, screen } from '@testing-library/svelte';
import { expect, test } from 'vitest';
import Counter from './Counter.svelte';

test('Counter increments value on click', async () => {
  render(Counter);
  const button = screen.getByRole('button', { name: /Count:/i });
  expect(button).toHaveTextContent('Count: 0');
  await button.click();
  expect(button).toHaveTextContent('Count: 1');
});
```

### 7.2. End-to-End (E2E) Testing

For critical user flows, use `Playwright` or `Cypress`.

```ts
// tests/example.spec.ts (Playwright)
import { test, expect } from '@playwright/test';

test('homepage has title and intro text', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/SvelteKit/);
  await expect(page.locator('h1')).toHaveText('Welcome to SvelteKit');
});
```