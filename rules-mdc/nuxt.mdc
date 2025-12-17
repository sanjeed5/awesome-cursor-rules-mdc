---
description: This guide provides definitive, actionable best practices for building robust, performant, and secure Nuxt applications, emphasizing modern patterns and common pitfalls.
globs: **/*.{vue,js,ts}
---
# Nuxt Best Practices

This document outlines the essential guidelines for developing high-quality Nuxt applications. Adhere to these rules to ensure maintainable, performant, and secure code.

## 1. Code Organization and Structure

Nuxt's convention-over-configuration is a superpower. Leverage it fully.

### 1.1. Directory Structure

Always follow the Nuxt opinionated directory structure. Do not deviate.

-   `components/`: Reusable Vue components.
-   `composables/`: Reusable Composition API functions (`useFoo()`).
-   `layouts/`: Application layouts.
-   `middleware/`: Route middleware.
-   `pages/`: Application pages (route definitions).
-   `plugins/`: Nuxt.js plugins.
-   `server/`: API routes and server-side logic.
-   `store/`: Pinia stores.
-   `utils/`: Generic utility functions.

### 1.2. File Naming Conventions

Consistency is key for auto-imports and readability.

-   **Components**: `PascalCase.vue`
    ❌ BAD: `my-button.vue`
    ✅ GOOD: `MyButton.vue`
-   **Composables**: `usePascalCase.ts`
    ❌ BAD: `fetchData.ts`
    ✅ GOOD: `useFetchData.ts`
-   **Pages/Layouts/Middleware/Plugins/Stores**: `kebab-case.ts` or `kebab-case.vue`
    ❌ BAD: `UserProfile.vue` (for a page)
    ✅ GOOD: `user-profile.vue`

### 1.3. TypeScript Adoption

Enable TypeScript everywhere. It prevents bugs and improves developer experience.

-   Configure `tsconfig.json` and ensure ESLint is set up for TypeScript.
-   Always explicitly type Pinia stores, middleware, and composables.
-   Run `nuxt prepare` or `nuxt dev` regularly to generate `imports.d.ts` for type safety.

## 2. Component Architecture

Design components for reusability, testability, and clarity.

### 2.1. Favor Composition over Inheritance

Use the Composition API to organize component logic.

❌ BAD:
```vue
<!-- MyComponent.vue -->
<script>
export default {
  mixins: [myMixin], // Avoid mixins
  // ...
}
</script>
```

✅ GOOD:
```vue
<!-- MyComponent.vue -->
<script setup lang="ts">
import { useMyFeature } from '~/composables/useMyFeature';
const { data, loading } = useMyFeature();
</script>
```

### 2.2. Use Slots for Flexibility

Slots make components adaptable without prop drilling.

❌ BAD:
```vue
<!-- MyCard.vue -->
<template>
  <div class="card">
    <h2 v-if="title">{{ title }}</h2>
    <p v-if="description">{{ description }}</p>
    <button v-if="showButton" @click="emit('action')">Action</button>
  </div>
</template>
<script setup lang="ts">
defineProps<{ title?: string; description?: string; showButton?: boolean }>();
const emit = defineEmits(['action']);
</script>
```

✅ GOOD:
```vue
<!-- MyCard.vue -->
<template>
  <div class="card">
    <slot name="header" />
    <slot /> <!-- Default slot for content -->
    <slot name="actions" />
  </div>
</template>
```

## 3. State Management (Pinia)

Pinia is the definitive state management solution for Nuxt.

### 3.1. Single Source of Truth & Immutability

Maintain a single, consistent source of truth. Always update state via actions/mutations.

❌ BAD:
```ts
// store/user.ts
import { defineStore } from 'pinia';
export const useUserStore = defineStore('user', {
  state: () => ({ name: 'John Doe' }),
  actions: {
    updateNameDirectly(newName: string) {
      this.name = newName; // Direct mutation outside action/mutation
    },
  },
});
```

✅ GOOD:
```ts
// store/user.ts
import { defineStore } from 'pinia';
export const useUserStore = defineStore('user', {
  state: () => ({ name: 'John Doe' }),
  actions: {
    setName(newName: string) {
      this.name = newName; // Mutation within an action
    },
  },
});
```

### 3.2. Modularization

Organize stores by feature.

```ts
// store/auth.ts
export const useAuthStore = defineStore('auth', { /* ... */ });

// store/products.ts
export const useProductsStore = defineStore('products', { /* ... */ });
```

## 4. Common Patterns and Anti-patterns

### 4.1. Data Fetching

Always use Nuxt's built-in data fetching composables for SSR compatibility and deduplication.

-   `useFetch`: For direct API calls.
-   `useAsyncData`: For custom async operations.

❌ BAD:
```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue';
const data = ref(null);
onMounted(async () => {
  data.value = await $fetch('/api/items'); // Not SSR-friendly, fetches twice
});
</script>
```

✅ GOOD:
```vue
<script setup lang="ts">
const { data } = await useFetch('/api/items'); // SSR-friendly, fetches once
</script>
```

### 4.2. Error Handling

Implement robust error handling.

-   **Centralized**: Use `NuxtError` component for global errors.
-   **API Errors**: Always catch and handle errors from `useFetch`/`useAsyncData`.

```vue
<script setup lang="ts">
const { data, error } = await useFetch('/api/data');
if (error.value) {
  // Handle specific error, e.g., redirect or show message
  console.error('Failed to fetch data:', error.value);
  throw createError({ statusCode: 500, statusMessage: 'Could not load data' });
}
</script>
```

### 4.3. Authentication & Authorization

Use Nuxt middleware for route protection.

```ts
// middleware/auth.ts
export default defineNuxtRouteMiddleware((to, from) => {
  const userStore = useUserStore();
  if (!userStore.isAuthenticated && to.path !== '/login') {
    return navigateTo('/login');
  }
});
```

## 5. Performance Considerations

Optimize for Core Web Vitals.

### 5.1. Smart Links

Use `<NuxtLink>` for all internal navigation. Configure `prefetchOn: 'interaction'` for optimal performance.

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  experimental: {
    defaults: {
      nuxtLink: {
        prefetchOn: 'interaction', // Prefetch only on user interaction
      },
    },
  },
});
```

### 5.2. Hybrid Rendering

Leverage `routeRules` for granular control over page rendering.

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    '/': { prerender: true }, // Static generation at build time
    '/products/**': { swr: 3600 }, // Stale-While-Revalidate for dynamic content
    '/admin/**': { ssr: false }, // Client-side rendering for authenticated sections
  },
});
```

### 5.3. Lazy Loading Components

Prefix components with `Lazy` to defer loading.

❌ BAD:
```vue
<template>
  <HeavyComponent v-if="show" />
</template>
```

✅ GOOD:
```vue
<template>
  <LazyHeavyComponent v-if="show" />
</template>
```

### 5.4. Image Optimization

Always use `@nuxt/image` and `<NuxtImg>`.

```vue
<template>
  <!-- Critical image, load ASAP -->
  <NuxtImg
    src="/hero.jpg"
    format="webp"
    preload
    loading="eager"
    fetch-priority="high"
    width="1200"
    height="600"
  />
  <!-- Non-critical image, lazy load -->
  <NuxtImg
    src="/gallery/item.jpg"
    format="webp"
    loading="lazy"
    fetch-priority="low"
    width="400"
    height="300"
  />
</template>
```

## 6. Accessibility (A11y)

Build inclusive applications from the start.

-   Use semantic HTML elements.
-   Provide `alt` text for all images.
-   Ensure keyboard navigation is fully functional.
-   Manage focus for modals and dynamic content.
-   Maintain sufficient color contrast.

## 7. Security Best Practices

Assume breach and validate everything.

### 7.1. Server-Side Input Validation

Always validate all incoming data on the server. Client-side validation is for UX, not security.

```ts
// server/api/users.post.ts
import { z } from 'zod'; // Use Zod for robust schema validation

const userSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8),
});

export default defineEventHandler(async (event) => {
  try {
    const body = await readBody(event);
    const validatedData = userSchema.parse(body);
    // Proceed with validatedData
    return { status: 'success', data: validatedData };
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw createError({ statusCode: 400, message: 'Invalid input', data: error.errors });
    }
    throw createError({ statusCode: 500, message: 'Server error' });
  }
});
```

### 7.2. Content Sanitization

Sanitize any user-generated HTML before rendering.

```ts
// utils/sanitize.ts
import DOMPurify from 'isomorphic-dompurify';

export function sanitizeHtml(dirtyHtml: string): string {
  return DOMPurify.sanitize(dirtyHtml, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br', 'a'],
    ALLOWED_ATTR: ['href', 'target'],
  });
}
```

### 7.3. Secure Cookies

For authentication tokens, use `httpOnly`, `secure`, and `sameSite` flags.

```ts
// server/api/auth/login.post.ts
export default defineEventHandler(async (event) => {
  // ... authentication logic ...
  setCookie(event, 'auth_token', 'your_jwt_token', {
    httpOnly: true, // Prevent client-side JS access
    secure: process.env.NODE_ENV === 'production', // Only send over HTTPS in production
    sameSite: 'lax', // CSRF protection
    maxAge: 60 * 60 * 24 * 7, // 7 days
    path: '/',
  });
  return { message: 'Logged in' };
});
```

### 7.4. Avoid Dangerous JavaScript Patterns

Never execute user input as code.

❌ BAD: `eval(userInput)`, `new Function(userInput)`, `element.innerHTML = userInput`
✅ GOOD: `JSON.parse(userInput)`, `element.textContent = userInput`

## 8. Common Pitfalls and Gotchas

-   **Forgetting `nuxt prepare`**: If TypeScript errors appear for auto-imports, run `npx nuxt prepare`.
-   **Heavy async work in module setup**: Defer time-consuming logic to Nuxt hooks (`onInstall`, `onUpgrade`) or runtime hooks, not the module's `setup` function.
-   **Not prefixing module exports**: Custom modules must prefix all exposed APIs, components, and composables (e.g., `useMyModuleFoo`).
-   **Directly mutating props**: Always emit events to update parent state.
-   **Ignoring server-side security**: Never trust client input; validate and sanitize on the server.