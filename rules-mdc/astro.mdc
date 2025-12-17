---
description: This guide provides opinionated, actionable best practices for building high-performance Astro applications, focusing on zero-JavaScript by default, island architecture, and type safety.
globs: **/*.{js,jsx,ts,tsx,astro}
---
# Astro Best Practices

Astro is a "disappearing" framework designed for shipping zero JavaScript by default, adding interactivity only where needed. Our team leverages Astro for its unparalleled performance and developer experience, especially for content-focused sites. Adhere to these guidelines to maintain a consistent, high-performing, and maintainable codebase.

## 1. Code Organization and Structure

Maintain a predictable and logical file structure for easy navigation and scaling.

### 1.1 Standard Project Structure

Always scaffold new projects with the official Astro starter and maintain the following structure:

```
src/
├── components/ # Reusable UI components (Astro or UI framework islands)
│   ├── ui/ # Generic, framework-agnostic components
│   └── react/ # React-specific components
├── content/ # Markdown/MDX files for Content Collections
├── layouts/ # Base page layouts
├── pages/ # Top-level routes
├── styles/ # Global CSS, utility classes, Tailwind config
├── env.d.ts # Global type declarations
└── astro.config.ts # Type-safe Astro configuration
```

### 1.2 Naming Conventions

- **Astro Components (`.astro`):** Use `kebab-case`.
- **UI Framework Components (e.g., React `.jsx`, `.tsx`):** Use `PascalCase`.
- **Pages (`.astro`):** Use `kebab-case` for file names.

❌ BAD
```astro
<!-- src/components/MyCard.astro -->
<div class="myCard">...</div>
```
```jsx
// src/components/Mycard.jsx
export default function Mycard() { return <div>...</div>; }
```

✅ GOOD
```astro
<!-- src/components/my-card.astro -->
<div class="my-card">...</div>
```
```jsx
// src/components/react/MyCard.tsx
export default function MyCard() { return <div>...</div>; }
```

### 1.3 TypeScript Configuration

Leverage TypeScript throughout the codebase for type safety, even in `.astro` components.

**`tsconfig.json`**
```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "verbatimModuleSyntax": true, // Enforce explicit type imports
    "plugins": [
      {
        "name": "@astrojs/ts-plugin" // Essential for editor support
      }
    ],
    "paths": {
      "@components/*": ["./src/components/*"],
      "@layouts/*": ["./src/layouts/*"],
      "@content/*": ["./src/content/*"],
      "@styles/*": ["./src/styles/*"]
    }
  },
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

**`astro.config.ts`**
Always use a `.ts` extension for your Astro configuration for type safety.
```ts
import { defineConfig } from 'astro/config';
import react from '@astrojs/react'; // Example integration

export default defineConfig({
  integrations: [react()],
  // ... other config
});
```

## 2. Component Architecture: Astro Islands

Astro's "islands" architecture is fundamental. Prioritize static HTML and introduce interactivity only when and where needed.

### 2.1 Astro Components for Static Content & Layouts

Use `.astro` files for pages, layouts, and components that are primarily static or orchestrate other components. They render to HTML on the server and ship zero JavaScript by default.

```astro
---
import Layout from '@layouts/Layout.astro';
import MyReactCounter from '@components/react/MyReactCounter';
---

<Layout title="Welcome to Astro">
  <main>
    <h1>Hello, Astro!</h1>
    <p>This is static content.</p>
    <!-- Render a React island only when visible -->
    <MyReactCounter client:visible />
  </main>
</Layout>
```

### 2.2 UI Framework Components for Interactivity (Islands)

Use React, Vue, Svelte, etc., components *only* for interactive UI pieces. These are your "islands."

```tsx
// src/components/react/MyReactCounter.tsx
import { useState } from 'react';

export default function MyReactCounter() {
  const [count, setCount] = useState(0);
  return (
    <div className="counter">
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

### 2.3 Client Directives for Selective Hydration

Explicitly tell Astro *when* to hydrate an island. Avoid `client:load` unless the component is critical for initial interaction.

❌ BAD (Hydrates immediately, even if not needed)
```astro
<MyHeavyAnimation client:load />
```

✅ GOOD (Hydrates only when visible, saving bandwidth and CPU)
```astro
<MyHeavyAnimation client:visible />
```

## 3. Performance Considerations

Astro's primary strength is performance. Ensure every decision aligns with shipping minimal JavaScript.

### 3.1 Zero JavaScript by Default

Always assume components are static unless explicitly told otherwise. If a component doesn't need client-side JS, don't add it.

❌ BAD (Unnecessary client-side script for a simple toggle)
```astro
<button id="toggleButton">Toggle</button>
<script>
  document.getElementById('toggleButton').addEventListener('click', () => { /* ... */ });
</script>
```

✅ GOOD (Pure CSS or server-rendered state for simple interactions)
```astro
<button class="toggle-button">Toggle</button>
<style>
  .toggle-button:focus + .content { display: block; } /* Example CSS toggle */
</style>
```

### 3.2 Optimize Client Directives

- **`client:visible`**: Default for most interactive components that appear lower on the page.
- **`client:idle`**: For non-critical interactivity that can wait until the main thread is free.
- **`client:load`**: Only for critical, above-the-fold interactions (e.g., main navigation, search bar).
- **`client:media={query}`**: For components that only become interactive on specific screen sizes.

### 3.3 Image Optimization

Use Astro's built-in image optimization for responsive, performant images.

```astro
---
import { Image } from 'astro:assets';
import myImage from '@assets/my-image.jpg'; // Import local images
---
<Image src={myImage} alt="A descriptive alt text" width={800} height={600} format="webp" quality={80} />
```

## 4. Common Pitfalls and Anti-patterns

Avoid these common mistakes that undermine Astro's performance benefits.

### 4.1 Over-Hydration

Do not wrap entire pages or large sections in a single interactive component. This defeats the purpose of islands.

❌ BAD (Treating Astro like an SPA)
```astro
<MyReactApp client:load /> <!-- MyReactApp contains the entire page -->
```

✅ GOOD (Granular islands)
```astro
<header>...</header>
<MyReactNav client:load />
<main>
  <MyStaticContent />
  <MyInteractiveCarousel client:visible />
</main>
<footer>...</footer>
```

### 4.2 Unnecessary Client-Side Scripts

Avoid `<script>` tags in `.astro` files unless they are explicitly marked `is:inline` (for tiny, critical scripts) or are part of an island.

❌ BAD (Implicit client-side JS)
```astro
<script>
  console.log('This script runs on the client by default, even if not needed.');
</script>
```

✅ GOOD (Explicitly inline or part of an island)
```astro
<script is:inline>
  // Only for tiny, critical, non-blocking scripts
  console.log('This script is inlined and runs early.');
</script>
```

## 5. State Management

Keep state management simple and localized. Astro is not an SPA framework; most "state" should be server-rendered or localized to islands.

### 5.1 Local Island State

For interactivity within an island, use the UI framework's native state management (e.g., React `useState`, Svelte stores).

```tsx
// src/components/react/ThemeToggle.tsx
import { useState } from 'react';

export default function ThemeToggle() {
  const [theme, setTheme] = useState('light');
  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light');

  return (
    <button onClick={toggleTheme}>
      Switch to {theme === 'light' ? 'dark' : 'light'} theme
    </button>
  );
}
```

### 5.2 Global State (Minimal)

For truly global, cross-island state (e.g., user authentication status), consider:
1. **Server-side data:** Fetch global data in the Astro component frontmatter and pass it down as props. This is the preferred method.
2. **Lightweight context/store within a shared island:** If client-side global state is unavoidable, use a minimal library like Zustand or Jotai within a single, shared UI framework component that wraps other islands.

❌ BAD (Heavy global state management for a content site)
```tsx
// src/components/react/MyApp.tsx (large, complex context provider)
<MyGlobalContextProvider>
  <Header client:load />
  <Content />
  <Footer />
</MyGlobalContextProvider>
```

✅ GOOD (Server-rendered global data, localized island state)
```astro
---
import Layout from '@layouts/Layout.astro';
import UserProfile from '@components/react/UserProfile';
import { getAuthStatus } from '../utils/auth'; // Server-side utility

const user = await getAuthStatus(Astro.request);
---
<Layout title="My Page">
  <header>
    <UserProfile client:load user={user} /> {/* Pass server data as props */}
  </header>
  <main>
    <!-- ... other static content and localized islands -->
  </main>
</Layout>
```

## 6. Accessibility

Build accessible experiences from the ground up.

### 6.1 Semantic HTML

Always use appropriate HTML5 semantic elements.

❌ BAD
```html
<div onclick="doSomething()">Click me</div>
```

✅ GOOD
```html
<button type="button" onclick="doSomething()">Click me</button>
```

### 6.2 ARIA Attributes

Apply ARIA attributes when native HTML semantics are insufficient, especially for complex interactive widgets within islands.

```jsx
// src/components/react/AccessibleAccordion.tsx
<div role="region" aria-labelledby="accordion-header">
  <h3 id="accordion-header">
    <button aria-expanded={isOpen} aria-controls="accordion-panel" onClick={toggle}>
      Section Title
    </button>
  </h3>
  {isOpen && <div id="accordion-panel" role="region">...</div>}
</div>
```

### 6.3 Image Alt Text

All `<img>` tags must have descriptive `alt` attributes.

❌ BAD
```html
<img src="/image.jpg" alt="">
```

✅ GOOD
```html
<img src="/image.jpg" alt="A detailed description of the image content.">
```

## 7. Testing Approaches

Implement a robust testing strategy that covers both static content and interactive islands.

### 7.1 Type Checking

Run `astro check` as part of your CI pipeline to catch TypeScript errors early.

```bash
# In your CI script
npm run astro check
```

### 7.2 Linting

Enforce code style and catch common errors with ESLint, including `eslint-plugin-astro`.

**`.eslintrc.cjs`**
```javascript
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:astro/recommended',
    'plugin:react/recommended', // If using React
    'plugin:@typescript-eslint/recommended' // If using TypeScript
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json',
    extraFileExtensions: ['.astro']
  },
  rules: {
    // Custom rules
  },
  overrides: [
    {
      files: ['*.astro'],
      parser: 'astro-eslint-parser',
      parserOptions: {
        parser: '@typescript-eslint/parser',
        extraFileExtensions: ['.astro']
      }
    }
  ],
  settings: {
    react: {
      version: 'detect'
    }
  }
};
```

### 7.3 Unit Testing Islands

Test interactive UI framework components (islands) using their native testing libraries (e.g., React Testing Library, Vue Test Utils).

```tsx
// src/components/react/MyReactCounter.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import MyReactCounter from './MyReactCounter';

test('increments count on button click', () => {
  render(<MyReactCounter />);
  const button = screen.getByRole('button', { name: /increment/i });
  const countDisplay = screen.getByText(/count: 0/i);

  expect(countDisplay).toBeInTheDocument();
  fireEvent.click(button);
  expect(screen.getByText(/count: 1/i)).toBeInTheDocument();
});
```

### 7.4 End-to-End (E2E) Testing

Use Playwright or Cypress for E2E tests to verify page rendering, navigation, and island hydration in a real browser environment.

```javascript
// tests/example.spec.ts (Playwright)
import { test, expect } from '@playwright/test';

test('basic page loads and counter works', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toHaveText('Hello, Astro!');

  // Interact with the React island
  const counterButton = page.getByRole('button', { name: 'Increment' });
  await expect(page.getByText('Count: 0')).toBeVisible();
  await counterButton.click();
  await expect(page.getByText('Count: 1')).toBeVisible();
});
```