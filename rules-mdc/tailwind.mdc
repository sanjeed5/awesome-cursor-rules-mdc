---
description: Provides definitive best practices for using Tailwind CSS, focusing on maintainability, performance, and scalability in modern web development.
globs: **/*.{css,html,jsx,tsx,vue}
---
# tailwind Best Practices

Tailwind CSS is a powerful utility-first framework. These rules ensure our projects leverage Tailwind effectively, promoting consistency, performance, and maintainability across the team.

## 1. Design System Configuration

**Always define your design system in `tailwind.config.js`.** Avoid arbitrary values in markup unless absolutely necessary for unique, non-reusable styles. This ensures UI consistency and enables efficient PurgeCSS tree-shaking.

❌ BAD: Arbitrary values in markup
```html
<div class="bg-[#1a2b3c] text-[15px] p-[1.25rem] rounded-[8px]">...</div>
```

✅ GOOD: Configured design tokens
```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        brand: { 500: '#1a2b3c' },
      },
      fontSize: {
        'body-md': '15px',
      },
      spacing: {
        'layout-sm': '1.25rem',
      },
      borderRadius: {
        'card': '8px',
      },
    },
  },
};
```
```html
<div class="bg-brand-500 text-body-md p-layout-sm rounded-card">...</div>
```

## 2. Component Abstraction

**Extract reusable UI patterns into framework-specific components.** This keeps your markup clean, promotes reusability, and centralizes complex styling logic. Use libraries like `clsx` or `cva` for dynamic class composition.

**Avoid `@apply` for general component styling.** It creates custom CSS that hides Tailwind's utility-first nature, making refactoring harder and reducing clarity. Reserve `@apply` for very specific, non-reusable base styles (e.g., resetting browser defaults) or legacy migrations.

❌ BAD: Duplicated utility soup in every instance
```html
<button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-hidden focus:ring-2 focus:ring-blue-500/50">
  Submit
</button>
<!-- ... many more buttons with the exact same classes ... -->
```

✅ GOOD: Reusable component (React example)
```jsx
// components/Button.jsx
import clsx from 'clsx';

export function Button({ variant = 'primary', size = 'md', className, ...props }) {
  const baseStyles = 'font-bold py-2 px-4 rounded focus:outline-hidden focus:ring-2';
  const variantStyles = {
    primary: 'bg-blue-500 hover:bg-blue-700 text-white focus:ring-blue-500/50',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-800 focus:ring-gray-300/50',
  };
  const sizeStyles = {
    sm: 'text-sm',
    md: 'text-base',
  };

  return (
    <button
      className={clsx(baseStyles, variantStyles[variant], sizeStyles[size], className)}
      {...props}
    />
  );
}
```
```jsx
// Usage
<Button>Submit</Button>
<Button variant="secondary">Cancel</Button>
```

## 3. Class Ordering & Readability

**Group utility classes logically for improved readability.** While Tailwind doesn't enforce an order, a consistent pattern makes scanning classes easier and reduces cognitive load. We recommend:
`Layout` > `Flexbox` > `Grid` > `Box Model (p, m, w, h)` > `Typography` > `Backgrounds` > `Borders` > `Effects` > `Interactivity` > `States (hover, focus, active, responsive)`

❌ BAD: Random order
```html
<div class="text-lg flex shadow-md p-4 items-center bg-white rounded-lg">...</div>
```

✅ GOOD: Consistent order
```html
<div class="flex items-center p-4 bg-white rounded-lg shadow-md text-lg">...</div>
```

## 4. Mobile-First & Responsive Design

**Adopt a mobile-first approach.** Design for the smallest screen first, then use responsive prefixes (`sm:`, `md:`, `lg:`, `xl:`, `2xl:`) to adapt styles for larger viewports. This is the default and most efficient way to build responsive UIs with Tailwind.

```html
<!-- Default (mobile) is stacked, medium screens and up are side-by-side -->
<div class="flex flex-col md:flex-row items-center md:items-start gap-4">
  <img class="w-24 h-24 rounded-full md:w-32 md:h-32" src="..." alt="Profile">
  <div class="text-center md:text-left">
    <h2 class="text-xl md:text-2xl font-bold">John Doe</h2>
    <p class="text-gray-600 md:text-lg">Software Engineer</p>
  </div>
</div>
```

## 5. Performance Optimization

**Configure `content` for PurgeCSS and leverage JIT.** Ensure your `tailwind.config.js` accurately lists *all* files containing Tailwind classes. This is critical for PurgeCSS to tree-shake unused CSS, keeping your bundle size minimal. Tailwind v4 uses the Just-In-Time (JIT) engine by default, providing fast compilation.

```js
// tailwind.config.js
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx,vue,html}', // Your source files
    './public/index.html', // Any raw HTML files
  ],
  theme: { extend: {} },
  plugins: [],
};
```

**Keep Tailwind updated.** Use `npx @tailwindcss/upgrade` to migrate from v3 to v4, benefiting from performance improvements and modern CSS features. Be aware of v4's modern browser requirements (Safari 16.4+, Chrome 111+, Firefox 128+).

## 6. Accessibility

**Prioritize semantic HTML.** Tailwind styles should enhance, not replace, proper HTML structure. Ensure all interactive elements have clear, visible focus states.

**`outline-hidden` vs `outline-none` (Tailwind v4):**
*   **`outline-hidden`**: Sets an invisible outline that *still shows in forced colors mode* for accessibility. **Prefer this for interactive elements** when you want to hide the default browser outline but ensure accessibility.
*   **`outline-none`**: Sets `outline-style: none`, completely removing the outline. **Avoid this for interactive elements** unless you provide a custom, accessible focus indicator (e.g., `focus:ring-2`).

❌ BAD: Removing focus entirely
```html
<button class="focus:outline-none">Click me</button>
```

✅ GOOD: Accessible focus state
```html
<button class="focus:outline-hidden focus:ring-2 focus:ring-blue-500/50">Click me</button>
```

## 7. Theming & Dark Mode

**Extend your `tailwind.config.js` for custom themes.** Define custom colors, fonts, spacing, and other design tokens to match your brand. Leverage the `dark:` variant for built-in dark mode support.

```js
// tailwind.config.js
module.exports = {
  darkMode: 'class', // or 'media'
  theme: {
    extend: {
      colors: {
        'primary-light': '#f0f9ff',
        'primary-dark': '#0c4a6e',
      },
    },
  },
};
```
```html
<div class="bg-primary-light dark:bg-primary-dark text-gray-900 dark:text-gray-100">
  Hello, world!
</div>
```