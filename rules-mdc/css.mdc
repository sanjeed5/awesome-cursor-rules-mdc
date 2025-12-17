---
description: Definitive guidelines for writing maintainable, performant, and accessible CSS. Focus on modern practices, robust architecture, and efficient styling patterns.
globs: **/*
---
# CSS Best Practices

This guide outlines the definitive best practices for writing CSS within our team. Adhering to these principles ensures our stylesheets are scalable, maintainable, performant, and accessible.

## 1. Code Organization & Structure

Adopt a modular, predictable structure. Use Sass for pre-processing.

### 1.1 File Structure (Modular & Scalable)

Organize CSS into logical, focused files. A simplified ITCSS-like structure is recommended.

```css
/*
  css/
  ├── base/             // Project-wide defaults (reset, typography, variables)
  │   ├── _reset.scss
  │   ├── _typography.scss
  │   └── _variables.scss // Design tokens
  ├── components/       // Self-contained UI components (buttons, cards)
  │   ├── _button.scss
  │   └── _card.scss
  ├── layout/           // Major page sections (header, footer, grid)
  │   ├── _grid.scss
  │   └── _header.scss
  └── utilities/        // Single-purpose helper classes (spacing, text alignment)
      ├── _spacing.scss
      └── _helpers.scss
*/

// main.scss
@import 'base/reset';
@import 'base/variables';
@import 'base/typography';

@import 'layout/grid';
@import 'layout/header';

@import 'components/button';
@import 'components/card';

@import 'utilities/spacing';
@import 'utilities/helpers';
```

### 1.2 Naming Convention (BEM)

Use BEM (Block-Element-Modifier) for clear, flat, and highly readable selectors. This prevents specificity wars and improves reusability.

❌ BAD: Deeply nested, ambiguous selectors
```css
.header .nav ul li a { /* ... */ }
.card.featured { /* ... */ }
```

✅ GOOD: BEM for clarity and reusability
```css
/* Block */
.card { /* ... */ }

/* Element */
.card__title { /* ... */ }
.card__image { /* ... */ }

/* Modifier */
.card--featured { /* ... */ }
.card__title--large { /* ... */ }
```

### 1.3 CSS Custom Properties (Design Tokens)

Centralize design values (colors, fonts, spacing) using CSS variables. This enables consistency, easy theming, and dynamic updates.

```css
/* _variables.scss */
:root {
  --color-primary: #007bff;
  --color-text: #333;
  --font-family-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
  --spacing-md: 1rem;
}

/* Usage */
.button {
  background-color: var(--color-primary);
  color: white;
  padding: var(--spacing-md);
  font-family: var(--font-family-sans);
}
```

### 1.4 Minimal Nesting (Sass)

Avoid excessive nesting in Sass; it leads to overly specific and hard-to-override CSS. Nest only when necessary for context or specificity (e.g., pseudo-states).

❌ BAD: Over-nested selectors
```scss
.nav {
  ul {
    li {
      a {
        color: var(--color-text);
        &:hover {
          color: var(--color-primary);
        }
      }
    }
  }
}
```

✅ GOOD: Flat structure, nest only for direct context
```scss
.nav {
  &__list { /* ... */ }
  &__item { /* ... */ }
  &__link {
    color: var(--color-text);
    &:hover {
      color: var(--color-primary);
    }
  }
}
```

## 2. Layout & Responsive Design

Prioritize mobile-first design using modern layout tools and relative units.

### 2.1 Mobile-First Approach

Design for small screens first, then progressively enhance for larger viewports using media queries.

```css
/* Mobile styles first */
.container {
  padding: 1rem;
}

/* Larger screens (e.g., tablet and up) */
@media (min-width: 768px) {
  .container {
    padding: 2rem;
  }
}
```

### 2.2 CSS Grid for 2D Layouts, Flexbox for 1D Layouts

Use the right tool for the job.

✅ GOOD: Flexbox for aligning items in a single dimension (row or column)
```css
.button-group {
  display: flex;
  gap: 1rem;
  justify-content: center; /* Horizontally center items */
  align-items: center;     /* Vertically center items */
}
```

✅ GOOD: CSS Grid for complex two-dimensional page layouts
```css
.page-layout {
  display: grid;
  grid-template-columns: 1fr; /* Single column on mobile */
  gap: 1.5rem;
}

@media (min-width: 992px) {
  .page-layout {
    grid-template-columns: 250px 1fr; /* Sidebar + main content on desktop */
    grid-template-areas: "sidebar main";
  }
  .page-layout__sidebar { grid-area: sidebar; }
  .page-layout__main { grid-area: main; }
}
```

### 2.3 Relative Units

Use relative units (`rem`, `em`, `vw`, `vh`, `%`) for flexible, scalable, and accessible designs. Avoid fixed `px` values for spacing and typography.

❌ BAD: Fixed `px` values
```css
.text { font-size: 16px; margin-bottom: 20px; }
.hero { height: 400px; }
```

✅ GOOD: Relative units
```css
.text { font-size: 1rem; margin-bottom: 1.25rem; } /* Based on root font-size */
.hero { min-height: 70vh; } /* Viewport height */
.image { max-width: 100%; height: auto; } /* Fluid images */
```

### 2.4 Container Queries

Leverage `@container` queries for component-level responsiveness, allowing components to adapt based on their parent container's size, not just the viewport.

```css
.card-container {
  container-type: inline-size;
  container-name: card-scope;
}

.card {
  display: flex;
  flex-direction: column;
}

@container card-scope (min-width: 400px) {
  .card {
    flex-direction: row;
    align-items: center;
  }
  .card__image {
    width: 150px;
    height: auto;
  }
}
```

## 3. Performance Considerations

Optimize CSS for fast loading and efficient rendering.

### 3.1 Efficient Selectors

Browsers read selectors right-to-left. Keep selectors short and avoid overly specific or universal selectors where possible.

❌ BAD: Overly specific, slow selector
```css
main div p.title { /* ... */ }
* { box-sizing: border-box; } /* Global, but expensive if not reset */
```

✅ GOOD: Short, class-based selectors
```css
.title { /* ... */ }
.button { /* ... */ }
```

### 3.2 Minimize Reflows & Repaints

Animate `transform` and `opacity` properties for smooth animations, as they can be hardware-accelerated. Avoid animating properties that trigger layout changes (e.g., `width`, `height`, `margin`, `padding`).

❌ BAD: Animating layout properties
```css
.modal {
  transition: width 0.3s ease, height 0.3s ease;
}
```

✅ GOOD: Animating `transform` and `opacity`
```css
.modal {
  transition: transform 0.3s ease, opacity 0.3s ease;
  transform: scale(0.9);
  opacity: 0;
}
.modal.is-open {
  transform: scale(1);
  opacity: 1;
}
```

### 3.3 Font Optimization

Use `font-display: swap` for web fonts to prevent invisible text during font loading (FOIT).

```css
@font-face {
  font-family: 'CustomFont';
  src: url('CustomFont.woff2') format('woff2');
  font-weight: 400;
  font-display: swap; /* Crucial for performance */
}
```

## 4. Accessibility

Bake accessibility into every styling decision.

### 4.1 Color Contrast

Ensure text and interactive elements meet WCAG AA contrast ratios (4.5:1 for normal text, 3:1 for large text/UI components). Use tools to check.

### 4.2 Visible Focus Indicators

Provide clear, visible focus styles for keyboard users.

❌ BAD: Removing outline without replacement
```css
a:focus, button:focus {
  outline: none; /* Don't do this without a custom focus style! */
}
```

✅ GOOD: Custom, visible focus styles
```css
a:focus, button:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
```

### 4.3 Respect `prefers-reduced-motion`

Offer a reduced motion experience for users who prefer it.

```css
/* Default animation */
.element {
  transition: transform 0.3s ease-out;
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  .element {
    transition: none;
    animation: none;
  }
}
```

## 5. Common Pitfalls & Anti-patterns

Avoid these patterns to maintain a healthy codebase.

### 5.1 Avoid `!important`

`!important` breaks the cascade and makes styles extremely difficult to override and debug. Use it only in rare, justified cases (e.g., utility classes that *must* override everything).

❌ BAD: Overuse of `!important`
```css
.button {
  background-color: red !important;
}
```

✅ GOOD: Manage specificity carefully, use BEM, and rely on the cascade.
```css
/* Adjust specificity or order */
.button--primary {
  background-color: var(--color-primary);
}
```

### 5.2 Avoid ID Selectors in CSS

IDs have extremely high specificity, leading to specificity wars and reduced reusability. Reserve IDs for JavaScript hooks or fragment identifiers.

❌ BAD: Styling with IDs
```css
#main-nav {
  /* ... */
}
```

✅ GOOD: Use classes instead
```css
.main-nav {
  /* ... */
}
```

### 5.3 No Inline Styles

Inline styles are difficult to manage, override, and maintain. Keep all styling in stylesheets.

❌ BAD: Inline styles
```html
<div style="color: red; font-size: 16px;">Hello</div>
```

✅ GOOD: Class-based styling
```html
<div class="text-error text-base">Hello</div>
```
```css
.text-error { color: red; }
.text-base { font-size: 1rem; }
```

### 5.4 Use Shorthand Properties

Use shorthand properties where appropriate to reduce file size and improve readability.

❌ BAD: Longhand properties
```css
.box {
  margin-top: 10px;
  margin-right: 20px;
  margin-bottom: 10px;
  margin-left: 20px;
}
```

✅ GOOD: Shorthand properties
```css
.box {
  margin: 10px 20px;
}
```

### 5.5 Browser Compatibility (`@supports`)

Use `@supports` to provide progressive enhancements for modern CSS features, with fallbacks for older browsers.

```css
/* Fallback for older browsers */
.gallery {
  display: block; /* Default to block layout */
}

/* Enhanced layout with Grid */
@supports (display: grid) {
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }
}
```