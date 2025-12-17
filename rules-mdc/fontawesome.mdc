---
description: This guide provides definitive rules for integrating and utilizing Font Awesome 7 icons, focusing on accessibility, performance, and maintainability through Kit-based setup and modern best practices.
globs: **/*
---
# fontawesome Best Practices

Font Awesome 7 is the definitive source for scalable UI icons in our web applications. Leveraging its power correctly ensures visual consistency, accessibility, and optimal performance. This guide outlines our team's mandatory best practices for Font Awesome integration.

## 1. Code Organization and Structure

Always use a Font Awesome Kit for integration. It's the most robust and feature-rich method, providing auto-accessibility and simplified management.

### 1.1 Kit-Based Setup

**Always embed your Font Awesome Kit script in the `<head>` of every page.** This ensures icons load early, prevents FOUT (Flash of Unstyled Text/Icons), and enables critical auto-accessibility features.

❌ **BAD: Script in `<body>` or via local files (unless explicitly self-hosting)**
```html
<body>
  <!-- ... other content ... -->
  <script src="https://kit.fontawesome.com/[YOUR_KIT_CODE].js" crossorigin="anonymous"></script>
</body>
```

✅ **GOOD: Script in `<head>`**
```html
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Awesome App</title>
  <!-- Font Awesome Kit must be here -->
  <script src="https://kit.fontawesome.com/[YOUR_KIT_CODE].js" crossorigin="anonymous"></script>
  <!-- ... other head elements ... -->
</head>
```

### 1.2 Icon Markup

Use the `<i>` tag with appropriate Font Awesome classes. This is the simplest and most widely supported method.

❌ **BAD: Using `<span>` or other elements without a strong reason**
```html
<span class="fa-solid fa-user"></span>
```

✅ **GOOD: Using `<i>` for standard icons**
```html
<i class="fa-solid fa-user"></i>
```

### 1.3 Custom Icons

For icons not available in Font Awesome, upload them to your Kit. Treat them exactly like native icons for consistency.

❌ **BAD: Manually embedding custom SVGs or using separate icon libraries**
```html
<img src="/assets/custom-icon.svg" alt="Custom Icon">
```

✅ **GOOD: Uploading custom SVGs to your Kit and using `fa-kit` prefix**
```html
<!-- In your Kit, you've uploaded 'my-custom-icon.svg' -->
<i class="fa-kit fa-my-custom-icon"></i>
<!-- For duotone custom icons -->
<i class="fa-kit-duotone fa-my-duotone-icon"></i>
```

## 2. Common Patterns and Anti-patterns

### 2.1 Icon Styles

Always specify the icon style (e.g., `fa-solid`, `fa-regular`, `fa-brands`). Consistency in style is crucial for a unified UI.

❌ **BAD: Omitting style, relying on default (which can change or be ambiguous)**
```html
<i class="fa-user"></i> <!-- Which style? -->
```

✅ **GOOD: Explicitly defining the style**
```html
<i class="fa-solid fa-user"></i>
<i class="fa-regular fa-star"></i>
<i class="fa-brands fa-github"></i>
<i class="fa-sharp fa-solid fa-gear"></i>
```

### 2.2 Sizing Icons

Utilize Font Awesome's sizing utilities to maintain consistent vertical alignment and scale. Avoid custom CSS `font-size` overrides unless absolutely necessary for unique cases.

#### 2.2.1 Relative Sizing (for UI elements)

Use `fa-xs` to `fa-2xl` when icons need to align with text or other UI components. These classes adjust size while preserving vertical alignment.

❌ **BAD: Custom CSS `font-size` for inline icons**
```html
<p><i class="fa-solid fa-coffee" style="font-size: 1.2em;"></i> My coffee</p>
```

✅ **GOOD: Relative sizing classes for text alignment**
```html
<p><i class="fa-solid fa-coffee fa-lg"></i> My coffee</p>
<button><i class="fa-solid fa-plus fa-sm"></i> Add Item</button>
```

#### 2.2.2 Literal Sizing (for standalone icons)

Use `fa-1x` to `fa-10x` for larger, standalone icons where precise text alignment is not the primary concern.

❌ **BAD: Scaling large icons with `em` units that might break alignment**
```html
<div style="font-size: 5em;"><i class="fa-solid fa-camera"></i></div>
```

✅ **GOOD: Literal sizing classes for large, independent icons**
```html
<div class="hero-icon">
  <i class="fa-solid fa-camera fa-5x"></i>
  <h3>Capture the Moment</h3>
</div>
```

## 3. Performance Considerations

### 3.1 Kit Configuration

**Always configure your Kit to load only the icon subsets you actually use.** This significantly reduces download size and improves page load times. Regularly review and prune unused icon styles or categories.

❌ **BAD: Using a Kit with all icon styles enabled by default**
(This is a configuration anti-pattern, not directly code-related, but impacts code performance.)

✅ **GOOD: Customizing Kit to include only `Solid` and `Brands` if those are the only styles used.**
(Action: Visit fontawesome.com/kits, configure your kit.)

### 3.2 Self-Hosting (Advanced)

Only consider self-hosting Font Awesome if you have strict performance requirements, advanced caching strategies, and full control over your asset delivery. For most projects, Kit-based CDN is superior.

❌ **BAD: Self-hosting without a clear performance benefit or maintenance plan**
(Increased complexity, manual updates, potential for outdated versions.)

✅ **GOOD: Kit-based CDN for most projects; self-hosting only with justification and expertise.**
(Action: Stick to Kit unless a performance audit explicitly demands self-hosting and your team has the resources.)

## 4. Common Pitfalls and Gotchas

### 4.1 Missing Kit Script

Icons won't render if the Kit script is missing or incorrectly placed. Always verify its presence in the `<head>`.

❌ **BAD: Icons showing as squares or empty space**
```html
<!-- No Font Awesome script in head -->
<i class="fa-solid fa-bug"></i> <!-- Renders as a square -->
```

✅ **GOOD: Verify script is loaded and accessible**
(Action: Check browser console for network errors related to the Kit script, ensure correct URL.)

### 4.2 Incorrect Class Names

Typos in icon names or style prefixes are common. Use the Font Awesome search tool to confirm exact class names.

❌ **BAD: Typo in icon name**
```html
<i class="fa-solid fa-calender"></i> <!-- Should be fa-calendar -->
```

✅ **GOOD: Double-checking icon names**
```html
<i class="fa-solid fa-calendar"></i>
```

### 4.3 SVG Upload Guidelines

When uploading custom SVGs to a Kit, adhere strictly to Font Awesome's guidelines to prevent rendering issues.

*   **`viewBox`**: Must have explicit `width` and `height`.
*   **Paths**: Single `<path>` for monotone, two `<path>` elements (one with <100% opacity) for duotone.
*   **No unsupported attributes**: Keep SVGs clean.

❌ **BAD: Uploading complex SVGs with multiple groups, raster images, or missing `viewBox`**
(Leads to broken icons, inconsistent scaling, or upload failures.)

✅ **GOOD: Simplified SVGs, optimized for icon use**
```xml
<!-- Monotone Example -->
<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <path fill="currentColor" d="M0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256z"/>
</svg>

<!-- Duotone Example (simplified) -->
<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <path fill="currentColor" opacity="0.4" d="M... (secondary layer) ..."/>
  <path fill="currentColor" d="M... (primary layer) ..."/>
</svg>
```

## 5. Accessibility

Accessibility is paramount. Font Awesome Kits provide "auto-accessibility" features, but developers must provide the correct context.

### 5.1 Decorative Icons

If an icon is purely visual and adds no unique meaning (e.g., a bullet point in a list), ensure it's hidden from screen readers. Auto-accessibility handles this if the icon is *not* the sole content of an interactive element.

❌ **BAD: Decorative icon without `aria-hidden` when not auto-handled**
```html
<i class="fa-solid fa-star"></i> Welcome!
```

✅ **GOOD: Rely on Auto-Accessibility or explicitly hide decorative icons**
```html
<p><i class="fa-solid fa-star" aria-hidden="true"></i> Welcome to our site!</p>
<!-- Auto-accessibility will add aria-hidden="true" if the icon is adjacent to text -->
<button type="submit">
  <i class="fa-solid fa-envelope"></i> Email Us!
</button>
```

### 5.2 Semantic Icons

If an icon conveys meaning (e.g., a "next page" arrow, a "message" icon in a button), provide an accessible text alternative. Use the `title` attribute, and Font Awesome's auto-accessibility will convert it into a screen-reader-only span.

❌ **BAD: Semantic icon without any accessible label**
```html
<button type="submit">
  <i class="fa-solid fa-arrow-right"></i>
</button>
```

✅ **GOOD: Semantic icon with a descriptive `title` attribute**
```html
<button type="submit">
  <i class="fa-solid fa-arrow-right" title="Next Page"></i>
</button>
<!-- Auto-accessibility will render this as: -->
<!-- <i class="fa-solid fa-arrow-right" aria-hidden="true"></i> -->
<!-- <span class="sr-only">Next Page</span> -->
```

### 5.3 Manual Accessibility (Fallback)

If Auto-Accessibility is unavailable (e.g., self-hosting older versions, specific framework integrations), you must manually apply `aria-hidden="true"` for decorative icons or an `sr-only` span for semantic ones.

❌ **BAD: Assuming auto-accessibility is always active**
(Always verify your setup or explicitly add attributes.)

✅ **GOOD: Understanding when manual intervention is needed**
```html
<!-- Manual decorative -->
<i class="fa-solid fa-bell" aria-hidden="true"></i> Notifications

<!-- Manual semantic -->
<button>
  <i class="fa-solid fa-trash" aria-hidden="true"></i>
  <span class="sr-only">Delete Item</span>
</button>
```

## 6. Theming

Font Awesome icons inherit `font-size` and `color` from their parent elements. Leverage this for consistent theming.

❌ **BAD: Applying inline styles or overly specific CSS to individual icons**
```html
<i class="fa-solid fa-gear" style="color: blue; font-size: 24px;"></i>
```

✅ **GOOD: Styling icons via parent elements or utility classes**
```css
/* In your CSS */
.primary-button .fa-icon {
  color: var(--color-primary);
}
.large-heading .fa-icon {
  font-size: 1.2em; /* Scales relative to heading */
}
```
```html
<button class="primary-button">
  <i class="fa-solid fa-gear fa-icon"></i> Settings
</button>
<h1 class="large-heading">
  <i class="fa-solid fa-chart-line fa-icon"></i> Dashboard
</h1>
```

## 7. Component Architecture

When integrating Font Awesome into component-based frameworks (React, Vue, Angular), **always use the official Font Awesome component libraries** for that framework. They handle SVG rendering, accessibility, and updates gracefully.

❌ **BAD: Manually embedding `<i>` tags within framework components without official wrappers**
```jsx
// React example
function MyButton() {
  return <button><i className="fa-solid fa-save"></i> Save</button>;
}
```

✅ **GOOD: Using official Font Awesome components**
```jsx
// React example with @fortawesome/react-fontawesome
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSave } from '@fortawesome/free-solid-svg-icons';

function MyButton() {
  return <button><FontAwesomeIcon icon={faSave} /> Save</button>;
}
```

## 8. Testing Approaches

### 8.1 Visual Regression Testing

Include Font Awesome icons in your visual regression tests (e.g., Storybook with Chromatic, Playwright visual comparisons). This catches unexpected changes in icon rendering, sizing, or alignment.

### 8.2 Accessibility Testing

Regularly run automated accessibility checks (e.g., Axe-core, Lighthouse) on pages containing icons. Manually test with screen readers to ensure semantic icons are correctly announced and decorative icons are ignored.

❌ **BAD: Assuming icons "look fine" and skipping visual/accessibility tests**
(Icons are critical UI elements and can easily break or become inaccessible.)

✅ **GOOD: Integrating icon rendering and accessibility into CI/CD pipelines.**
(Action: Add visual regression and accessibility checks to your component stories or end-to-end tests.)