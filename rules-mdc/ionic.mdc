---
description: This guide provides definitive best practices for developing high-performance, maintainable, and accessible Ionic applications using modern TypeScript and web standards.
globs: **/*.{js,jsx,ts,tsx}
---
# ionic Best Practices

Ionic Framework enables powerful cross-platform applications. Adhering to these guidelines ensures your codebase is robust, performant, and aligned with modern web development standards.

## 1. Code Organization and Structure

Adopt a consistent, component-centric file structure for clarity and maintainability.

### 1.1 One Component Per Directory

Each Ionic component (or Stencil component it wraps) must reside in its own directory, co-locating all related files.

❌ **BAD: Scattered files**
```
// components/my-button.tsx
// components/my-button.css
// components/my-card.tsx
// components/my-card.css
```

✅ **GOOD: Component-per-directory**
```
├── components
│   ├── ion-button
│   │   ├── ion-button.css
│   │   └── ion-button.tsx
│   └── ion-card
│       ├── ion-card.css
│       └── ion-card.tsx
```

### 1.2 Naming Conventions

Use descriptive, noun-based names for components, prefixed for uniqueness.

*   **HTML Tag**: Use a unique, brand-specific prefix (e.g., `ion-`). Names should be nouns.
*   **TS Class**: No prefix for the ES6 class name, as classes are scoped.

❌ **BAD: Verb-based or un-prefixed tags**
```tsx
// my-animating.tsx
@Component({ tag: 'animating-component' })
export class AnimatingComponent {}
```

✅ **GOOD: Noun-based, prefixed tags**
```tsx
// ion-animation.tsx
@Component({ tag: 'ion-animation' })
export class Animation {}
```

### 1.3 Component File Structure (Newspaper Metaphor)

Organize component class members logically, from high-level summaries to detailed implementations.

```typescript
@Component({
  tag: 'ion-example',
  styleUrls: {
    ios: 'ion-example.ios.css',
    md: 'ion-example.md.css',
  },
})
export class IonExample {
  /** 1. Internal Properties (not exposed) */
  private internalValue: number = 0;

  /** 2. Reference to host HTML element */
  @Element() el!: HTMLElement;

  /** 3. State() variables (internal reactive state) */
  @State() isActive: boolean = false;

  /** 4. Public Property API (@Prop()) with JSDocs */
  /**
   * The text to display on the button.
   */
  @Prop() text: string = '';

  /**
   * If `true`, the button is disabled.
   */
  @Prop() disabled: boolean = false;

  /** 5. Prop lifecycle events (@Watch()) */
  @Watch('disabled')
  disabledChanged(newValue: boolean, oldValue: boolean) {
    console.log(`Disabled changed from ${oldValue} to ${newValue}`);
  }

  /** 6. Events section (@Event()) with JSDocs */
  /**
   * Emitted when the button is clicked.
   */
  @Event() ionClick!: EventEmitter<void>;

  /** 7. Component lifecycle events (ordered by natural call order) */
  connectedCallback() { /* ... */ }
  disconnectedCallback() { /* ... */ }
  componentWillLoad() { /* ... */ }
  componentDidLoad() { /* ... */ }
  componentWillRender() { /* ... */ }
  componentDidRender() { /* ... */ }

  /** 8. Listeners (@Listen()) */
  @Listen('click', { target: 'document' })
  onClick(event: Event) {
    if (event.target === this.el) {
      this.ionClick.emit();
    }
  }

  /** 9. Public Methods (@Method()) with JSDocs */
  /**
   * Focuses the button.
   */
  @Method()
  async setFocus() {
    this.el.focus();
  }

  /** 10. Private Methods (internal logic) */
  private updateInternalState() {
    this.internalValue++;
  }

  /** 11. Render function (JSX/TSX) */
  render() {
    return (
      <button disabled={this.disabled} onClick={() => this.ionClick.emit()}>
        {this.text}
      </button>
    );
  }
}
```

## 2. TypeScript Best Practices

Leverage TypeScript for type safety, improved tooling, and maintainability.

### 2.1 Enforce Strict Typing

Always enable `noImplicitAny` and `strictNullChecks` in your `tsconfig.json`. Avoid `any` unless absolutely necessary.

❌ **BAD: Implicit `any`**
```typescript
function processData(data) { // data is implicitly 'any'
  console.log(data.value);
}
```

✅ **GOOD: Explicit typing**
```typescript
interface Data {
  value: string;
}
function processData(data: Data): void {
  console.log(data.value);
}
```

### 2.2 Use `private` for Internal Members

Mark internal class properties and methods as `private` to enforce encapsulation and aid dead code detection.

❌ **BAD: All public by default**
```typescript
export class MyComponent {
  helperMethod() { /* ... */ } // Public, but only used internally
}
```

✅ **GOOD: Encapsulated internals**
```typescript
export class MyComponent {
  private helperMethod() { /* ... */ } // Clearly internal
}
```

### 2.3 JSDocs for Public API

Document all `@Prop()`, `@Event()`, and `@Method()` decorators with JSDoc comments to generate documentation and improve editor experience.

❌ **BAD: Undocumented public API**
```typescript
@Prop() value: string;
@Event() ionChange: EventEmitter<string>;
```

✅ **GOOD: Documented public API**
```typescript
/**
 * The current value of the input.
 */
@Prop() value: string = '';

/**
 * Emitted when the value changes.
 */
@Event() ionChange!: EventEmitter<string>;
```

## 3. Common Patterns and Anti-patterns

Adopt patterns that leverage Ionic's strengths and avoid common pitfalls.

### 3.1 Always `await` Asynchronous Native Plugin Calls

Capacitor/Cordova plugin calls are asynchronous. Always `await` their resolution to prevent race conditions and unexpected behavior.

❌ **BAD: Fire-and-forget native calls**
```typescript
import { Camera } from '@capacitor/camera';

async takePhoto() {
  Camera.getPhoto({ /* ... */ }); // Missing await
  console.log('Photo request sent'); // May log before photo is taken
}
```

✅ **GOOD: Await native calls**
```typescript
import { Camera, CameraResultType } from '@capacitor/camera';

async takePhoto() {
  const photo = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Uri
  });
  console.log('Photo taken:', photo.webPath);
}
```

### 3.2 Prefer Ionic Components for UI

Leverage Ionic's rich set of UI components for adaptive styling, performance, and accessibility. Avoid custom implementations where an Ionic component exists.

❌ **BAD: Custom button for navigation**
```html
<button onclick="navigateTo('/home')">Home</button>
```

✅ **GOOD: Ionic button with router link**
```html
<ion-button routerLink="/home" routerDirection="forward">Home</ion-button>
```

### 3.3 Theming with CSS Variables

Customize your app's look and feel using Ionic's CSS Variables for consistent, platform-adaptive theming.

❌ **BAD: Hardcoded styles**
```css
.my-component {
  background-color: #3880ff; /* Primary blue */
}
```

✅ **GOOD: CSS Variables for theming**
```css
.my-component {
  background-color: var(--ion-color-primary);
}
```

## 4. Performance Considerations

Ionic is built for performance. Follow these to maintain a fast, responsive app.

### 4.1 Optimize DOM Updates

Avoid direct, heavy DOM manipulation. Let the framework (Angular, React, Vue, Stencil) manage DOM updates efficiently.

❌ **BAD: Manually updating styles on scroll**
```typescript
// In a scroll event listener
this.el.style.transform = `translateY(${scrollTop}px)`; // Triggers layout thrashing
```

✅ **GOOD: Use CSS transforms or Ionic's built-in scroll events**
```typescript
// Use CSS for animations or leverage Ionic's virtual scroll
// Or, if absolutely necessary, batch DOM reads/writes
```

### 4.2 Leverage Built-in Optimizations

Ionic components often include hardware-accelerated transitions, lazy loading, and tree-shaking. Ensure your build process enables these.

*   **Tree-shaking**: Use modern bundlers (Webpack, Rollup) and ES Modules.
*   **Lazy Loading**: For routes and components, especially in Angular/React/Vue.

## 5. Common Pitfalls and Gotchas

Be aware of these common issues to prevent bugs and performance regressions.

### 5.1 Ignoring Accessibility (a11y)

Ionic components are built with accessibility in mind, but developers must use them correctly. Run automated accessibility audits as part of CI.

❌ **BAD: Missing `aria-label` or semantic elements**
```html
<ion-icon name="menu" (click)="openMenu()"></ion-icon>
```

✅ **GOOD: Accessible elements**
```html
<ion-menu-button auto-hide="false"></ion-menu-button>
<!-- Or for a standalone icon that triggers an action -->
<ion-icon name="menu" aria-label="Open menu" (click)="openMenu()"></ion-icon>
```

### 5.2 Using Deprecated TSLint Configurations

TSLint is deprecated. Migrate to ESLint with `@stencil-community/eslint-plugin` and `eslint-config-ionic` for modern static analysis.

❌ **BAD: Relying on `tslint.json`**
```json
// tslint.json
{
  "extends": "tslint-ionic-rules/strict"
}
```

✅ **GOOD: Using ESLint**
```json
// .eslintrc.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@stencil-community/recommended',
    '@ionic/eslint-config/recommended'
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', '@stencil-community'],
  // ... other rules and configs
};
```

## 6. Testing Approaches

Implement a robust testing strategy for reliability and maintainability.

### 6.1 Unit Testing

Use Jest for unit testing individual components, services, and utilities. Focus on isolated logic.

```typescript
// my-service.spec.ts
import { MyService } from './my-service';

describe('MyService', () => {
  let service: MyService;

  beforeEach(() => {
    service = new MyService();
  });

  it('should return the correct value', () => {
    expect(service.calculate(2, 3)).toBe(5);
  });
});
```

### 6.2 End-to-End (E2E) Testing

Use Playwright or Cypress for E2E tests to simulate user interactions across your application.

```javascript
// e2e/home.spec.ts (example with Playwright)
import { test, expect } from '@playwright/test';

test('should navigate to home and display title', async ({ page }) => {
  await page.goto('/home');
  await expect(page.locator('ion-title')).toHaveText('Home');
});
```