---
description: This guide provides definitive, opinionated best practices for writing high-performance, maintainable SolidJS applications, focusing on reactivity, component architecture, and common pitfalls.
globs: **/*.{js,jsx}
---
# solidjs Best Practices

This document outlines the definitive best practices for developing with SolidJS 1.x. Adhere to these guidelines to leverage Solid's fine-grained reactivity, ensure optimal performance, and maintain a consistent, scalable codebase.

---

## 1. Use SolidJS 1.x (Stable)

**Always use the current stable 1.x release.** SolidJS 2.0 is experimental and not ready for production. Avoid any features or APIs mentioned for 2.0.

## 2. Component Architecture & Code Organization

### 2.1. Component File Naming

**Use PascalCase for component filenames.** This clearly distinguishes components from other modules.

*   ❌ BAD: `mybutton.jsx`, `user-profile.jsx`
*   ✅ GOOD: `MyButton.jsx`, `UserProfile.jsx`

### 2.2. Keep Components Lean

**Components should be small and focused.** Extract complex logic or UI parts into separate components or custom primitives.

### 2.3. Colocate Styles

**Use Vite's CSS Modules for component-specific styling.** This provides scoped styles without runtime overhead.

*   ❌ BAD: Global CSS files, CSS-in-JS libraries with runtime.
*   ✅ GOOD:
    ```jsx
    // MyComponent.module.css
    .container {
      background-color: var(--color-primary);
    }
    .text {
      color: white;
    }

    // MyComponent.jsx
    import styles from './MyComponent.module.css';

    function MyComponent() {
      return <div class={styles.container}><span class={styles.text}>Hello</span></div>;
    }
    ```

## 3. Props Handling

SolidJS props are reactive getters. Incorrect handling breaks reactivity.

### 3.1. Never Destructure Props Directly

**Do not destructure `props` at the component's top level.** This immediately loses reactivity. Access props via `props.propertyName`.

*   ❌ BAD:
    ```jsx
    function MyComponent({ name, age }) { // React pattern, breaks Solid reactivity
      return <div>{name} is {age}</div>;
    }
    ```
*   ✅ GOOD:
    ```jsx
    function MyComponent(props) {
      return <div>{props.name} is {props.age}</div>; // Access directly
    }
    ```

### 3.2. Apply Default Props with `mergeProps`

**Use `mergeProps` for default props.** This preserves reactivity and merges objects non-destructively.

*   ❌ BAD:
    ```jsx
    function MyComponent(props) {
      const name = props.name || "Guest"; // Loses reactivity if props.name is a signal
      return <div>Hello, {name}</div>;
    }
    ```
*   ✅ GOOD:
    ```jsx
    import { mergeProps } from 'solid-js';

    function MyComponent(props) {
      const merged = mergeProps({ name: "Guest", greeting: "Hello" }, props);
      return <div>{merged.greeting}, {merged.name}!</div>;
    }
    ```

### 3.3. Use `splitProps` for Spreading

**When spreading props, use `splitProps` to separate known props from rest props.** This is crucial for accessibility and avoiding prop conflicts.

*   ❌ BAD:
    ```jsx
    function MyInput(props) {
      return <input type="text" {...props} />; // Unsafe, can overwrite internal props
    }
    ```
*   ✅ GOOD:
    ```jsx
    import { splitProps } from 'solid-js';

    function MyInput(props) {
      const [local, rest] = splitProps(props, ["label", "onChange"]);
      return (
        <label>
          {local.label}
          <input type="text" onChange={local.onChange} {...rest} />
        </label>
      );
    }
    ```

## 4. State Management

### 4.1. Signals for Local State

**Use `createSignal` for simple, primitive, or local component state.** It's the most performant and idiomatic way to manage reactive values.

*   ❌ BAD: `let count = 0;` (non-reactive)
*   ✅ GOOD:
    ```jsx
    import { createSignal } from 'solid-js';

    function Counter() {
      const [count, setCount] = createSignal(0);
      return <button onClick={() => setCount(c => c + 1)}>{count()}</button>;
    }
    ```

### 4.2. Stores for Complex/Global State

**Use `createStore` for nested, mutable objects or global state.** Stores provide deep reactivity for objects.

*   ❌ BAD: `createSignal({ user: { name: "John" } })` (nested updates are not reactive without manual spreading)
*   ✅ GOOD:
    ```jsx
    import { createStore } from 'solid-js/store';

    const [user, setUser] = createStore({ firstName: "John", lastName: "Doe" });

    function UserProfile() {
      return (
        <div>
          <p>Name: {user.firstName} {user.lastName}</p>
          <button onClick={() => setUser("firstName", "Jane")}>Change Name</button>
        </div>
      );
    }
    ```

## 5. Side Effects & Lifecycle

### 5.1. `createEffect` for Side Effects

**Encapsulate all side effects in `createEffect`.** Solid automatically tracks dependencies, ensuring effects run only when necessary.

*   ❌ BAD: `console.log(mySignal());` outside `createEffect` (runs once)
*   ✅ GOOD:
    ```jsx
    import { createSignal, createEffect } from 'solid-js';

    function Logger() {
      const [value, setValue] = createSignal("initial");
      createEffect(() => {
        console.log("Value changed:", value()); // Runs on initial render and whenever value() changes
      });
      return <input onInput={(e) => setValue(e.target.value)} value={value()} />;
    }
    ```

### 5.2. `onCleanup` for Resource Management

**Use `onCleanup` to dispose of resources (subscriptions, timers, event listeners).** It runs when the reactive scope (component, effect) is destroyed.

*   ❌ BAD: Global `clearInterval` or `removeEventListener` without `onCleanup`.
*   ✅ GOOD:
    ```jsx
    import { createEffect, onCleanup } from 'solid-js';

    function Timer() {
      createEffect(() => {
        const interval = setInterval(() => console.log("tick"), 1000);
        onCleanup(() => clearInterval(interval)); // Cleans up when component/effect unmounts
      });
      return <div>Timer running...</div>;
    }
    ```

## 6. Performance Considerations

### 6.1. Avoid Unnecessary Re-renders

Solid's fine-grained reactivity minimizes re-renders, but be mindful of creating new functions or objects inside JSX that aren't memoized.

*   ❌ BAD:
    ```jsx
    <MyComponent data={{ id: 1 }} /> // Creates new object on every parent update
    ```
*   ✅ GOOD:
    ```jsx
    const data = { id: 1 };
    <MyComponent data={data} /> // Object reference remains stable
    ```

### 6.2. Lazy Loading

**Use dynamic imports for route-level components.** This reduces initial bundle size.

*   ✅ GOOD:
    ```jsx
    import { lazy } from 'solid-js';
    import { Routes, Route } from '@solidjs/router';

    const HomePage = lazy(() => import('./pages/Home'));
    const AboutPage = lazy(() => import('./pages/About'));

    function AppRoutes() {
      return (
        <Routes>
          <Route path="/" component={HomePage} />
          <Route path="/about" component={AboutPage} />
        </Routes>
      );
    }
    ```

## 7. Ecosystem Tools

**Leverage official and community-vetted tools.**

*   **Bundling**: `vite-plugin-solid`
*   **Routing**: `solid-router` (or SolidStart)
*   **Head Management**: `solid-meta`
*   **Linting**: `eslint-plugin-solid`
*   **Testing**: `solid-testing-library`

## 8. Accessibility

**Prioritize semantic HTML and WAI-ARIA attributes.** SolidJS does not abstract away the DOM, making direct accessibility implementation straightforward.

*   ✅ GOOD:
    ```jsx
    <button aria-label="Close dialog" onClick={closeDialog}>X</button>
    <input type="checkbox" id="agree" checked={agreed()} onChange={toggleAgreed} />
    <label for="agree">I agree to terms</label>
    ```

## 9. Testing Approaches

**Use `solid-testing-library` for unit and integration tests.** It provides a user-centric API similar to React Testing Library.

*   ✅ GOOD:
    ```jsx
    import { render, screen } from 'solid-testing-library';
    import { createSignal } from 'solid-js';
    import MyComponent from './MyComponent';

    test('MyComponent displays the correct count', async () => {
      const [count, setCount] = createSignal(0);
      render(() => <MyComponent count={count()} />);

      expect(screen.getByText(/Count: 0/i)).toBeInTheDocument();

      setCount(1);
      await Promise.resolve(); // Wait for Solid's microtask queue
      expect(screen.getByText(/Count: 1/i)).toBeInTheDocument();
    });
    ```