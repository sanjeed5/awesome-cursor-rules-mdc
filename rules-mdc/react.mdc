---
description: Definitive guidelines for writing idiomatic, maintainable, and performant React applications using modern best practices and TypeScript.
globs: **/*.{jsx,tsx}
---
# react Best Practices

This guide outlines the non-negotiable standards for building React applications within our team. Adherence ensures predictable behavior, simplifies debugging, and enables future optimizations.

## 1. Core React Principles: Purity & Rules of Hooks

Components and Hooks **must be pure**. They should always return the same output given the same inputs (props, state, context) and not cause side effects during rendering. Obey the [Rules of Hooks](https://react.dev/reference/rules/rules-of-hooks) without exception.

### ❌ BAD: Impure component / Side effect in render
```tsx
function ProductList({ products }) {
  // ❌ Modifies external data during render
  products.sort((a, b) => a.name.localeCompare(b.name));
  return (/* ... */);
}

function MyComponent() {
  // ❌ Hook called conditionally
  if (Math.random() > 0.5) {
    const [count, setCount] = useState(0);
  }
  return (/* ... */);
}
```

### ✅ GOOD: Pure component / Correct Hook usage
```tsx
import { useMemo, useState } from 'react';

function ProductList({ products }) {
  // ✅ Sort data immutably or memoize if expensive
  const sortedProducts = useMemo(() => 
    [...products].sort((a, b) => a.name.localeCompare(b.name)),
    [products]
  );
  return (/* ... */);
}

function MyComponent() {
  // ✅ Hooks always at the top level
  const [count, setCount] = useState(0); 
  // ... conditional logic after hooks
  return (/* ... */);
}
```

## 2. Code Organization & Naming

Organize code by **feature** using the `bulletproof-react` pattern. Use TypeScript (`.tsx`) for all components.

*   **One Component Per File**: Except for small, pure, stateless components closely related to a parent.
*   **Naming**:
    *   Components: `PascalCase` (e.g., `UserProfile.tsx`)
    *   Custom Hooks: `use` prefix + `PascalCase` (e.g., `useAuth.ts`)
    *   Functions/Variables: `camelCase`
    *   CSS Classes: `kebab-case` (via CSS Modules or utility classes)

### ✅ GOOD: Feature-based structure
```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   │   ├── LoginForm.tsx
│   │   │   └── AuthButton.tsx
│   │   ├── hooks/
│   │   │   └── useAuth.ts
│   │   └── api/auth.ts
│   └── products/
│       ├── components/
│       │   ├── ProductCard.tsx
│       │   └── ProductList.tsx
│       └── hooks/useProducts.ts
├── components/ui/  // Reusable, generic UI components
│   ├── Button.tsx
│   └── Modal.tsx
└── App.tsx
```

## 3. Component Design & Patterns

Prioritize **function components with Hooks**. Separate concerns into "smart" (data/logic) and "dumb" (presentational) components.

### ❌ BAD: Class components / Mixed concerns
```tsx
// ❌ Class component (avoid)
class UserProfile extends React.Component { /* ... */ }

// ❌ Component fetches data AND renders complex UI
function ProductPage() {
  const [products, setProducts] = useState([]);
  useEffect(() => { /* fetch products */ }, []);
  return (/* complex product list UI */);
}
```

### ✅ GOOD: Function components / Separation of concerns
```tsx
// ✅ Use function components with hooks
function UserProfile({ user }) { /* ... */ }

// ✅ Smart component (container) handles data fetching
function ProductListContainer() {
  const { products, isLoading } = useProducts(); // Custom hook for data
  if (isLoading) return <LoadingSpinner />;
  return <ProductList products={products} />; // Renders dumb component
}

// ✅ Dumb component (presentational) focuses on UI
function ProductList({ products }) {
  return (
    <ul>
      {products.map(product => <ProductCard key={product.id} product={product} />)}
    </ul>
  );
}
```

## 4. State Management

Start with **local state (`useState`, `useReducer`)**. Lift state up when necessary. Use **Context API** for global state that rarely changes. For complex global state, use dedicated libraries (e.g., Zustand, Jotai, Redux Toolkit). **Avoid prop drilling.**

### ❌ BAD: Prop drilling
```tsx
function Grandparent() {
  const [theme, setTheme] = useState('dark');
  return <Parent theme={theme} setTheme={setTheme} />;
}
function Parent({ theme, setTheme }) {
  return <Child theme={theme} setTheme={setTheme} />;
}
function Child({ theme, setTheme }) {
  return <Button onClick={() => setTheme('light')}>Toggle Theme</Button>;
}
```

### ✅ GOOD: Context API for global state
```tsx
import { createContext, useContext, useState, ReactNode } from 'react';

type Theme = 'light' | 'dark';
type ThemeContextType = { theme: Theme; toggleTheme: () => void };

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark');
  const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Usage:
function MyComponent() {
  const { theme, toggleTheme } = useTheme();
  return <button onClick={toggleTheme}>Current theme: {theme}</button>;
}
```

## 5. Performance & Optimization

Optimize only when profiling indicates a bottleneck. Use `React.memo`, `useCallback`, `useMemo` judiciously.

### ❌ BAD: Premature optimization / Incorrect memoization
```tsx
// ❌ Memoizing a component that re-renders frequently or has no expensive props
const MyButton = React.memo(({ onClick, children }) => <button onClick={onClick}>{children}</button>);

// ❌ Callback with missing dependency, causing stale closure
function Parent() {
  const [count, setCount] = useState(0);
  const handleClick = useCallback(() => {
    console.log(count); // ❌ 'count' is stale if not in dependency array
  }, []); 
  return <Child onClick={handleClick} />;
}
```

### ✅ GOOD: Targeted optimization / Correct dependencies
```tsx
import React, { useCallback, useMemo, useState } from 'react';

// ✅ Memoize only if component is expensive AND props are stable
const ExpensiveList = React.memo(({ items }) => {
  console.log('Rendering ExpensiveList');
  return (/* ... render many items ... */);
});

function Parent() {
  const [count, setCount] = useState(0);
  // ✅ Callback with correct dependencies
  const handleClick = useCallback(() => {
    setCount(prev => prev + 1); // Use functional update to avoid 'count' in deps
  }, []); 

  // ✅ Memoize expensive calculations
  const computedValue = useMemo(() => {
    // ... heavy computation based on count ...
    return count * 2;
  }, [count]);

  return (
    <>
      <ExpensiveList items={[{ id: 1, name: 'Item 1' }]} /> {/* Example usage */}
      <Child onClick={handleClick} />
      <p>Count: {count}, Computed: {computedValue}</p>
    </>
  );
}
// Child component that receives the memoized callback
const Child = React.memo(({ onClick }: { onClick: () => void }) => {
  console.log('Rendering Child');
  return <button onClick={onClick}>Increment</button>;
});
```

## 6. Common Pitfalls

*   **Never mutate props or state directly.** Always create new objects/arrays.
*   **Never call component functions directly.** Use JSX.
*   **Ensure `useEffect` cleanup functions are always provided** for subscriptions or timers.
*   **Correct `useEffect` dependency arrays** are critical to avoid infinite loops or stale closures.

### ❌ BAD: Direct mutation / Calling component as function
```tsx
function MyComponent({ items }) {
  // ❌ Mutating props directly
  items.push('new item'); 

  const [data, setData] = useState({ value: 1 });
  // ❌ Mutating state directly
  data.value = 2; 
  setData(data);

  // ❌ Calling component as a function
  return MyOtherComponent(); 
}
```

### ✅ GOOD: Immutable updates / JSX usage
```tsx
function MyComponent({ items }) {
  const [data, setData] = useState({ value: 1 });

  // ✅ Create a new array for updates
  const updatedItems = [...items, 'new item']; 

  // ✅ Create a new object for state updates
  setData(prevData => ({ ...prevData, value: 2 }));

  // ✅ Use JSX for components
  return <MyOtherComponent />;
}
```

## 7. Accessibility (A11y) & Testing

Build for accessibility from the start. Test components as a user would.

*   **Semantic HTML**: Use native HTML elements (`<button>`, `<input>`, `<a>`) whenever possible.
*   **ARIA Attributes**: Use `aria-*` attributes only when semantic HTML is insufficient.
*   **Keyboard Navigation**: Ensure all interactive elements are keyboard accessible and have proper focus management.
*   **React Testing Library**: Use `RTL` to test component behavior, not implementation details.

### ❌ BAD: Non-semantic HTML / Untestable implementation
```tsx
// ❌ Div acting as a button, missing keyboard interaction
function MyButton() {
  return <div onClick={() => alert('Clicked!')}>Click Me</div>;
}

// ❌ Testing internal state or component instance (implementation detail)
test('MyComponent sets count to 1', () => {
  const { instance } = render(<MyComponent />);
  expect(instance.state.count).toBe(1); // ❌ Avoid
});
```

### ✅ GOOD: Semantic HTML / User-centric testing
```tsx
import { render, screen, fireEvent } from '@testing-library/react';

// ✅ Proper button element with click handler
function MyButton() {
  return <button type="button" onClick={() => alert('Clicked!')}>Click Me</button>;
}

// ✅ Test user interaction and visible output
test('MyButton alerts on click', () => {
  render(<MyButton />);
  fireEvent.click(screen.getByRole('button', { name: /click me/i }));
  expect(window.alert).toHaveBeenCalledWith('Clicked!'); // Assuming alert is mocked
});
```