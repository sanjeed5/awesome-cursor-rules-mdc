---
description: This guide outlines definitive best practices for building high-performance, resumable Qwik applications, focusing on maximizing lazy loading and ensuring optimal developer experience.
globs: **/*.{js,jsx,ts,tsx}
---
# qwik Best Practices

Qwik is an edge-first framework built on **resumability**, not hydration. This means your application's JavaScript is only loaded and executed when absolutely necessary, leading to instant Time-To-Interactive (TTI). Our guidelines prioritize this core principle.

## Critical Guidelines:

### 1. Define Every UI Piece with `component$`

All interactive UI components **must** be wrapped in `component$`. This is Qwik's fundamental primitive for enabling automatic code-splitting and lazy loading.

❌ **BAD: Regular function component**
```typescript
// my-button.tsx
export const MyButton = () => { // Missing '$'
  return <button onClick={() => alert('Clicked!')}>Click me</button>;
};
```

✅ **GOOD: `component$` for resumability**
```typescript
// my-button.tsx
import { component$ } from '@builder.io/qwik';

export const MyButton = component$(() => {
  return <button onClick$={() => alert('Clicked!')}>Click me</button>; // onClick$ is also a QRL
});
```

### 2. Use `useStore` for Reactive State

Manage component-local reactive state using `useStore`. This ensures state is serializable and resumable across server-side rendering and client-side interactions. Always type your stores for clarity and safety.

❌ **BAD: Non-reactive or non-serializable state**
```typescript
// counter.tsx
import { component$ } from '@builder.io/qwik';

export const Counter = component$(() => {
  let count = 0; // Not reactive, won't persist across renders/resumability
  const increment = () => { count++; };
  return <button onClick$={increment}>{count}</button>;
});
```

✅ **GOOD: `useStore` with TypeScript interface**
```typescript
// counter.tsx
import { component$, useStore } from '@builder.io/qwik';

interface CounterStore {
  count: number;
}

export const Counter = component$(() => {
  const store = useStore<CounterStore>({ count: 0 });
  return (
    <button onClick$={() => store.count++}>
      Count: {store.count}
    </button>
  );
});
```

### 3. Defer Client-Side Effects with `useVisibleTask$`

Execute client-specific logic, such as third-party library initialization or DOM manipulation, only when a component becomes visible in the viewport. This prevents unnecessary JavaScript execution on initial load. For effects that *must* run on the client regardless of visibility, use `useClientEffect$`.

❌ **BAD: Eager client-side effect in component render**
```typescript
// chart-component.tsx
import { component$ } from '@builder.io/qwik';
import Chart from 'chart.js'; // This will be bundled and executed eagerly

export const ChartComponent = component$(() => {
  // This code runs on the server and client, potentially causing errors or unnecessary work
  const canvasRef = useRef<HTMLCanvasElement>();
  useEffect(() => { // React-style hook, not Qwik-native
    if (canvasRef.current) {
      new Chart(canvasRef.current, { /* ... */ });
    }
  }, []);
  return <canvas ref={canvasRef}></canvas>;
});
```

✅ **GOOD: `useVisibleTask$` for client-side effects**
```typescript
// chart-component.tsx
import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
// Import client-side library dynamically or ensure it's tree-shaken
import type Chart from 'chart.js'; // Use type import to avoid bundling

export const ChartComponent = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();

  useVisibleTask$(({ track }) => {
    track(() => canvasRef.value); // Re-run if canvasRef changes
    if (canvasRef.value) {
      // Dynamically import Chart.js to ensure it's only loaded on the client when needed
      import('chart.js').then(({ default: Chart }) => {
        new Chart(canvasRef.value!, {
          type: 'bar',
          data: {
            labels: ['Red', 'Blue', 'Yellow'],
            datasets: [{
              label: '# of Votes',
              data: [12, 19, 3],
              backgroundColor: ['red', 'blue', 'yellow'],
            }],
          },
        });
      });
    }
  });

  return <canvas ref={canvasRef}></canvas>;
});
```

### 4. Use `$` Suffix for All QRLs

Any function that needs to be code-split and lazy-loaded by Qwik's optimizer **must** end with a `$` suffix. This includes components (`component$`), event handlers (`onClick$`, `onInput$`), and hooks (`useStore`, `useTask$`, `useVisibleTask$`, `useClientEffect$`, `useResource$`, `useSignal$`).

❌ **BAD: Missing `$` suffix on event handler**
```typescript
// button.tsx
import { component$ } from '@builder.io/qwik';

export const MyButton = component$(() => {
  const handleClick = () => { // Missing '$'
    console.log('Button clicked!');
  };
  return <button onClick={handleClick}>Click me</button>; // Will not be lazy-loaded
});
```

✅ **GOOD: Correct `$` suffix for QRLs**
```typescript
// button.tsx
import { component$ } from '@builder.io/qwik';

export const MyButton = component$(() => {
  const handleClick = component$(() => { // Even internal functions can be QRLs
    console.log('Button clicked!');
  });
  return <button onClick$={handleClick}>Click me</button>; // Correctly lazy-loaded
});
```

### 5. Leverage Qwik City for Routing and Data Loading

For applications using Qwik City, utilize its file-system based routing, layouts, and `loader$` functions for efficient data fetching. This provides a zero-overhead solution for server-side data loading and routing.

❌ **BAD: Manual routing or client-side data fetching for initial load**
```typescript
// routes/products/index.tsx
import { component$, useClientEffect$, useStore } from '@builder.io/qwik';

export default component$(() => {
  const store = useStore({ products: [] });
  useClientEffect$(async () => { // Data fetched on client, after hydration
    const res = await fetch('/api/products');
    store.products = await res.json();
  });
  return <div>{/* ... render products ... */}</div>;
});
```

✅ **GOOD: `loader$` for server-side data fetching**
```typescript
// routes/products/index.tsx
import { component$ } from '@builder.io/qwik';
import { routeLoader$ } from '@builder.io/qwik-city';

interface Product {
  id: string;
  name: string;
}

// Data fetched on the server during SSR, then serialized and resumed on client
export const useProducts = routeLoader$(async () => {
  const res = await fetch('https://api.example.com/products'); // Fetch from an API
  const products: Product[] = await res.json();
  return products;
});

export default component$(() => {
  const products = useProducts(); // Access the loaded data
  return (
    <div>
      <h1>Products</h1>
      <ul>
        {products.value.map((product) => (
          <li key={product.id}>{product.name}</li>
        ))}
      </ul>
    </div>
  );
});
```

### 6. Use `PropFunction` for Passing Functions as Props

When passing a function as a prop to a child component that itself is a `component$`, ensure the prop type is `PropFunction<(...args: any[]) => any>`. This signals to Qwik that the function is a QRL and should be serialized and lazy-loaded.

❌ **BAD: Passing a regular function type as a prop**
```typescript
// child-button.tsx
import { component$ } from '@builder.io/qwik';

interface ChildButtonProps {
  onClick: () => void; // Not a PropFunction
}

export const ChildButton = component$((props: ChildButtonProps) => {
  return <button onClick$={props.onClick}>Child Button</button>;
});

// parent-component.tsx
import { component$ } from '@builder.io/qwik';
import { ChildButton } from './child-button';

export const ParentComponent = component$(() => {
  const handleParentClick = component$(() => console.log('Parent handled click'));
  return <ChildButton onClick={handleParentClick} />; // Type mismatch, potential issues
});
```

✅ **GOOD: Using `PropFunction` for function props**
```typescript
// child-button.tsx
import { component$, type PropFunction } from '@builder.io/qwik';

interface ChildButtonProps {
  onClick$: PropFunction<() => void>; // Correctly typed as a PropFunction
}

export const ChildButton = component$((props: ChildButtonProps) => {
  return <button onClick$={props.onClick$}>Child Button</button>;
});

// parent-component.tsx
import { component$ } from '@builder.io/qwik';
import { ChildButton } from './child-button';

export const ParentComponent = component$(() => {
  const handleParentClick = component$(() => console.log('Parent handled click'));
  return <ChildButton onClick$={handleParentClick} />; // Correctly passing QRL
});
```

### 7. Adopt Atomic CSS (e.g., Tailwind CSS)

Integrate styling solutions like Tailwind CSS that generate static, utility-first classes at build time. This avoids runtime style calculations and keeps CSS payloads minimal, aligning with Qwik's performance goals.

❌ **BAD: Runtime CSS-in-JS or complex dynamic styling**
```typescript
// styled-button.tsx
import { component$ } from '@builder.io/qwik';
import { css } from '@emotion/css'; // Runtime CSS-in-JS

const buttonStyle = css`
  background-color: blue;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  &:hover {
    background-color: darkblue;
  }
`;

export const StyledButton = component$(() => {
  return <button class={buttonStyle}>Styled Button</button>;
});
```

✅ **GOOD: Tailwind CSS for static styling**
```typescript
// tailwind-button.tsx
import { component$ } from '@builder.io/qwik';

export const TailwindButton = component$(() => {
  return (
    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
      Tailwind Button
    </button>
  );
});
```

### 8. Use Vitest and Playwright for Testing

Leverage Qwik's built-in testing stack: Vitest for unit and component tests, and Playwright for end-to-end scenarios. These tools are integrated and optimized for Qwik's unique architecture, ensuring your lazy-loading and resumability work correctly.

❌ **BAD: Using Jest/Enzyme for Qwik component testing**
```typescript
// my-component.test.js (using Jest)
import { render } from '@testing-library/react'; // Incorrect library for Qwik
import { MyComponent } from './my-component';

test('MyComponent renders correctly', () => {
  const { getByText } = render(<MyComponent />);
  expect(getByText('Hello')).toBeInTheDocument();
});
```

✅ **GOOD: Vitest for Qwik component testing**
```typescript
// my-component.test.tsx (using Vitest)
import { render, screen } from '@builder.io/qwik/testing';
import { component$ } from '@builder.io/qwik';
import { MyComponent } from './my-component'; // Assuming MyComponent is a Qwik component

describe('MyComponent', () => {
  it('should render correctly', async () => {
    await render(<MyComponent />);
    expect(screen.getByText('Hello Qwik!')).toBeInTheDocument();
  });

  it('should handle click event', async () => {
    const mockFn = vi.fn();
    const TestComponent = component$(() => {
      return <button onClick$={mockFn}>Click Me</button>;
    });

    await render(<TestComponent />);
    await screen.getByRole('button', { name: 'Click Me' }).click();
    expect(mockFn).toHaveBeenCalledTimes(1);
  });
});
```