---
description: This rule provides guidelines for using Zustand, a simple and unopinionated state management library, in React applications. It covers best practices for code organization, performance optimization, testing, and common pitfalls to avoid.
globs: *.js,*.jsx,*.ts,*.tsx
---
# Zustand Best Practices

This document outlines best practices for using Zustand in your React applications. Zustand is a simple and unopinionated state management library. Following these guidelines will help you write maintainable, performant, and scalable applications.

## 1. Code Organization and Structure

### 1.1. Directory Structure

Organize your store files in a dedicated directory, such as `store` or `state`, at the root of your project or within a specific feature directory. This enhances discoverability and maintainability.


src/
├── components/
│   ├── ...
├── store/
│   ├── index.ts          # Main store file (optional)
│   ├── bearStore.ts      # Example store
│   ├── fishStore.ts      # Example store
│   └── utils.ts         # Utility functions for stores
├── App.tsx
└── ...


### 1.2. File Naming Conventions

Use descriptive names for your store files, typically reflecting the domain or feature the store manages. For example, `userStore.ts`, `cartStore.js`, or `settingsStore.tsx`.  Use PascalCase for the store name itself (e.g., `UserStore`).

### 1.3. Module Organization

- **Single Store per File:**  Prefer defining one Zustand store per file.  This improves readability and maintainability.
- **Slices Pattern:** For complex stores, consider using the slices pattern to divide the store into smaller, more manageable pieces.  Each slice manages a specific part of the state and its related actions.

typescript
// store/bearStore.ts
import { StateCreator, create } from 'zustand';

interface BearSlice {
  bears: number;
  addBear: () => void;
}

const createBearSlice: StateCreator<BearSlice> = (set) => ({
  bears: 0,
  addBear: () => set((state) => ({ bears: state.bears + 1 })),
});

export const useBearStore = create<BearSlice>()((...a) => ({
  ...createBearSlice(...a),
}));

// Another slice could be in fishStore.ts, etc.


### 1.4. Component Architecture

- **Presentational and Container Components:**  Separate presentational (UI) components from container components that interact with the Zustand store. Container components fetch data from the store and pass it down to presentational components.
- **Hooks for Data Fetching:** Utilize Zustand's `useStore` hook within container components to subscribe to specific parts of the state.

### 1.5. Code Splitting Strategies

- **Lazy Loading Stores:**  Load stores on demand using dynamic imports.  This can reduce the initial bundle size, especially for larger applications.
- **Feature-Based Splitting:** Split your application into feature modules and create separate stores for each feature.  This allows for independent loading and reduces dependencies between different parts of the application.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to Zustand

- **Colocated Actions and State:**  Keep actions and the state they modify within the same store. This promotes encapsulation and makes it easier to understand how the store's state is updated.
- **Selectors:** Use selectors to derive computed values from the store's state. Selectors should be memoized to prevent unnecessary re-renders.

typescript
// store/userStore.ts
import { create } from 'zustand';

interface UserState {
  name: string;
  age: number;
}

interface UserActions {
  setName: (name: string) => void;
  isAdult: () => boolean; // Selector
}

export const useUserStore = create<UserState & UserActions>((set, get) => ({
  name: 'John Doe',
  age: 20,
  setName: (name) => set({ name }),
  isAdult: () => get().age >= 18, // Selector
}));


### 2.2. Recommended Approaches for Common Tasks

- **Asynchronous Actions:** Use `async/await` within actions to handle asynchronous operations such as fetching data from an API.

typescript
interface DataState {
  data: any | null;
  isLoading: boolean;
  fetchData: () => Promise<void>;
}

export const useDataStore = create<DataState>((set) => ({
  data: null,
  isLoading: false,
  fetchData: async () => {
    set({ isLoading: true });
    try {
      const response = await fetch('https://api.example.com/data');
      const data = await response.json();
      set({ data, isLoading: false });
    } catch (error) {
      console.error('Error fetching data:', error);
      set({ isLoading: false, data: null });
    }
  },
}));


- **Persisting State:** Use the `zustand/middleware`'s `persist` middleware to persist the store's state to local storage or another storage mechanism.  Configure a `partialize` function to select the state you want to persist.

typescript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  token: string | null;
  setToken: (token: string | null) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      setToken: (token) => set({ token }),
    }),
    {
      name: 'auth-storage', // unique name
      partialize: (state) => ({ token: state.token }), // Only persist the token
    }
  )
)


### 2.3. Anti-patterns and Code Smells to Avoid

- **Over-reliance on Global State:** Avoid storing everything in a single global store.  Instead, break down your application's state into smaller, more manageable stores based on feature or domain.
- **Mutating State Directly:** Never mutate the state directly. Always use the `set` function provided by Zustand to ensure proper state updates and re-renders.  Consider using Immer middleware (`zustand/middleware`) for immutable updates with mutable syntax.
- **Complex Selectors without Memoization:**  Avoid complex selectors that perform expensive computations without memoization.  This can lead to performance issues, especially with frequent state updates.
- **Creating Stores Inside Components:**  Do not define Zustand stores inside React components.  This will cause the store to be re-created on every render, leading to data loss and unexpected behavior.

### 2.4. State Management Best Practices

- **Single Source of Truth:**  Treat the Zustand store as the single source of truth for your application's state.
- **Minimize State:** Store only the essential data in the store.  Derive computed values using selectors.
- **Clear State Transitions:**  Ensure that state transitions are predictable and well-defined.  Avoid complex and convoluted update logic.

### 2.5. Error Handling Patterns

- **Try/Catch Blocks in Actions:**  Wrap asynchronous actions in `try/catch` blocks to handle potential errors.  Update the state to reflect the error status.
- **Error Boundary Components:**  Use React error boundary components to catch errors that occur during rendering.
- **Centralized Error Logging:** Implement a centralized error logging mechanism to track errors that occur in your application.  Send error reports to a monitoring service.

## 3. Performance Considerations

### 3.1. Optimization Techniques

- **Selective Updates:**  Use selectors to subscribe to only the specific parts of the state that your component needs. This prevents unnecessary re-renders when other parts of the state change.
- **Shallow Equality Checks:** By default, Zustand uses strict equality (`===`) for state comparisons. If you're using complex objects, use `shallow` from `zustand/shallow` as an equality function in `useStore` to avoid unnecessary re-renders.
- **Memoization:** Memoize selectors using libraries like `reselect` or `lodash.memoize` to prevent unnecessary re-computations.
- **Batch Updates:** Use `unstable_batchedUpdates` from `react-dom` to batch multiple state updates into a single render cycle.

### 3.2. Memory Management

- **Avoid Leaks:** Ensure that you are not creating memory leaks by properly cleaning up any subscriptions or event listeners that you create.
- **Limit Store Size:** Keep your stores as small as possible by only storing the data that you need.

### 3.3. Rendering Optimization

- **Virtualization:** If you are rendering large lists of data, use virtualization techniques (e.g., `react-window`, `react-virtualized`) to only render the visible items.
- **Code Splitting:**  Split your application into smaller bundles to reduce the initial load time.

### 3.4. Bundle Size Optimization

- **Tree Shaking:** Ensure that your build process is configured to perform tree shaking, which removes unused code from your bundles.
- **Minimize Dependencies:**  Use only the dependencies that you need.  Avoid importing entire libraries when you only need a few functions.
- **Compression:**  Compress your bundles using tools like Gzip or Brotli.

### 3.5. Lazy Loading Strategies

- **Component-Level Lazy Loading:**  Lazy load components using `React.lazy` and `Suspense`.
- **Route-Based Lazy Loading:**  Lazy load routes using `react-router-dom`'s `lazy` function.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

- **Cross-Site Scripting (XSS):** Sanitize user inputs to prevent XSS attacks.
- **Cross-Site Request Forgery (CSRF):** Protect your API endpoints from CSRF attacks by implementing CSRF tokens.
- **Sensitive Data Exposure:** Avoid storing sensitive data in the store unless absolutely necessary. Encrypt sensitive data if it must be stored.

### 4.2. Input Validation

- **Server-Side Validation:** Always validate user inputs on the server-side.
- **Client-Side Validation:** Perform client-side validation to provide immediate feedback to the user.

### 4.3. Authentication and Authorization Patterns

- **JSON Web Tokens (JWT):** Use JWTs for authentication and authorization.
- **Role-Based Access Control (RBAC):** Implement RBAC to control access to different parts of your application based on user roles.

### 4.4. Data Protection Strategies

- **Encryption:** Encrypt sensitive data at rest and in transit.
- **Data Masking:** Mask sensitive data in logs and other outputs.

### 4.5. Secure API Communication

- **HTTPS:** Use HTTPS to encrypt communication between the client and server.
- **API Rate Limiting:** Implement API rate limiting to prevent abuse.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

- **Test Store Logic:** Write unit tests to verify the logic of your Zustand stores.
- **Mock Dependencies:** Mock any external dependencies (e.g., API calls) to isolate the store logic.
- **Test State Transitions:** Ensure that state transitions are correct by asserting the expected state after each action.

### 5.2. Integration Testing

- **Test Component Interactions:** Write integration tests to verify that your components interact correctly with the Zustand store.
- **Render Components:** Render your components and simulate user interactions to test the integration between the UI and the store.

### 5.3. End-to-End Testing

- **Simulate User Flows:** Write end-to-end tests to simulate complete user flows through your application.
- **Test Real-World Scenarios:** Test real-world scenarios to ensure that your application is working correctly in a production-like environment.

### 5.4. Test Organization

- **Colocate Tests:** Colocate your tests with the code that they are testing.
- **Use Descriptive Names:** Use descriptive names for your test files and test cases.

### 5.5. Mocking and Stubbing

- **Jest Mocks:** Use Jest's mocking capabilities to mock external dependencies.
- **Sinon Stubs:** Use Sinon to create stubs for functions and methods.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

- **Incorrectly Using `set`:** Forgetting to use the functional form of `set` when updating state based on the previous state.
- **Not Handling Asynchronous Errors:** Failing to handle errors in asynchronous actions.
- **Over-relying on `shallow`:**  Using `shallow` equality when a deep comparison is actually required.

### 6.2. Edge Cases to Be Aware Of

- **Race Conditions:** Be aware of potential race conditions when handling asynchronous actions.
- **Context Loss:** Ensure that the Zustand provider is properly configured to avoid context loss issues.

### 6.3. Version-Specific Issues

- **Check Release Notes:**  Always check the release notes for new versions of Zustand to be aware of any breaking changes or new features.

### 6.4. Compatibility Concerns

- **React Version:**  Ensure that your version of React is compatible with the version of Zustand that you are using.

### 6.5. Debugging Strategies

- **Zustand Devtools:** Use the Zustand Devtools extension to inspect the store's state and track state changes.
- **Console Logging:**  Use console logging to debug your store logic and component interactions.
- **Breakpoints:** Set breakpoints in your code to step through the execution and inspect the state.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

- **VS Code:** Use VS Code as your code editor.
- **ESLint:**  Use ESLint to enforce code style and prevent errors.
- **Prettier:** Use Prettier to automatically format your code.
- **Zustand Devtools:**  Use the Zustand Devtools extension to inspect the store's state.

### 7.2. Build Configuration

- **Webpack:** Use Webpack to bundle your code.
- **Parcel:** Use Parcel for zero-configuration bundling.
- **Rollup:** Use Rollup for building libraries.

### 7.3. Linting and Formatting

- **ESLint:** Configure ESLint to enforce code style and best practices.
- **Prettier:** Configure Prettier to automatically format your code.
- **Husky:** Use Husky to run linters and formatters before committing code.

### 7.4. Deployment Best Practices

- **Environment Variables:** Use environment variables to configure your application for different environments.
- **CDN:** Use a CDN to serve your static assets.
- **Caching:** Implement caching to improve performance.

### 7.5. CI/CD Integration

- **GitHub Actions:** Use GitHub Actions to automate your build, test, and deployment processes.
- **CircleCI:** Use CircleCI to automate your build, test, and deployment processes.
- **Jenkins:** Use Jenkins to automate your build, test, and deployment processes.

## 8. Zustand-X (zustandx.udecode.dev)

Zustand-X builds on top of Zustand to reduce boilerplate and enhance features, providing a store factory with derived selectors and actions.

### 8.1. Key Features

- **Less Boilerplate:** Simplified store creation.
- **Modular State Management:** Derived selectors and actions.
- **Middleware Support:** Integrates with `immer`, `devtools`, and `persist`.
- **TypeScript Support:** Full TypeScript support.
- **React-Tracked Support:** Integration with `react-tracked`.

### 8.2. Usage

javascript
import { createStore } from 'zustand-x';

const repoStore = createStore('repo')({
  name: 'zustandX',
  stars: 0,
  owner: {
    name: 'someone',
    email: 'someone@xxx.com',
  },
});

// Hook store
repoStore.useStore;

// Vanilla store
repoStore.store;

// Selectors
repoStore.use.name();
repoStore.get.name();

// Actions
repoStore.set.name('new name');

// Extend selectors
repoStore.extendSelectors((state, get, api) => ({
  validName: () => get.name().trim(),
  title: (prefix) => `${prefix + get.validName()} with ${get.stars()} stars`,
}));

// Extend actions
repoStore.extendActions((set, get, api) => ({
  validName: (name) => {
    set.name(name.trim());
  },
  reset: (name) => {
    set.validName(name);
    set.stars(0);
  },
}));


### 8.3. Global Store

Combine multiple stores into a global store for easier access.

javascript
import { mapValuesKey } from 'zustand-x';

export const rootStore = {
  repo: repoStore,
  // other stores
};

export const useStore = () => mapValuesKey('use', rootStore);
export const useTrackedStore = () => mapValuesKey('useTracked', rootStore);
export const store = mapValuesKey('get', rootStore);
export const actions = mapValuesKey('set', rootStore);

// Usage
useStore().repo.name();
actions.repo.stars(store.repo.stars + 1);


## 9. zustand-ards

A library of simple, opinionated utilities for Zustand to improve the developer experience.

### 9.1 Installation

bash
pnpm i zustand-ards
# or
npm i zustand-ards


### 9.2 Basic Usage

javascript
import { withZustandards } from 'zustand-ards';

const useWithZustandards = withZustandards(useStore);
const { bears, increaseBears } = useWithZustandards(['bears', 'increaseBears']);


### 9.3 Store Hook Enhancements

- **`withZustandards`:** Combines `withArraySelector` and `withDefaultShallow`.
- **`withArraySelector`:** Adds an array selector to the store hook, eliminating the need for multiple hooks or complex selector functions.
- **`withDefaultShallow`:** Makes the store hook shallow by default, equivalent to passing `shallow` from `zustand/shallow` to the original hook.

javascript
import { withArraySelector } from 'zustand-ards';

const useStoreWithArray = withArraySelector(useExampleStore);
const { bears, increaseBears } = useStoreWithArray(['bears', 'increaseBears']);

import { withDefaultShallow } from 'zustand-ards';

const useShallowStore = withDefaultShallow(useExampleStore);
const { wizards } = useShallowStore((state) => ({ wizards: state.wizards }));


## 10. TypeScript Guide

### 10.1. Basic Usage

Annotate the store's state type using `create<T>()(...)`.

typescript
import { create } from 'zustand';

interface BearState {
  bears: number;
  increase: (by: number) => void;
}

const useBearStore = create<BearState>()((set) => ({
  bears: 0,
  increase: (by) => set((state) => ({ bears: state.bears + by }))
}));


### 10.2. Using `combine`

`combine` infers the state, so no need to type it.

typescript
import { create } from 'zustand';
import { combine } from 'zustand/middleware';

const useBearStore = create(
  combine({ bears: 0 }, (set) => ({
    increase: (by: number) => set((state) => ({ bears: state.bears + by }))
  }))
);


### 10.3. Using Middlewares

Use middlewares immediately inside `create` to ensure contextual inference works correctly. Devtools should be used last to prevent other middlewares from mutating the `setState` before it.

typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface BearState {
  bears: number;
  increase: (by: number) => void;
}

const useBearStore = create<BearState>()(
  devtools(
    persist(
      (set) => ({
        bears: 0,
        increase: (by) => set((state) => ({ bears: state.bears + by }))
      }),
      { name: 'bearStore' }
    )
  )
);


### 10.4. Authoring Middlewares and Advanced Usage

Zustand middlewares can mutate the store.  Higher-kinded mutators are available for complex type problems.

### 10.5. Middleware Examples

**Middleware that doesn't change the store type:**

typescript
import { create, State, StateCreator, StoreMutatorIdentifier } from 'zustand';

type Logger = <
  T extends State,
  Mps extends [StoreMutatorIdentifier, unknown][] = [],
  Mcs extends [StoreMutatorIdentifier, unknown][] = [],
>(f: StateCreator<T, Mps, Mcs>, name?: string) => StateCreator<T, Mps, Mcs>;

const loggerImpl = (f, name) => (set, get, store) => {
  const loggedSet = (...a) => {
    set(...a);
    console.log(...(name ? [`${name}:`] : []), get());
  };
  const setState = store.setState;
  store.setState = (...a) => {
    setState(...a);
    console.log(...(name ? [`${name}:`] : []), store.getState());
  };
  return f(loggedSet, get, store);
};

export const logger = loggerImpl;


**Middleware that changes the store type:** Requires advanced TypeScript features.

### 10.6. Slices Pattern

The slices pattern is a way to split a store into smaller, more manageable parts.

typescript
import { create, StateCreator } from 'zustand';

interface BearSlice {
  bears: number;
  addBear: () => void;
  eatFish: () => void;
}

interface FishSlice {
  fishes: number;
  addFish: () => void;
}

const createBearSlice: StateCreator<BearSlice> = (set) => ({
  bears: 0,
  addBear: () => set((state) => ({ bears: state.bears + 1 })), 
  eatFish: () => set((state) => ({fishes: state.fishes -1}))
});

const createFishSlice: StateCreator<FishSlice> = (set) => ({
  fishes: 0,
  addFish: () => set((state) => ({ fishes: state.fishes + 1 }))
});


export const useBoundStore = create<BearSlice & FishSlice>()((...a) => ({
  ...createBearSlice(...a),
  ...createFishSlice(...a),
}));


### 10.7. Bounded useStore Hook for Vanilla Stores

Provides type safety when using `useStore` with vanilla stores.

## 11. Zustand Tools (zustand-tools)

Tools for simpler Zustand usage with React.

### 11.1. `createSimple(initStore, middlewares)`

Creates a simple store with correct typings and hooks for easier usage.

javascript
import { createSimple } from 'zustand-tools';

const demoStore = createSimple({ foo: 'bar' });

/*
 * will provide:
 * demoStore.useStore.getState().foo
 * demoStore.useStore.getState().setFoo(value)
 * demoStore.hooks.useFoo() => [value, setter] // like useState
 */

const useFoo = demoStore.hooks.useFoo;

function App() {
  const [foo, setFoo] = useFoo();

  useEffect(() => {
    setFoo('newBar');
  }, [setFoo]);

  return <div>{foo}</div>;
}


### 11.2. `createSimpleContext(initStore, middlewares)`

Basically the same as `createSimple` but returns a provider to use the store only for a specific context.

### 11.3. `useAllData()`

Returns all data from the store using a shallow compare.

### 11.4. Adding Additional Actions

Add additional actions to the generated store.

javascript
import { createSimple } from 'zustand-tools';

const { useStore } = createSimple(
  { foo: 1 },
  {
    actions: (set) => ({
      increaseFoo: (amount) => set((state) => ({ foo: state.foo + amount }))
    })
  }
);

useStore.getState().increaseFoo(5);


### 11.5. Adding Middlewares

Middlewares can be added by passing an array as middlewares in the second parameter.

javascript
import { createSimple } from 'zustand-tools';
import { devtools } from 'zustand/middleware';

const demoStore = createSimple({ foo: 'bar' }, { middlewares: [(initializer) => devtools(initializer, { enabled: true })] });


## 12. Practice with no Store Actions

### 12.1. Colocated Actions and State

The recommended usage is to colocate actions and states within the store.

javascript
export const useBoundStore = create((set) => ({
  count: 0,
  text: 'hello',
  inc: () => set((state) => ({ count: state.count + 1 })),
  setText: (text) => set({ text }),
}));


### 12.2. External Actions

An alternative approach is to define actions at the module level, external to the store.

javascript
export const useBoundStore = create(() => ({
  count: 0,
  text: 'hello',
}));

export const inc = () =>
  useBoundStore.setState((state) => ({ count: state.count + 1 }));
export const setText = (text) => useBoundStore.setState({ text });


## Conclusion

By following these best practices, you can build robust and scalable applications using Zustand. Remember to adapt these guidelines to your specific project needs and coding style.