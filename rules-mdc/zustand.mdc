---
description: Definitive guidelines for using Zustand in React projects, focusing on type safety, modularity, performance, and maintainability with practical code examples.
globs: **/*.{js,jsx}
---
# zustand Best Practices

Zustand is our go-to for global state management due to its minimal API and built-in performance optimizations. This guide outlines the definitive patterns for using Zustand effectively in our projects, ensuring type safety, modularity, and optimal performance.

## 1. Typed Store Shape (TypeScript First)

Always define explicit TypeScript interfaces for your store's state and actions. This provides invaluable type safety and auto-completion across the application, preventing common runtime errors.

❌ BAD: Untyped store, prone to runtime errors
```typescript
import { create } from 'zustand';

const useStore = create((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}));
```

✅ GOOD: Fully typed store for robust development
```typescript
import { create } from 'zustand';

interface CounterState {
  count: number;
  increment: () => void;
  decrement: () => void;
}

const useCounterStore = create<CounterState>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
}));
```

## 2. Slice-Based Organization

For scalable applications, organize your store into logical "slices" (e.g., `authSlice`, `uiSlice`). This keeps concerns separated, improves readability, and makes testing easier. Compose these slices into a single root store.

`src/store/types.ts`:
```typescript
import { StateCreator } from 'zustand';

// Define individual slice interfaces
export interface AuthSlice {
  user: { id: string; name: string } | null;
  token: string | null;
  login: (user: { id: string; name: string }, token: string) => void;
  logout: () => void;
}

export interface UISlice {
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
}

// Combine all slice interfaces into the global AppState
export type AppState = AuthSlice & UISlice;

// Type for creating slices that will be composed into AppState
export type AppStateCreator<T> = StateCreator<AppState, [], [], T>;
```

`src/store/slices/createAuthSlice.ts`:
```typescript
import { AppStateCreator, AuthSlice } from '../types';

export const createAuthSlice: AppStateCreator<AuthSlice> = (set) => ({
  user: null,
  token: null,
  login: (user, token) => set({ user, token }),
  logout: () => set({ user: null, token: null }),
});
```

`src/store/slices/createUISlice.ts`:
```typescript
import { AppStateCreator, UISlice } from '../types';

export const createUISlice: AppStateCreator<UISlice> = (set) => ({
  isLoading: false,
  setLoading: (loading) => set({ isLoading: loading }),
});
```

`src/store/useAppStore.ts`:
```typescript
import { create } from 'zustand';
import { AppState } from './types';
import { createAuthSlice } from './slices/createAuthSlice';
import { createUISlice } from './slices/createUISlice';

export const useAppStore = create<AppState>()((...a) => ({
  ...createAuthSlice(...a),
  ...createUISlice(...a),
}));
```

## 3. Naming Conventions

Follow consistent naming for clarity and discoverability.

*   **Store Hook**: `use[Feature]Store` or `useAppStore` for the root.
*   **Actions**: Verb-oriented (e.g., `increment`, `setUser`, `fetchData`).
*   **Store Files**: Located under `src/store/` or `src/features/[feature]/store.ts`.
*   **Exports**: Only export the custom hook, never the raw `create` object.

❌ BAD: Inconsistent naming, exposing raw store
```typescript
// store.js
export const myStore = create(...); // Exposes raw store
export const useMyStore = myStore; // Bad naming, use `useMyStore` as the primary hook

// component.jsx
myStore.setState({ ... }); // Direct mutation outside hook, bypasses React lifecycle
```

✅ GOOD: Clear, consistent, and encapsulated
```typescript
// src/store/useAppStore.ts
export const useAppStore = create<AppState>(...);

// src/features/auth/useAuthStore.ts (if using feature-specific stores)
export const useAuthStore = create<AuthSlice>(...);

// component.tsx
import { useAppStore } from 'src/store/useAppStore';
const { user, login } = useAppStore();
```

## 4. Functional Updates to Prevent Stale Closures

Always use functional updates (`set(state => ...)`) when an action's new state depends on the current state. This prevents issues with stale closures in asynchronous operations or rapid updates.

❌ BAD: Potential stale closure, especially in async operations
```typescript
const useCounterStore = create<CounterState>((set, get) => ({
  count: 0,
  // ...
  incrementAsync: async () => {
    await someAsyncOperation();
    const currentCount = get().count; // 'currentCount' might be outdated if another update occurred
    set({ count: currentCount + 1 });
  },
}));
```

✅ GOOD: Robust functional update, `state` is always the latest
```typescript
const useCounterStore = create<CounterState>((set) => ({
  count: 0,
  // ...
  incrementAsync: async () => {
    await someAsyncOperation();
    set((state) => ({ count: state.count + 1 }));
  },
}));
```

## 5. Selectors and Shallow Comparison for Performance

Consume only the necessary parts of the state using selectors. For objects or arrays, use `shallow` (or a custom equality function) to prevent unnecessary re-renders when only nested properties change.

❌ BAD: Re-renders component on any state change in the store
```typescript
const MyComponent = () => {
  const { user, token } = useAppStore(); // Re-renders if any part of AppState changes
  // ...
};
```

✅ GOOD: Optimized re-renders with selectors and `shallow`
```typescript
import { shallow } from 'zustand/shallow';
import { useAppStore } from 'src/store/useAppStore';

const UserProfile = () => {
  // Only re-renders if user.name or user.email changes
  const { name, email } = useAppStore(
    (state) => ({ name: state.user?.name, email: state.user?.email }),
    shallow
  );
  // ...
};

const AuthStatus = () => {
  // Only re-renders if token changes
  const token = useAppStore((state) => state.token);
  // ...
};
```

## 6. Essential Middleware Usage

Leverage Zustand's middleware for common concerns like persistence, devtools integration, and immutable updates.

*   **`persist`**: For local storage or IndexedDB. Always provide a `name` and consider `version` for migrations.
*   **`devtools`**: Integrate with Redux DevTools. Enable only in development.
*   **`immer`**: For simplified immutable updates, especially with deeply nested objects.

```typescript
import { create } from 'zustand';
import { persist, devtools, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface SettingsState {
  theme: 'light' | 'dark';
  notifications: { enabled: boolean; sound: boolean };
  setTheme: (theme: 'light' | 'dark') => void;
  toggleNotifications: () => void;
  toggleNotificationSound: () => void;
}

const useSettingsStore = create<SettingsState>()(
  devtools( // Enable devtools for debugging
    persist( // Persist state to storage
      immer((set) => ({ // Use immer for easier immutable updates
        theme: 'light',
        notifications: { enabled: true, sound: true },
        setTheme: (theme) => set({ theme }),
        toggleNotifications: () =>
          set((state) => {
            state.notifications.enabled = !state.notifications.enabled; // Direct mutation with immer
          }),
        toggleNotificationSound: () =>
          set((state) => {
            state.notifications.sound = !state.notifications.sound; // Direct mutation with immer
          }),
      })),
      {
        name: 'app-settings', // Unique name for storage key
        storage: createJSONStorage(() => localStorage), // Choose storage type
        version: 1, // Crucial for future migrations
        partialize: (state) => ({ theme: state.theme }), // Only persist 'theme'
      }
    ),
    { name: 'Settings Store', enabled: process.env.NODE_ENV === 'development' } // Devtools options
  )
);
```

## 7. Initializing Stores Outside Components

Declare your `create` calls at the module level (top of the file) to ensure a single, consistent store instance across your application. Never call `create` inside a React component.

❌ BAD: Creates new store instance on every render
```typescript
const MyComponent = () => {
  // This creates a NEW store instance every time MyComponent re-renders!
  const useLocalStore = create(() => ({ value: 0 }));
  const value = useLocalStore((state) => state.value);
  return <div>{value}</div>;
};
```

✅ GOOD: Single store instance, declared once
```typescript
// At the top of your store file (e.g., src/store/useCounterStore.ts)
const useCounterStore = create<CounterState>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}));

const MyComponent = () => {
  const count = useCounterStore((state) => state.count);
  return <div>{count}</div>;
};
```

## 8. Asynchronous Actions

Handle asynchronous operations directly within your store actions. Zustand doesn't require special middleware for async, keeping the API simple and direct.

```typescript
interface UserState {
  user: { id: string; name: string } | null;
  loading: boolean;
  error: string | null;
  fetchUser: (userId: string) => Promise<void>;
}

const useUserStore = create<UserState>((set) => ({
  user: null,
  loading: false,
  error: null,
  fetchUser: async (userId) => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`/api/users/${userId}`);
      if (!response.ok) throw new Error('Failed