---
description: This guide provides opinionated, actionable best practices for building scalable and performant React applications using MobX, focusing on modern patterns and React Compiler compatibility.
globs: **/*.{js,jsx,ts,tsx}
---
# react-mobx Best Practices

MobX, paired with React, offers a powerful and low-boilerplate approach to state management. By adhering to these guidelines, your team will build predictable, scalable applications ready for React 19 and the React Compiler.

## 1. Store Design and Organization

Organize your application state into domain-specific stores, composed into a single root store.

### 1.1. Class-Based Stores with `makeObservable`

Define each domain area as a class. Use `makeObservable` in the constructor to explicitly declare observables, computed values, and actions. This is the modern, explicit way to define your reactive state.

❌ **BAD: Implicit MobX decorators (legacy)**
```javascript
// src/stores/UserStore.js
import { observable, action, computed } from 'mobx';

class UserStore {
  @observable users = [];
  @observable isLoading = false;

  @computed get activeUsers() {
    return this.users.filter(u => u.isActive);
  }

  @action addUser(user) {
    this.users.push(user);
  }
}
```

✅ **GOOD: Explicit `makeObservable` (modern)**
```typescript
// src/stores/UserStore.ts
import { makeObservable, observable, action, computed } from 'mobx';

interface User {
  id: string;
  name: string;
  isActive: boolean;
}

export class UserStore {
  users: User[] = [];
  isLoading: boolean = false;

  constructor() {
    makeObservable(this, {
      users: observable,
      isLoading: observable,
      activeUsers: computed,
      addUser: action,
      fetchUsers: action,
    });
  }

  get activeUsers(): User[] {
    return this.users.filter(u => u.isActive);
  }

  addUser(user: User) {
    this.users.push(user);
  }

  async fetchUsers() {
    this.isLoading = true;
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    this.users = [{ id: '1', name: 'Alice', isActive: true }];
    this.isLoading = false;
  }
}
```

### 1.2. Root Store Composition

Create a `RootStore` that instantiates and holds references to all other stores. This provides a single entry point for your application's state and facilitates inter-store communication.

```typescript
// src/stores/RootStore.ts
import { UserStore } from './UserStore';
import { AuthStore } from './AuthStore';

export class RootStore {
  userStore: UserStore;
  authStore: AuthStore;

  constructor() {
    this.authStore = new AuthStore(this); // Pass root store for cross-store access
    this.userStore = new UserStore();
  }
}

// Export a singleton instance for convenience, or create one in your app entry.
export const rootStore = new RootStore();
```

### 1.3. Consistent Naming Conventions

Maintain clear naming for stores, observables, and actions.

*   **Stores**: PascalCase (e.g., `UserStore`, `ProductStore`).
*   **Observable Fields**: camelCase (e.g., `users`, `isLoading`).
*   **Computed Values**: camelCase, often prefixed with `get` if a getter (e.g., `get activeUsers`).
*   **Actions**: Verb-prefixed camelCase (e.g., `addUser`, `fetchUsers`).

## 2. Context-Based Store Injection

Avoid prop-drilling by providing your root store via React Context and consuming it with a custom hook.

```typescript
// src/contexts/StoreContext.ts
import React, { createContext, useContext } from 'react';
import { RootStore, rootStore as defaultRootStore } from '../stores/RootStore';

export const StoreContext = createContext<RootStore>(defaultRootStore);

export const StoreProvider: React.FC<{ store?: RootStore }> = ({ children, store }) => {
  return (
    <StoreContext.Provider value={store || defaultRootStore}>
      {children}
    </StoreContext.Provider>
  );
};

export const useStore = (): RootStore => {
  const store = useContext(StoreContext);
  if (!store) {
    throw new Error('useStore must be used within a StoreProvider.');
  }
  return store;
};
```

**Usage in `App.tsx`:**
```typescript
// src/App.tsx
import React from 'react';
import { StoreProvider } from './contexts/StoreContext';
import { rootStore } from './stores/RootStore';
import UserList from './components/UserList';

function App() {
  return (
    <StoreProvider store={rootStore}>
      <div className="App">
        <h1>My MobX App</h1>
        <UserList />
      </div>
    </StoreProvider>
  );
}

export default App;
```

## 3. Observer Integration and Component Granularity

Wrap components that read observable data with `observer` to enable automatic re-rendering. Design small, focused components to maximize MobX's fine-grained reactivity.

### 3.1. Wrap Components with `observer`

Use the `observer` HOC (or `useObserver` if preferred) to make your functional components react to observable changes.

```typescript
// src/components/UserList.tsx
import React, { useEffect } from 'react';
import { observer } from 'mobx-react-lite'; // Use mobx-react-lite for functional components
import { useStore } from '../contexts/StoreContext';

const UserList: React.FC = observer(() => {
  const { userStore } = useStore();

  useEffect(() => {
    userStore.fetchUsers();
  }, [userStore]);

  if (userStore.isLoading) {
    return <div>Loading users...</div>;
  }

  return (
    <div>
      <h2>Users ({userStore.activeUsers.length} active)</h2>
      <ul>
        {userStore.users.map(user => (
          <UserItem key={user.id} user={user} />
        ))}
      </ul>
    </div>
  );
});

export default UserList;
```

### 3.2. Small, Focused `observer` Components

Break down your UI into many small `observer` components. This limits the re-render surface, allowing MobX to optimize updates efficiently.

```typescript
// src/components/UserItem.tsx
import React from 'react';
import { observer } from 'mobx-react-lite';

interface UserItemProps {
  user: { id: string; name: string; isActive: boolean };
}

const UserItem: React.FC<UserItemProps> = observer(({ user }) => {
  console.log(`Rendering UserItem: ${user.name}`); // Observe re-renders
  return (
    <li>
      {user.name} {user.isActive ? '(Active)' : ''}
    </li>
  );
});

export default UserItem;
```

### 3.3. De-reference Observables as Late as Possible

Access observable properties inside the `observer` component's render function (or JSX) rather than in intermediate variables outside of it. This allows MobX to track dependencies more precisely.

❌ **BAD: Early de-referencing**
```typescript
const UserProfile: React.FC = observer(({ user }) => {
  const userName = user.name; // De-referenced too early
  return <div>Name: {userName}</div>;
});
```

✅ **GOOD: Late de-referencing**
```typescript
const UserProfile: React.FC = observer(({ user }) => {
  return <div>Name: {user.name}</div>; // De-referenced inside JSX
});
```

### 3.4. Use Stable Keys for Lists

Always provide a stable, unique `key` prop for elements in lists. Never use array indexes as keys, as this leads to incorrect re-rendering behavior when items are added, removed, or reordered.

❌ **BAD: Array index as key**
```typescript
<ul>
  {userStore.users.map((user, index) => (
    <UserItem key={index} user={user} />
  ))}
</ul>
```

✅ **GOOD: Unique ID as key**
```typescript
<ul>
  {userStore.users.map(user => (
    <UserItem key={user.id} user={user} />
  ))}
</ul>
```

## 4. MobX-State-Tree (MST) Specifics

If using MobX-State-Tree, most MobX principles apply. However, there's one critical anti-pattern to avoid.

### 4.1. Never Spread MST Models as Props

Passing an MST model instance directly is crucial for MobX's tracking. Spreading an MST model (`{...model}`) breaks reactivity and should be strictly avoided.

❌ **BAD: Spreading an MST model**
```typescript
// Assuming 'user' is an MST model instance
<UserProfile {...user} />
```

✅ **GOOD: Passing an MST model instance directly**
```typescript
// Assuming 'user' is an MST model instance
<UserProfile user={user} />
```

## 5. Type Safety with TypeScript

Leverage TypeScript interfaces to define the shape of your stores and models. This catches errors at compile time and improves developer experience.

```typescript
// src/stores/AuthStore.ts
import { makeObservable, observable, action } from 'mobx';
import { RootStore } from './RootStore'; // Import RootStore for type safety

export interface IAuthStore {
  token: string | null;
  isAuthenticated: boolean;
  login(username: string, password: string): Promise<void>;
  logout(): void;
}

export class AuthStore implements IAuthStore {
  token: string | null = null;
  isAuthenticated: boolean = false;
  private rootStore: RootStore; // Reference to root store

  constructor(rootStore: RootStore) {
    makeObservable(this, {
      token: observable,
      isAuthenticated: observable,
      login: action,
      logout: action,
    });
    this.rootStore = rootStore;
  }

  async login(username: string, password: string) {
    // ... API call ...
    this.token = 'some-jwt-token';
    this.isAuthenticated = true;
    // Example of cross-store action
    // this.rootStore.userStore.fetchCurrentUser();
  }

  logout() {
    this.token = null;
    this.isAuthenticated = false;
  }
}
```

## 6. Testing

Unit test your stores independently and use React Testing Library for components, mocking the store context.

### 6.1. Store Unit Tests (Jest)

```typescript
// src/stores/UserStore.test.ts
import { UserStore } from './UserStore';

describe('UserStore', () => {
  let userStore: UserStore;

  beforeEach(() => {
    userStore = new UserStore();
  });

  it('should initialize with empty users and not loading', () => {
    expect(userStore.users).toEqual([]);
    expect(userStore.isLoading).toBe(false);
  });

  it('should add a user', () => {
    const newUser = { id: '2', name: 'Bob', isActive: false };
    userStore.addUser(newUser);
    expect(userStore.users).toEqual([newUser]);
  });

  it('should compute active users correctly', () => {
    userStore.addUser({ id: '1', name: 'Alice', isActive: true });
    userStore.addUser({ id: '2', name: 'Bob', isActive: false });
    expect(userStore.activeUsers).toEqual([{ id: '1', name: 'Alice', isActive: true }]);
  });

  it('should set isLoading during fetch and update users', async () => {
    const promise = userStore.fetchUsers();
    expect(userStore.isLoading).toBe(true);
    await promise;
    expect(userStore.isLoading).toBe(false);
    expect(userStore.users.length).toBeGreaterThan(0);
  });
});
```

### 6.2. Component Tests (React Testing Library)

Mock the `useStore` hook to provide controlled store instances for component tests.

```typescript
// src/components/UserList.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import UserList from './UserList';
import { useStore } from '../contexts/StoreContext';
import { UserStore } from '../stores/UserStore';
import { RootStore } from '../stores/RootStore';

// Mock the useStore hook
jest.mock('../contexts/StoreContext', () => ({
  useStore: jest.fn(),
}));

describe('UserList', () => {
  let mockUserStore: UserStore;
  let mockRootStore: RootStore;

  beforeEach(() => {
    mockUserStore = new UserStore();
    mockRootStore = new RootStore();
    mockRootStore.userStore = mockUserStore; // Inject the mock user store

    (useStore as jest.Mock).mockReturnValue(mockRootStore); // Return the mock root store
  });

  it('renders loading state initially', () => {
    mockUserStore.isLoading = true;
    render(<UserList />);
    expect(screen.getByText('Loading users...')).toBeInTheDocument();
  });

  it('renders users after loading', async () => {
    mockUserStore.isLoading = false;
    mockUserStore.users = [{ id: '1', name: 'Alice', isActive: true }];
    mockUserStore.fetchUsers = jest.fn(() => {
      mockUserStore.users = [{ id: '1', name: 'Alice', isActive: true }];
      return Promise.resolve();
    });

    render(<UserList />);

    await waitFor(() => {
      expect(screen.getByText('Users (1 active)')).toBeInTheDocument();
      expect(screen.getByText('Alice (Active)')).toBeInTheDocument();
    });
  });
});
```