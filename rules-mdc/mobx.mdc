---
description: Definitive guidelines for structuring MobX applications with React, focusing on predictable state management, optimal rendering, and modern best practices.
globs: **/*.{js,jsx}
---
# mobx Best Practices

MobX provides a powerful, reactive state management solution for React applications. This guide outlines the definitive best practices for using MobX effectively, ensuring predictable state, optimal performance, and maintainable code.

## 1. Code Organization and Structure

Organize your MobX application for clarity and scalability. A well-defined structure separates concerns, making it easier to navigate and maintain.

### 1.1. Root Store Architecture

Always create a single `RootStore` that composes all domain-specific child stores. This provides a central entry point for your application's state and facilitates communication between stores.

**❌ BAD: Scattered Store Instantiation**
```javascript
// In App.js or various components
const authStore = new AuthStore();
const userStore = new UserStore();
// ... many other stores
```

**✅ GOOD: Centralized Root Store**
```javascript
// src/stores/RootStore.js
import { AuthStore } from './AuthStore';
import { UserStore } from './UserStore';
import { AlertsStore } from './AlertsStore';

export class RootStore {
  authStore: AuthStore;
  userStore: UserStore;
  alertsStore: AlertsStore;

  constructor() {
    this.authStore = new AuthStore(this); // Pass root for inter-store communication
    this.userStore = new UserStore(this);
    this.alertsStore = new AlertsStore(this);
  }
}

// src/stores/UserStore.js
import { makeObservable, observable, action } from 'mobx';
// import type { RootStore } from './RootStore'; // For TypeScript

export class UserStore {
  // rootStore: RootStore; // Uncomment for TypeScript
  @observable name = '';

  constructor(rootStore /*: RootStore */) { // Add type annotation for TypeScript
    makeObservable(this, {
      name: observable,
      setName: action,
    });
    this.rootStore = rootStore;
  }

  setName(name: string) {
    // Example: Accessing another store
    // if (this.rootStore.authStore.isAuthenticated) { // Uncomment for TypeScript
      this.name = name;
    // }
  }
}
```

### 1.2. Clear Folder Layout

Adopt a consistent and logical folder structure.

**✅ GOOD: Recommended Folder Structure**
```
src/
├── components/    // Reusable UI components (often `observer`)
├── containers/    // Top-level components orchestrating logic
├── contexts/      // React Context for stores
├── hooks/         // Custom React hooks
├── services/      // API interaction logic
├── stores/        // MobX stores (observable state, actions, computed)
└── App.jsx
```

## 2. State Definition and Actions

Explicitly define your observable state, computed values, and actions.

### 2.1. Declare Observables and Actions

Always use `makeObservable` (or `makeAutoObservable` for simpler cases) to explicitly mark observables, computed values, and actions. This is crucial for MobX to track changes.

**❌ BAD: Implicit Observables (pre-MobX 6 default)**
```javascript
// This style is deprecated and won't work reliably without decorators + config
class Todo {
  id = Math.random();
  title = "";
  finished = false;
  toggle() { this.finished = !this.finished; }
}
```

**✅ GOOD: Explicit Declaration with `makeObservable`**
```javascript
import { makeObservable, observable, action, computed } from 'mobx';

class TodoStore {
  todos = []; // This is just a plain array initially

  constructor() {
    makeObservable(this, {
      todos: observable, // Make the array observable
      addTodo: action,
      toggleTodo: action,
      completedTodosCount: computed,
    });
  }

  get completedTodosCount() {
    return this.todos.filter(todo => todo.finished).length;
  }

  addTodo(title: string) {
    this.todos.push({ id: Date.now(), title, finished: false });
  }

  toggleTodo(id: number) {
    const todo = this.todos.find(t => t.id === id);
    if (todo) {
      todo.finished = !todo.finished; // MobX tracks changes to observable properties within the array
    }
  }
}
```

## 3. React Integration and Performance

Optimize your React components for MobX's reactive updates.

### 3.1. Provide Stores via Context

Use `React.createContext` and a custom hook to provide your `RootStore` to the component tree, avoiding prop drilling.

**✅ GOOD: Store Context and Hook**
```javascript
// src/contexts/StoreContext.jsx
import React, { createContext, useContext } from 'react';
// import { RootStore } from '../stores/RootStore'; // For TypeScript

// const StoreContext = createContext<RootStore | undefined>(undefined); // For TypeScript
const StoreContext = createContext(undefined);

export const StoreProvider = ({ children, store }) => { // Add type annotations for TypeScript
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
};

export const useStore = () => {
  const store = useContext(StoreContext);
  if (store === undefined) {
    throw new Error('useStore must be used within a StoreProvider');
  }
  return store;
};

// src/App.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { RootStore } from './stores/RootStore';
import { StoreProvider } from './contexts/StoreContext';
import AppContent from './AppContent'; // Your main application component

const rootStore = new RootStore();

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <StoreProvider store={rootStore}>
      <AppContent />
    </StoreProvider>
  </React.StrictMode>
);
```

### 3.2. Wrap Only Leaf Components with `observer`

Apply `observer` only to the React components that *actually read* observable data. This minimizes re-renders to the smallest possible UI fragments.

**❌ BAD: Over-observing Parent Components**
```javascript
// Parent component re-renders even if only a child's specific data changes
import { observer } from 'mobx-react-lite';
import { useStore } from '../contexts/StoreContext';

const UserProfilePage = observer(() => {
  const { userStore, alertsStore } = useStore();
  return (
    <div>
      <h1>Welcome, {userStore.name}</h1>
      <AlertsDisplay /> {/* This component might only need alertsStore */}
      <UserDetails />   {/* This component might only need userStore details */}
    </div>
  );
});
```

**✅ GOOD: Targeted `observer` Usage**
```javascript
// src/components/UserProfilePage.jsx
import { useStore } from '../contexts/StoreContext';

const UserProfilePage = () => { // Not an observer
  const { userStore } = useStore(); // Only reads userStore.name here
  return (
    <div>
      <h1>Welcome, {userStore.name}</h1>
      <AlertsDisplay />
      <UserDetails />
    </div>
  );
};

// src/components/AlertsDisplay.jsx
import { observer } from 'mobx-react-lite';
import { useStore } from '../contexts/StoreContext';

const AlertsDisplay = observer(() => { // Only this component re-renders on alert changes
  const { alertsStore } = useStore();
  return (
    <div>
      <h2>Alerts</h2>
      {alertsStore.alerts.map(alert => <div key={alert.id}>{alert.message}</div>)}
    </div>
  );
});
```

### 3.3. De-Reference Values Late

Access observable values as late as possible in your component tree. This ensures that only the deepest components that depend on a specific value re-render.

**❌ BAD: Early Dereferencing**
```javascript
import { observer } from 'mobx-react-lite';

const UserCard = observer(({ user }) => {
  // `user.name` is dereferenced here, if name changes, UserCard re-renders
  return <DisplayName name={user.name} />;
});
```

**✅ GOOD: Late Dereferencing**
```javascript
import { observer } from 'mobx-react-lite';

const UserCard = observer(({ user }) => {
  // `user` (the observable object) is passed, DisplayName dereferences `name`
  return <DisplayName user={user} />;
});

const DisplayName = observer(({ user }) => {
  // Only DisplayName re-renders if user.name changes
  return <span>{user.name}</span>;
});
```

### 3.4. Render Lists in Dedicated Components

For large collections, render them in dedicated, `observer`-wrapped list components. This isolates re-renders to the list itself, not the parent.

**✅ GOOD: Dedicated List Component**
```javascript
// src/components/TodosList.jsx
import { observer } from 'mobx-react-lite';
import { useStore } from '../contexts/StoreContext';
import { TodoItem } from './TodoItem';

export const TodosList = observer(() => {
  const { todoStore } = useStore();
  return (
    <ul>
      {todoStore.todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
});

// src/components/TodoItem.jsx
import { observer } from 'mobx-react-lite';

export const TodoItem = observer(({ todo }) => (
  <li>
    <input type="checkbox" checked={todo.finished} onChange={() => todo.toggle()} />
    {todo.title}
  </li>
));
```

## 4. Common Pitfalls and Anti-patterns

Avoid these common mistakes to prevent bugs and performance issues.

### 4.1. Avoid Spreading MobX Models

Never use the spread syntax (`{...model}`) when passing MobX (or MobX-State-Tree) models as props. This can strip away MobX's observability metadata.

**❌ BAD: Spreading a MobX Model**
```javascript
import { observer } from 'mobx-react-lite';

const MyComponent = observer(({ user }) => <UserDisplay {...user} />); // Loses observability
```

**✅ GOOD: Pass the Model Directly**
```javascript
import { observer } from 'mobx-react-lite';

const MyComponent = observer(({ user }) => <UserDisplay user={user} />);

const UserDisplay = observer(({ user }) => (
  <div>Name: {user.name}</div> // user is still observable
));
```

### 4.2. Use Stable IDs for React Keys

Always use stable, unique IDs as `key` props in lists. Never use array indexes if the list can change order, be filtered, or have items added/removed.

**❌ BAD: Array Index as Key**
```javascript
{items.map((item, index) => <ItemView item={item} key={index} />)}
```

**✅ GOOD: Stable Unique ID as Key**
```javascript
{items.map(item => <ItemView item={item} key={item.id} />)}
```

## 5. Side Effects and Data Fetching

Manage asynchronous operations and side effects predictably within your stores.

### 5.1. Centralize API Calls in Stores

Handle all data fetching logic within your MobX stores, not directly in components. Invoke these actions from a single `useEffect` in a top-level component or via a MobX reaction to prevent duplicate requests.

**❌ BAD: Duplicate API Calls in Multiple Components**
```javascript
// src/components/AlertsVisualization.jsx
import { observer } from 'mobx-react-lite';
import { useEffect } from 'react';
import { useStore } from '../contexts/StoreContext';

const AlertsVisualization = observer(({ id }) => {
  const { alertsStore } = useStore();
  useEffect(() => {
    alertsStore.getAlertsForId(id); // Duplicate call if AlertsList also calls it
  }, [id, alertsStore]);
  return (/* ... */);
});

// src/components/AlertsList.jsx
import { observer } from 'mobx-react-lite';
import { useEffect } from 'react';
import { useStore } from '../contexts/StoreContext';

const AlertsList = observer(({ id }) => {
  const { alertsStore } = useStore();
  useEffect(() => {
    alertsStore.getAlertsForId(id); // Duplicate call
  }, [id, alertsStore]);
  return (/* ... */);
});
```

**✅ GOOD: Centralized Fetching in a Parent or Store Reaction**
```javascript
// src/services/AlertsService.js
export const AlertsService = {
  fetchAlerts: async (id) => {
    // Simulate API call
    return new Promise(resolve => setTimeout(() => resolve([{ id: 1, message: `Alert for ${id}` }]), 500));
  }
};

// src/stores/AlertsStore.js
import { makeObservable, observable, action } from 'mobx';
import { AlertsService } from '../services/AlertsService'; // Dedicated service

export class AlertsStore {
  alerts = [];
  isLoading = false;
  error = null;

  constructor() {
    makeObservable(this, {
      alerts: observable,
      isLoading: observable,
      error: observable,
      getAlertsForId: action,
    });
  }

  async getAlertsForId(id) {
    if (this.isLoading) return; // Prevent duplicate requests
    this.isLoading = true;
    this.error = null;
    try {
      this.alerts = await AlertsService.fetchAlerts(id);
    } catch (e) {
      this.error = e;
    } finally {
      this.isLoading = false;
    }
  }
}

// src/containers/DashboardPage.jsx (Parent component)
import { observer } from 'mobx-react-lite';
import { useEffect } from 'react';
import { useStore } from '../contexts/StoreContext';
import { AlertsVisualization } from '../components/AlertsVisualization';
import { AlertsList } from '../components/AlertsList';

const DashboardPage = observer(({ userId }) => {
  const { alertsStore } = useStore();

  useEffect(() => {
    alertsStore.getAlertsForId(userId); // Single source of truth for fetching
  }, [userId, alertsStore]);