---
description: Definitive guidelines for building robust, performant, and maintainable React Native applications using modern TypeScript, functional components, and a feature-first architecture.
globs: **/*.{js,jsx,ts,tsx}
---
# React Native Best Practices

This guide outlines the mandatory practices for React Native development. Adherence ensures a consistent, high-quality, and scalable codebase.

## 1. Code Organization and Structure

Adopt a feature-based structure with clear separation of concerns.

### 1.1. Feature-First Directory Structure

Organize code by feature, not by type. Each component, screen, or hook should reside in its own directory with co-located styles and tests.

**✅ GOOD:**

```
src/
├── features/
│   ├── Auth/
│   │   ├── components/
│   │   │   ├── LoginForm/
│   │   │   │   ├── LoginForm.tsx
│   │   │   │   ├── LoginForm.styles.ts
│   │   │   │   └── LoginForm.test.tsx
│   │   ├── screens/
│   │   │   ├── LoginScreen/
│   │   │   │   ├── LoginScreen.tsx
│   │   │   │   ├── LoginScreen.styles.ts
│   │   │   │   └── LoginScreen.test.tsx
│   │   └── hooks/
│   │       └── useAuth.ts
│   ├── Profile/
│   └── ...
├── components/ # Shared, generic UI components
│   ├── Button/
│   │   ├── Button.tsx
│   │   ├── Button.styles.ts
│   │   └── Button.test.tsx
│   └── ...
├── navigation/ # App-level navigation config
├── services/   # Global API clients, utility services
├── utils/      # Generic helper functions
├── types/      # Global TypeScript types
└── App.tsx
```

### 1.2. File Naming and Co-location

Use PascalCase for components/screens, camelCase for hooks/utilities. Co-locate related files.

**❌ BAD:**

```
// components/Button.js
// components/buttonStyles.js
// components/buttonTests.js
```

**✅ GOOD:**

```
// components/Button/Button.tsx
// components/Button/Button.styles.ts
// components/Button/Button.test.tsx
```

### 1.3. Absolute Imports

Configure and use absolute imports to avoid deeply nested relative paths.

**❌ BAD:**

```typescript
import { Button } from '../../../../components/Button/Button';
```

**✅ GOOD:**

```typescript
import { Button } from '@components/Button/Button';
```

**`tsconfig.json` (or `jsconfig.json`):**

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@components/*": ["src/components/*"],
      "@features/*": ["src/features/*"],
      "@navigation/*": ["src/navigation/*"],
      "@services/*": ["src/services/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

### 1.4. One Component Per File

Each `.tsx` file must export exactly one React component as its default export. Pure stateless components can be co-located if they are tightly coupled and small.

## 2. Component Architecture

Prioritize functional components, hooks, and strong typing.

### 2.1. Functional Components with Hooks

Always use functional components and React Hooks. Class components are deprecated.

**❌ BAD:**

```typescript
import React, { Component } from 'react';
class MyComponent extends Component { /* ... */ }
```

**✅ GOOD:**

```typescript
import React, { useState, useEffect } from 'react';
const MyComponent: React.FC = () => { /* ... */ };
```

### 2.2. Custom Hooks for Reusable Logic

Extract stateful logic into custom hooks for reusability and testability.

**❌ BAD:**

```typescript
const MyComponent: React.FC = () => {
  const [count, setCount] = useState(0);
  useEffect(() => { /* complex logic */ }, [count]);
  // ... duplicated in other components
};
```

**✅ GOOD:**

```typescript
// hooks/useCounter.ts
import { useState, useEffect } from 'react';
export const useCounter = (initialValue: number = 0) => {
  const [count, setCount] = useState(initialValue);
  useEffect(() => { /* complex logic */ }, [count]);
  return { count, setCount };
};

// MyComponent.tsx
import { useCounter } from '@hooks/useCounter';
const MyComponent: React.FC = () => {
  const { count, setCount } = useCounter(10);
  // ...
};
```

### 2.3. Presentational vs. Container Components

Separate UI rendering (presentational) from business logic and data fetching (container/screen).

*   **Presentational Components:** Receive data via props, render UI, emit events. No direct state management beyond UI concerns.
*   **Container/Screen Components:** Manage state, fetch data, handle business logic, pass data/callbacks to presentational components.

## 3. State Management

Choose the right tool for the job.

### 3.1. Local Component State

Use `useState` and `useReducer` for state confined to a single component or a small, local subtree.

### 3.2. Global Application State

For complex, global state, use **Redux Toolkit**. For simpler global state needs, **Zustand** or **Jotai** are acceptable.

**❌ BAD (for complex global state):**

```typescript
// Over-reliance on React Context for app-wide state
// leading to re-renders and complex prop drilling.
```

**✅ GOOD (Redux Toolkit):**

```typescript
// store/authSlice.ts
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
interface AuthState { user: string | null; token: string | null; }
const authSlice = createSlice({
  name: 'auth',
  initialState: { user: null, token: null } as AuthState,
  reducers: {
    setCredentials: (state, action: PayloadAction<AuthState>) => {
      state.user = action.payload.user;
      state.token = action.payload.token;
    },
  },
});
export const { setCredentials } = authSlice.actions;
export default authSlice.reducer;
```

## 4. Styling

Use `StyleSheet.create` for performance and maintainability.

### 4.1. Co-located `StyleSheet.create`

Define styles in a separate `.styles.ts` file using `StyleSheet.create`.

**❌ BAD:**

```typescript
// MyComponent.tsx
const MyComponent: React.FC = () => {
  return <View style={{ backgroundColor: 'red', padding: 10 }} />;
};
```

**✅ GOOD:**

```typescript
// MyComponent.styles.ts
import { StyleSheet } from 'react-native';
export const styles = StyleSheet.create({
  container: {
    backgroundColor: 'red',
    padding: 10,
  },
});

// MyComponent.tsx
import { styles } from './MyComponent.styles';
const MyComponent: React.FC = () => {
  return <View style={styles.container} />;
};
```

### 4.2. Theming

Centralize colors, fonts, and spacing in a theme file for consistent UI.

## 5. Navigation

Use **React Navigation** for all navigation needs.

### 5.1. Dedicated Navigation Files

Define navigators and screens in a `navigation/` directory.

```typescript
// navigation/AppNavigator.tsx
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { LoginScreen } from '@features/Auth/screens/LoginScreen/LoginScreen';
import { HomeScreen } from '@features/Home/screens/HomeScreen/HomeScreen';

const Stack = createNativeStackNavigator();

export const AppNavigator: React.FC = () => (
  <Stack.Navigator>
    <Stack.Screen name="Login" component={LoginScreen} />
    <Stack.Screen name="Home" component={HomeScreen} />
  </Stack.Navigator>
);
```

## 6. API Calls

Centralize API logic in a service layer.

### 6.1. Service Layer with `axios`

Use `axios` for API requests and encapsulate logic in dedicated service files.

```typescript
// services/authService.ts
import axios from 'axios';
const API_URL = 'https://api.example.com/auth';

export const authService = {
  login: async (credentials: LoginCredentials) => {
    const response = await axios.post(`${API_URL}/login`, credentials);
    return response.data;
  },
  // ... other auth methods
};
```

## 7. Performance Considerations

Optimize for smooth user experience and efficient resource usage.

### 7.1. Memoization

Use `React.memo`, `useCallback`, and `useMemo` to prevent unnecessary re-renders.

**❌ BAD:**

```typescript
const MyList: React.FC<{ items: Item[] }> = ({ items }) => {
  const renderItem = ({ item }: { item: Item }) => <ListItem item={item} />;
  return <FlatList data={items} renderItem={renderItem} />;
};
```

**✅ GOOD:**

```typescript
const MyList: React.FC<{ items: Item[] }> = React.memo(({ items }) => {
  const renderItem = useCallback(({ item }: { item: Item }) => <ListItem item={item} />, []);
  return <FlatList data={items} renderItem={renderItem} />;
});
```

### 7.2. Virtualized Lists

Always use `FlatList` or `SectionList` for displaying large lists of data.

### 7.3. Image Optimization

Use optimized image formats (WebP), proper sizing, and caching libraries.

## 8. Accessibility

Build accessible UIs from the start.

### 8.1. Semantic Props

Use `accessibilityLabel`, `accessibilityRole`, and `accessibilityState` for all interactive elements.

```typescript
<TouchableOpacity
  onPress={handlePress}
  accessibilityLabel="Tap to submit form"
  accessibilityRole="button"
>
  <Text>Submit</Text>
</TouchableOpacity>
```

## 9. Testing Approaches

Implement a robust testing strategy.

### 9.1. Unit and Integration Testing

Use **Jest** and **React Native Testing Library** for unit and integration tests. Co-locate tests with components (`.test.tsx`).

```typescript
// Button.test.tsx
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Button } from './Button';

describe('Button', () => {
  it('renders correctly and calls onPress', () => {
    const mockOnPress = jest.fn();
    const { getByText } = render(<Button title="Press Me" onPress={mockOnPress} />);
    fireEvent.press(getByText('Press Me'));
    expect(mockOnPress).toHaveBeenCalledTimes(1);
  });
});
```

### 9.2. End-to-End Testing

Use **Detox** for end-to-end tests to simulate user interactions across the entire application.

## 10. Common Pitfalls and Gotchas

Avoid common mistakes that lead to bugs and performance issues.

### 10.1. Directly Mutating State

Never directly modify state variables. Always use the state setter function.

**❌ BAD:**

```typescript
const [user, setUser] = useState({ name: 'John' });
user.name = 'Jane'; // Direct mutation!
```

**✅ GOOD:**

```typescript
const [user, setUser] = useState({ name: 'John' });
setUser(prevUser => ({ ...prevUser, name: 'Jane' }));
```

### 10.2. Conditional Rendering with `&&`

Avoid `&&` for conditional rendering in React Native, as it can render `0` or `false` as text, leading to `Invariant Violation` errors if not inside a `<Text>` component.

**❌ BAD:**

```typescript
const MyComponent: React.FC<{ showMessage: boolean }> = ({ showMessage }) => {
  return (
    <View>
      {showMessage && <Text>Hello!</Text>}
      {/* If showMessage is false, it renders '0' */}
    </View>
  );
};
```

**✅ GOOD:**

```typescript
const MyComponent: React.FC<{ showMessage: boolean }> = ({ showMessage }) => {
  return (
    <View>
      {showMessage ? <Text>Hello!</Text> : null}
    </View>
  );
};
```

### 10.3. Ignoring Platform Differences

Always test on both iOS and Android. Use `Platform.select` for platform-specific code.

```typescript
import { Platform, StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    paddingTop: Platform.select({
      ios: 20,
      android: 0,
    }),
  },
});
```