---
description: This guide provides definitive, opinionated best practices for building robust, performant, and maintainable cross-platform mobile applications with Expo, leveraging modern React Native patterns and the latest SDK features.
globs: **/*
---
# expo Best Practices

This document outlines the definitive standards for Expo development. Adhere to these guidelines to ensure consistency, performance, and maintainability across our projects.

## 1. Code Organization and Structure

Maintain a clean, scalable project structure.

*   **Root Structure:**
    *   `assets/`: Static assets (images, fonts).
    *   `src/`: All application logic.
        *   `src/components/`: Reusable UI components.
        *   `src/screens/`: Top-level components representing distinct app views.
        *   `src/navigation/`: Navigation configuration (if not using Expo Router's `app/`).
        *   `src/hooks/`: Custom React Hooks.
        *   `src/utils/`: Utility functions, constants.
        *   `src/services/`: API clients, data fetching logic.
*   **Naming Conventions:**
    *   **Files/Folders:** `kebab-case` for directories, `PascalCase` for components, `camelCase` for hooks/utils.
    *   **Components:** `PascalCase` (e.g., `Button.tsx`, `HomeScreen.tsx`).
    *   **Variables/Functions:** `camelCase` (e.g., `userName`, `fetchData`).
    *   **Constants:** `UPPER_SNAKE_CASE` for global constants.

## 2. Component Best Practices

Prioritize functional components, TypeScript, and clear styling.

*   **Functional Components with Hooks:** Always use functional components.
    ❌ BAD:
    ```typescript
    class MyComponent extends React.Component { /* ... */ }
    ```
    ✅ GOOD:
    ```typescript
    const MyComponent: React.FC = () => { /* ... */ };
    ```
*   **TypeScript for Type Safety:** All new code must be in TypeScript.
    ❌ BAD:
    ```javascript
    function greet(name) { return `Hello, ${name}`; }
    ```
    ✅ GOOD:
    ```typescript
    const greet = (name: string): string => `Hello, ${name}`;
    ```
*   **Styling with `StyleSheet`:** Centralize styles for readability and performance.
    ❌ BAD:
    ```typescript
    <Text style={{ fontSize: 16, color: 'blue' }}>Hello</Text>
    ```
    ✅ GOOD:
    ```typescript
    import { StyleSheet, Text } from 'react-native';
    const MyComponent = () => <Text style={styles.text}>Hello</Text>;
    const styles = StyleSheet.create({ text: { fontSize: 16, color: 'blue' } });
    ```

## 3. Navigation with Expo Router

Leverage file-based routing for intuitive navigation.

*   **File-System Based Routing:** Use `app/` directory for routing.
    ❌ BAD: Manual stack/tab navigator setup in a single file for all routes.
    ✅ GOOD:
    ```
    // app/_layout.tsx
    import { Stack } from 'expo-router';
    export default function RootLayout() {
      return <Stack />;
    }

    // app/index.tsx (maps to /)
    export default function HomePage() { /* ... */ }

    // app/profile/[id].tsx (maps to /profile/:id)
    export default function ProfilePage() {
      const { id } = useLocalSearchParams();
      // ...
    }
    ```

## 4. Data & State Management

Manage state efficiently and immutably.

*   **Immutable State Updates:** Always create new objects/arrays when updating state.
    ❌ BAD:
    ```typescript
    const [user, setUser] = useState({ name: 'Alice' });
    user.name = 'Bob'; // Direct mutation
    setUser(user);
    ```
    ✅ GOOD:
    ```typescript
    const [user, setUser] = useState({ name: 'Alice' });
    setUser(prev => ({ ...prev, name: 'Bob' }));
    ```
*   **`useEffect` Dependency Arrays:** Carefully manage dependencies to prevent unnecessary re-renders or infinite loops.
    ❌ BAD:
    ```typescript
    useEffect(() => { fetchData(); }); // Runs on every render
    ```
    ✅ GOOD:
    ```typescript
    useEffect(() => { fetchData(); }, []); // Runs once on mount
    useEffect(() => { saveUser(user); }, [user]); // Runs when user changes
    ```

## 5. Environment Variables

Securely manage environment-specific values.

*   **`EXPO_PUBLIC_` Prefix:** Use `.env` files with `EXPO_PUBLIC_` for client-side variables.
    ❌ BAD: `API_KEY=mysecret` in `.env` and `process.env.API_KEY`. (Exposed in bundle, not automatically injected)
    ✅ GOOD:
    ```
    // .env
    EXPO_PUBLIC_API_URL=https://api.example.com/staging
    ```
    ```typescript
    // In your code
    const apiUrl = process.env.EXPO_PUBLIC_API_URL;
    ```
    > **Warning:** Never store sensitive keys (e.g., private API keys) in `EXPO_PUBLIC_` variables. They are bundled client-side. Use EAS Secrets for server-side secrets.

## 6. Performance Considerations

Optimize for a smooth user experience.

*   **`React.memo` for Pure Components:** Wrap pure functional components to prevent re-renders when props are unchanged.
    ❌ BAD:
    ```typescript
    const MyItem = ({ data }) => { /* ... */ }; // Renders even if data is same object reference
    ```
    ✅ GOOD:
    ```typescript
    const MyItem = React.memo(({ data }) => { /* ... */ });
    ```
*   **`useCallback` and `useMemo`:** Memoize functions and values passed to `React.memo` components or expensive computations.
    ❌ BAD:
    ```typescript
    const handlePress = () => { /* ... */ }; // New function on every render
    <MyItem onPress={handlePress} />
    ```
    ✅ GOOD:
    ```typescript
    const handlePress = useCallback(() => { /* ... */ }, []);
    <MyItem onPress={handlePress} />
    ```
*   **Lazy Loading Screens:** For large apps, lazy load screens with `React.lazy` and `Suspense`.
    ```typescript
    const LazyScreen = React.lazy(() => import('./LazyScreen'));
    // In your navigation or component:
    <Suspense fallback={<LoadingSpinner />}>
      <LazyScreen />
    </Suspense>
    ```

## 7. Error Handling

Implement robust error handling mechanisms.

*   **`try/catch` for Async Operations:** Handle potential errors in asynchronous code.
    ❌ BAD:
    ```typescript
    const fetchData = async () => { await api.get('/data'); };
    ```
    ✅ GOOD:
    ```typescript
    const fetchData = async () => {
      try {
        await api.get('/data');
      } catch (error) {
        console.error('Failed to fetch data:', error);
        // Display user-friendly error message
      }
    };
    ```
*   **Global Error Boundary:** Catch UI errors in React components.
    ```typescript
    // src/components/ErrorBoundary.tsx
    class ErrorBoundary extends React.Component {
      state = { hasError: false };
      static getDerivedStateFromError() { return { hasError: true }; }
      render() {
        if (this.state.hasError) { return <Text>Something went wrong.</Text>; }
        return this.props.children;
      }
    }
    // In App.tsx or _layout.tsx
    <ErrorBoundary><App /></ErrorBoundary>
    ```

## 8. Testing Approaches

Ensure code quality with comprehensive testing.

*   **Unit Tests with Jest & React Testing Library:** Focus on component logic and user interactions. Aim for ≥80% coverage.
    ❌ BAD: No tests, or only shallow rendering tests.
    ✅ GOOD:
    ```typescript
    // src/components/Button.test.tsx
    import { render, fireEvent } from '@testing-library/react-native';
    import Button from './Button';

    test('renders correctly and calls onPress', () => {
      const mockOnPress = jest.fn();
      const { getByText } = render(<Button title="Click Me" onPress={mockOnPress} />);
      fireEvent.press(getByText('Click Me'));
      expect(mockOnPress).toHaveBeenCalledTimes(1);
    });
    ```
*   **Snapshot Testing for UI Components:** Capture UI structure to detect unintended changes.
    ```typescript
    // src/components/Button.test.tsx (continued)
    import renderer from 'react-test-renderer';

    test('renders correctly (snapshot)', () => {
      const tree = renderer.create(<Button title="Test" onPress={() => {}} />).toJSON();
      expect(tree).toMatchSnapshot();
    });
    ```