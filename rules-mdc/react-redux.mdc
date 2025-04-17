---
description: Enforces best practices for structuring and maintaining React-Redux applications, focusing on code organization, performance, and maintainability. This rule provides guidelines for developers to write efficient, scalable, and testable React-Redux code.
globs: *.js,*.jsx,*.ts,*.tsx
---
# React-Redux Best Practices

This document outlines best practices for developing React applications with Redux for state management. Following these guidelines will help you write maintainable, scalable, and performant code.

## 1. Code Organization and Structure

### 1.1. Directory Structure

Organize your code around features rather than technical concerns. A feature-based directory structure promotes modularity and makes it easier to understand the application's purpose.


src/
  ├── app/
  │   ├── components/
  │   │   └── App.tsx
  │   └── App.css
  ├── features/
  │   ├── featureName/
  │   │   ├── components/
  │   │   │   ├── FeatureComponent.tsx
  │   │   │   └── FeatureComponent.css
  │   │   ├── FeatureSlice.ts  # Redux slice for the feature
  │   │   ├── FeatureSelectors.ts  # Selectors for accessing feature state
  │   │   ├── FeatureActions.ts  # Actions related to this feature
  │   │   └── api.ts  # API calls related to the feature
  │   ├── anotherFeature/
  │   │   └── ...
  ├── store.ts          # Redux store configuration
  ├── index.tsx         # Entry point for the React application
  └── ...


### 1.2. File Naming Conventions

*   **Components:** Use PascalCase (e.g., `MyComponent.tsx`).
*   **Redux Slice Files:** Use PascalCase and end with `Slice.ts` or `Slice.js` (e.g., `UserSlice.ts`).
*   **Selectors:**  `selectors.ts` or `selectors.js` or `FeatureSelectors.ts` to include a feature name prefix
*   **Actions:** `actions.ts` or `actions.js` or `FeatureActions.ts` to include a feature name prefix
*   **Styles:** Use either `ComponentName.module.css` (for CSS Modules) or `ComponentName.css` (for global styles).
*   **Utility Functions:** Use camelCase (e.g., `formatDate.ts`).

### 1.3. Module Organization

*   **Encapsulation:** Each feature should be a self-contained module with its components, actions, reducers, and selectors.
*   **Single Responsibility Principle:** Each module should have a single, well-defined purpose.
*   **Clear Boundaries:** Define clear boundaries between modules to minimize dependencies and prevent tight coupling.

### 1.4. Component Architecture

*   **Presentational and Container Components:**
    *   **Presentational (Dumb) Components:** Focus on rendering UI and receiving data and callbacks as props. They should be reusable and independent of Redux.
    *   **Container (Smart) Components:** Connect to the Redux store and pass data and actions to presentational components.  Use `connect` or `useSelector` and `useDispatch` hooks.
*   **Small and Reusable Components:** Break down complex UIs into smaller, reusable components to promote code reuse and maintainability.
*   **Component Composition:** Build complex UIs by composing smaller components.

### 1.5. Code Splitting Strategies

*   **Route-Based Splitting:** Split the application into chunks based on routes to reduce the initial load time. Use `React.lazy` and `Suspense` for lazy loading components.
*   **Component-Based Splitting:** Split large components into smaller chunks that can be loaded on demand. Use dynamic imports for loading components asynchronously.
*   **Library Splitting:** Extract commonly used libraries into separate chunks to allow browsers to cache them independently.

## 2. Common Patterns and Anti-Patterns

### 2.1. Design Patterns

*   **Selectors:** Use selectors to abstract the state shape and compute derived data. Selectors improve performance by memoizing results and prevent components from re-rendering unnecessarily.
*   **Custom Hooks:** Create custom hooks for accessing Redux state and dispatching actions. This simplifies component logic and promotes code reuse. For example:

    typescript
    import { useDispatch, useSelector } from 'react-redux';
    import { increment, decrement } from './counterSlice';
    import { RootState } from './store';

    export const useCounter = () => {
      const count = useSelector((state: RootState) => state.counter.value);
      const dispatch = useDispatch();

      const handleIncrement = () => {
        dispatch(increment());
      };

      const handleDecrement = () => {
        dispatch(decrement());
      };

      return { count, handleIncrement, handleDecrement };
    };
    
*   **Redux Toolkit:** Utilize Redux Toolkit to simplify Redux setup and reduce boilerplate code.  Redux Toolkit provides utilities for creating reducers, actions, and the store.

### 2.2. Recommended Approaches

*   **Normalize State:** Structure your state as a normalized data structure, where each piece of data has a unique ID, and relationships between data are represented by IDs. This improves performance and simplifies data management.  Use libraries like Normalizr to help with this.
*   **Immutability:** Treat your state as immutable and use immutable update patterns. This ensures predictable state transitions and prevents unexpected side effects.
*   **Middleware for Side Effects:** Use Redux middleware (e.g., Redux Thunk, Redux Saga) to handle asynchronous operations and side effects. Avoid performing side effects directly in reducers.

### 2.3. Anti-Patterns and Code Smells

*   **Mutating State Directly:** Never mutate the state directly in reducers. Always create a new copy of the state with the desired changes.
*   **Performing Side Effects in Reducers:** Reducers should be pure functions and should not perform any side effects (e.g., API calls, logging).
*   **Storing UI State in Redux:** Avoid storing UI-specific state (e.g., component visibility, form input values) in the Redux store. Store UI state locally in component state.
*   **Over-reliance on Global State:** Avoid putting everything in the Redux store.  Only store data that needs to be accessed by multiple components or across the entire application.  Consider React Context or local component state for component-specific data.

### 2.4. State Management Best Practices

*   **Minimize Global State:** Only store data in the Redux store that needs to be shared across multiple components. For component-specific state, use local component state.
*   **Use Selectors:** Use selectors to access and transform data from the Redux store. This allows you to abstract the state shape and prevent components from re-rendering unnecessarily.
*   **Immutable Updates:** Always update state immutably to ensure predictable state transitions and prevent unexpected side effects. Use libraries like Immer to simplify immutable updates.

### 2.5. Error Handling Patterns

*   **Centralized Error Handling:** Implement a centralized error handling mechanism to catch and handle errors consistently throughout the application. Use error boundary components to catch errors in the UI.
*   **Action Creators for Errors:** Dispatch error actions to the Redux store when errors occur. This allows you to track errors and display error messages in the UI.
*   **Logging Errors:** Log errors to a central logging service for debugging and monitoring purposes.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Memoization:** Use memoization techniques (e.g., `React.memo`, `useMemo`, Reselect) to prevent components from re-rendering unnecessarily.
*   **Batch Updates:** Batch multiple Redux dispatches into a single update to improve performance. Use `redux-batched-updates` or `unstable_batchedUpdates` from React DOM.
*   **Virtualization:** Use virtualization techniques (e.g., `react-window`, `react-virtualized`) to efficiently render large lists and tables.
*   **Code Splitting:** Split the application into smaller chunks to reduce the initial load time.

### 3.2. Memory Management

*   **Avoid Memory Leaks:** Be mindful of memory leaks, especially in event listeners, timers, and subscriptions. Clean up resources properly when components unmount.
*   **Use WeakRefs:** Use WeakRefs to hold references to objects without preventing them from being garbage collected.
*   **Profile Memory Usage:** Use the browser's developer tools to profile memory usage and identify potential memory leaks.

### 3.3. Rendering Optimization

*   **ShouldComponentUpdate / React.memo:** Use `shouldComponentUpdate` (for class components) or `React.memo` (for functional components) to prevent components from re-rendering unnecessarily.
*   **Pure Components:** Use pure components to automatically implement `shouldComponentUpdate` based on prop comparisons.
*   **Immutable Data Structures:** Use immutable data structures to make it easier to detect changes and prevent re-renders.

### 3.4. Bundle Size Optimization

*   **Tree Shaking:** Use tree shaking to remove dead code from your bundles. Configure your bundler (e.g., Webpack, Parcel) to enable tree shaking.
*   **Code Splitting:** Split the application into smaller chunks to reduce the initial load time.
*   **Minification:** Minify your code to reduce the bundle size.
*   **Compression:** Compress your bundles using gzip or Brotli.

### 3.5. Lazy Loading Strategies

*   **Route-Based Lazy Loading:** Load components only when their corresponding routes are visited.
*   **Component-Based Lazy Loading:** Load components on demand when they become visible in the UI.
*   **Image Lazy Loading:** Load images only when they scroll into view.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

*   **Cross-Site Scripting (XSS):** Prevent XSS vulnerabilities by sanitizing user input and escaping output.
*   **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by using anti-CSRF tokens.
*   **Injection Attacks:** Prevent injection attacks (e.g., SQL injection, command injection) by validating user input and using parameterized queries.
*   **Authentication and Authorization:** Implement secure authentication and authorization mechanisms to protect sensitive data and resources.

### 4.2. Input Validation

*   **Server-Side Validation:** Always validate user input on the server-side to prevent malicious data from being stored in the database.
*   **Client-Side Validation:** Use client-side validation to provide immediate feedback to users and reduce the load on the server.
*   **Whitelisting:** Use whitelisting to allow only specific characters or patterns in user input.

### 4.3. Authentication and Authorization Patterns

*   **JSON Web Tokens (JWT):** Use JWTs for authentication and authorization. JWTs are stateless and can be easily verified on the server-side.
*   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization. OAuth 2.0 allows users to grant third-party applications access to their resources without sharing their credentials.
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.

### 4.4. Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Masking:** Mask sensitive data to prevent unauthorized access.
*   **Data Anonymization:** Anonymize data to remove personally identifiable information (PII).

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS to encrypt communication between the client and the server.
*   **API Keys:** Use API keys to authenticate API requests.
*   **Rate Limiting:** Implement rate limiting to prevent abuse and denial-of-service attacks.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   **Test Reducers:** Test reducers to ensure that they handle actions correctly and update the state as expected.
*   **Test Selectors:** Test selectors to ensure that they return the correct data from the state.
*   **Test Actions:** Test action creators to ensure that they return the correct actions.
*   **Test Components:** Test components to ensure that they render correctly and handle user interactions as expected. Use tools like React Testing Library.

### 5.2. Integration Testing

*   **Test Component Interactions:** Test how components interact with each other and with the Redux store.
*   **Test Data Flow:** Test the flow of data through the application, from the UI to the Redux store and back.
*   **Test API Integrations:** Test how the application integrates with external APIs.

### 5.3. End-to-End Testing

*   **Test User Flows:** Test complete user flows to ensure that the application functions correctly from end to end.  Use tools like Cypress or Playwright.
*   **Test Critical Functionality:** Test critical functionality (e.g., login, checkout) to ensure that it is working as expected.
*   **Test Accessibility:** Test the application for accessibility to ensure that it is usable by people with disabilities.

### 5.4. Test Organization

*   **Test Folder Structure:** Organize tests in a way that mirrors the application's directory structure.
*   **Test Suites:** Group tests into logical suites based on functionality or component.
*   **Test Naming Conventions:** Use clear and consistent naming conventions for tests.

### 5.5. Mocking and Stubbing

*   **Mock API Calls:** Mock API calls to isolate components and reducers during testing.
*   **Stub External Dependencies:** Stub external dependencies to control their behavior during testing.
*   **Use Mock Store:** Use a mock Redux store to test components that are connected to the store.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Forgetting to Connect Components:** Forgetting to connect components to the Redux store, resulting in components not receiving state updates.
*   **Incorrectly Mapping State to Props:** Incorrectly mapping state to props, resulting in components not receiving the correct data.
*   **Not Updating State Immutably:** Not updating state immutably, leading to unexpected side effects and re-renders.
*   **Using `setState` with Redux:** Mixing `setState` with Redux can lead to inconsistencies and difficult-to-debug issues. Avoid using `setState` in components connected to Redux, unless for purely local, non-Redux-related state.

### 6.2. Edge Cases

*   **Race Conditions:** Be aware of race conditions when handling asynchronous operations.
*   **Memory Leaks:** Watch out for memory leaks in event listeners, timers, and subscriptions.
*   **Performance Bottlenecks:** Identify and address performance bottlenecks, such as unnecessary re-renders or slow API calls.

### 6.3. Version-Specific Issues

*   **Breaking Changes:** Be aware of breaking changes when upgrading React and Redux.
*   **Deprecated Features:** Avoid using deprecated features and migrate to the recommended alternatives.

### 6.4. Compatibility Concerns

*   **Browser Compatibility:** Test the application in different browsers to ensure compatibility.
*   **Device Compatibility:** Test the application on different devices to ensure compatibility.

### 6.5. Debugging Strategies

*   **Redux DevTools:** Use the Redux DevTools to inspect the Redux store, track actions, and time travel through state changes.
*   **Console Logging:** Use console logging to debug code and track the flow of data.
*   **Breakpoints:** Use breakpoints to pause execution and inspect variables in the debugger.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **Redux DevTools:** The Redux DevTools extension for Chrome and Firefox allows you to inspect the Redux store, track actions, and time travel through state changes.
*   **React Developer Tools:** The React Developer Tools extension for Chrome and Firefox allows you to inspect the React component tree and profile component performance.
*   **ESLint:** ESLint is a linter that helps you identify and fix code style issues and potential errors.
*   **Prettier:** Prettier is a code formatter that automatically formats your code to ensure consistency.

### 7.2. Build Configuration

*   **Webpack:** Webpack is a module bundler that bundles your code and dependencies into optimized bundles for production.
*   **Parcel:** Parcel is a zero-configuration bundler that is easy to use and provides fast build times.
*   **Create React App:** Create React App is a tool that sets up a React development environment with sensible defaults.

### 7.3. Linting and Formatting

*   **ESLint:** Use ESLint to enforce code style rules and prevent potential errors. Configure ESLint to use a popular style guide, such as Airbnb or Standard.
*   **Prettier:** Use Prettier to automatically format your code to ensure consistency. Integrate Prettier with ESLint to automatically fix code style issues.
*   **Husky and Lint-Staged:** Use Husky and Lint-Staged to automatically run linters and formatters on staged files before committing code.

### 7.4. Deployment Best Practices

*   **Environment Variables:** Use environment variables to configure the application for different environments (e.g., development, staging, production).
*   **Continuous Integration:** Use a continuous integration (CI) system to automatically build and test the application whenever code is committed.
*   **Continuous Deployment:** Use a continuous deployment (CD) system to automatically deploy the application to production whenever a new release is created.
*   **Caching:** Configure caching on the server-side and client-side to improve performance.

### 7.5. CI/CD Integration

*   **GitHub Actions:** Use GitHub Actions to automate the build, test, and deployment process.
*   **CircleCI:** Use CircleCI to automate the build, test, and deployment process.
*   **Jenkins:** Use Jenkins to automate the build, test, and deployment process.