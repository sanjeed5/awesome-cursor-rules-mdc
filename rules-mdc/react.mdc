---
description: Comprehensive guide to React best practices, covering code organization, performance, security, testing, and common pitfalls. Adhering to these guidelines helps developers build maintainable, scalable, and high-performing React applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
# React Best Practices: A Comprehensive Guide

This document outlines the best practices for developing React applications, covering various aspects from code organization to security and testing. Following these guidelines leads to more maintainable, scalable, and performant applications.

## 1. Code Organization and Structure

### 1.1 Directory Structure

A well-defined directory structure is crucial for maintainability. Here's a recommended structure:


src/
  ├── components/
  │   ├── Button/
  │   │   ├── Button.jsx
  │   │   ├── Button.module.css
  │   │   └── Button.test.jsx
  │   ├── Input/
  │   │   ├── Input.jsx
  │   │   ├── Input.module.css
  │   │   └── Input.test.jsx
  │   └── ...
  ├── contexts/
  │   ├── AuthContext.jsx
  │   └── ThemeContext.jsx
  ├── hooks/
  │   ├── useAuth.js
  │   └── useTheme.js
  ├── pages/
  │   ├── Home.jsx
  │   ├── About.jsx
  │   └── ...
  ├── services/
  │   ├── api.js
  │   └── auth.js
  ├── utils/
  │   ├── helpers.js
  │   └── validators.js
  ├── App.jsx
  ├── index.jsx
  └── ...


-   **`components/`**: Reusable UI components.
    -   Each component has its own directory containing the component file, associated styles (using CSS modules), and tests.
-   **`contexts/`**: React context providers.
-   **`hooks/`**: Custom React hooks.
-   **`pages/`**: Top-level components representing different routes or views.
-   **`services/`**: API interaction logic.
-   **`utils/`**: Utility functions.

### 1.2 File Naming Conventions

-   **Components**: Use PascalCase (e.g., `MyComponent.jsx`).
-   **Hooks**: Use camelCase prefixed with `use` (e.g., `useMyHook.js`).
-   **Contexts**: Use PascalCase suffixed with `Context` (e.g., `MyContext.jsx`).
-   **Services/Utils**: Use camelCase (e.g., `apiService.js`, `stringUtils.js`).
-   **CSS Modules**: Use `.module.css` or `.module.scss` (e.g., `Button.module.css`).

### 1.3 Module Organization

-   **Co-location**: Keep related files (component, styles, tests) together in the same directory.
-   **Single Responsibility**: Each module should have a clear and specific purpose.
-   **Avoid Circular Dependencies**: Ensure modules don't depend on each other in a circular manner.

### 1.4 Component Architecture

-   **Atomic Design**: Consider using Atomic Design principles (Atoms, Molecules, Organisms, Templates, Pages) to structure components.
-   **Composition over Inheritance**: Favor component composition to reuse code and functionality.
-   **Presentational and Container Components**: Separate UI rendering (presentational) from state management and logic (container).

### 1.5 Code Splitting Strategies

-   **Route-Based Splitting**: Use `React.lazy` and `Suspense` to load components only when a specific route is accessed.  This is very common and improves initial load time.
-   **Component-Based Splitting**: Split large components into smaller chunks that can be loaded on demand.
-   **Bundle Analyzer**: Use a tool like `webpack-bundle-analyzer` to identify large dependencies and optimize bundle size.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

-   **Higher-Order Components (HOCs)**: Reusable logic that wraps components (use with caution; prefer hooks).
-   **Render Props**: Sharing code using a prop whose value is a function.
-   **Compound Components**: Components that work together implicitly (e.g., `Tabs`, `Tab`).
-   **Hooks**: Reusable stateful logic that can be shared across functional components.

### 2.2 Recommended Approaches

-   **Form Handling**: Use controlled components with local state or a form library like Formik or React Hook Form.
-   **API Calls**: Use `useEffect` hook to make API calls and manage loading states.
-   **Conditional Rendering**: Use short-circuit evaluation (`&&`) or ternary operators for simple conditions; use separate components for complex scenarios.
-   **List Rendering**: Always provide a unique and stable `key` prop when rendering lists.

### 2.3 Anti-patterns and Code Smells

-   **Direct DOM Manipulation**: Avoid directly manipulating the DOM; let React handle updates.
-   **Mutating State Directly**: Always use `setState` or the state updater function to modify state.
-   **Inline Styles**: Use CSS modules or styled-components for maintainable styles.
-   **Over-Engineering**: Avoid using complex solutions for simple problems.
-   **Prop Drilling**: Passing props through multiple levels of components without them being used.

### 2.4 State Management Best Practices

-   **Local State**: Use `useState` for component-specific state.
-   **Context API**: Use `useContext` for global state accessible to many components, but avoid for very frequently updated data.
-   **Redux/Mobx**: Use these libraries for complex state management in large applications.
-   **Recoil/Zustand**: Lightweight alternatives to Redux, often easier to set up and use.
-   **Immutable Data**: Treat state as immutable to prevent unexpected side effects.

### 2.5 Error Handling Patterns

-   **Error Boundaries**: Wrap components with error boundaries to catch errors during rendering and prevent crashes.
-   **Try-Catch Blocks**: Use try-catch blocks for handling errors in asynchronous operations and event handlers.
-   **Centralized Error Logging**: Implement a centralized error logging service to track errors and improve application stability.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Memoization**: Use `React.memo`, `useMemo`, and `useCallback` to prevent unnecessary re-renders and recalculations.
-   **Virtualization**: Use libraries like `react-window` or `react-virtualized` to efficiently render large lists or tables.
-   **Debouncing/Throttling**: Limit the rate at which functions are executed (e.g., in input fields).
-   **Code Splitting**: Load code on demand using `React.lazy` and `Suspense`.

### 3.2 Memory Management

-   **Avoid Memory Leaks**: Clean up event listeners, timers, and subscriptions in `useEffect`'s cleanup function.
-   **Release Unused Objects**: Avoid holding onto large objects in memory when they are no longer needed.
-   **Garbage Collection**: Understand how JavaScript's garbage collection works and avoid creating unnecessary objects.

### 3.3 Rendering Optimization

-   **Minimize State Updates**: Avoid unnecessary state updates that trigger re-renders.
-   **Batch Updates**: Batch multiple state updates into a single update using `ReactDOM.unstable_batchedUpdates`.
-   **Keys**: Ensure that keys are unique and consistent across renders.

### 3.4 Bundle Size Optimization

-   **Tree Shaking**: Remove unused code during the build process.
-   **Minification**: Reduce the size of JavaScript and CSS files.
-   **Image Optimization**: Compress and optimize images to reduce file size.
-   **Dependency Analysis**: Use tools like `webpack-bundle-analyzer` to identify large dependencies.

### 3.5 Lazy Loading Strategies

-   **Route-Based Lazy Loading**: Load components when a user navigates to a specific route.
-   **Component-Based Lazy Loading**: Load components when they are about to be rendered.
-   **Intersection Observer**: Load components when they become visible in the viewport.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

-   **Cross-Site Scripting (XSS)**: Sanitize user input to prevent malicious code injection.
-   **Cross-Site Request Forgery (CSRF)**: Use anti-CSRF tokens to protect against unauthorized requests.
-   **Denial of Service (DoS)**: Implement rate limiting and request validation to prevent abuse.
-   **Injection Attacks**: Avoid directly embedding user input into database queries or system commands.

### 4.2 Input Validation

-   **Client-Side Validation**: Validate user input in the browser to provide immediate feedback.
-   **Server-Side Validation**: Always validate user input on the server to prevent malicious data.
-   **Sanitize Input**: Sanitize user input to remove potentially harmful characters or code.

### 4.3 Authentication and Authorization

-   **Secure Authentication**: Use secure authentication mechanisms like OAuth 2.0 or JWT.
-   **Role-Based Access Control (RBAC)**: Implement RBAC to control access to resources based on user roles.
-   **Multi-Factor Authentication (MFA)**: Enable MFA to add an extra layer of security.

### 4.4 Data Protection Strategies

-   **Encryption**: Encrypt sensitive data at rest and in transit.
-   **Data Masking**: Mask sensitive data in logs and UI displays.
-   **Regular Backups**: Create regular backups of application data.

### 4.5 Secure API Communication

-   **HTTPS**: Use HTTPS to encrypt communication between the client and the server.
-   **API Keys**: Protect API keys and secrets.
-   **CORS**: Configure Cross-Origin Resource Sharing (CORS) to prevent unauthorized access to APIs.

## 5. Testing Approaches

### 5.1 Unit Testing

-   **Test Components**: Test individual components in isolation.
-   **Testing Library**: Use React Testing Library for UI testing, focusing on user behavior.
-   **Jest**: Use Jest as the test runner.

### 5.2 Integration Testing

-   **Test Component Interactions**: Test how components interact with each other.
-   **Mock API Calls**: Mock API calls to test component behavior in different scenarios.
-   **React Testing Library**: Effective for testing integration points in components.

### 5.3 End-to-End (E2E) Testing

-   **Test Full Application Flows**: Test complete user flows, such as login, registration, and checkout.
-   **Cypress/Playwright**: Use tools like Cypress or Playwright for E2E testing.
-   **Automated Browser Tests**: Automate browser tests to ensure application stability.

### 5.4 Test Organization

-   **Co-locate Tests**: Keep test files close to the components they test (e.g., `Button.test.jsx` in the `Button` directory).
-   **Descriptive Names**: Use descriptive names for test files and test cases.
-   **Test Suites**: Organize tests into logical suites.

### 5.5 Mocking and Stubbing

-   **Mock Modules**: Mock external modules or API calls to isolate components during testing.
-   **Stub Functions**: Stub function implementations to control component behavior.
-   **Jest Mocks**: Utilize Jest's mocking capabilities for effective unit testing.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

-   **Ignoring Keys in Lists**: Forgetting to provide unique and stable `key` props when rendering lists.
-   **Incorrect State Updates**: Mutating state directly instead of using `setState` or the state updater function.
-   **Missing Dependencies in `useEffect`**: Not including all dependencies in the dependency array of the `useEffect` hook.
-   **Over-Using State**: Storing derived data in state instead of calculating it on demand.

### 6.2 Edge Cases

-   **Asynchronous State Updates**: Handling state updates in asynchronous operations.
-   **Race Conditions**: Preventing race conditions when making multiple API calls.
-   **Handling Errors in Event Handlers**: Properly handling errors in event handlers to prevent crashes.

### 6.3 Version-Specific Issues

-   **React 16 vs. React 17/18**: Understanding differences in lifecycle methods, error handling, and concurrent mode.
-   **Deprecated Features**: Being aware of deprecated features and using recommended alternatives.

### 6.4 Compatibility Concerns

-   **Browser Compatibility**: Ensuring compatibility with different browsers and devices.
-   **Library Compatibility**: Ensuring compatibility between React and other libraries.

### 6.5 Debugging Strategies

-   **React DevTools**: Use React DevTools to inspect component hierarchies, props, and state.
-   **Console Logging**: Use console logging to debug code and track variables.
-   **Breakpoints**: Set breakpoints in the code to step through execution and inspect variables.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

-   **VS Code**: A popular code editor with excellent React support.
-   **Create React App**: A tool for quickly setting up a new React project.
-   **React DevTools**: A browser extension for inspecting React components.
-   **ESLint**: A linter for enforcing code style and preventing errors.
-   **Prettier**: A code formatter for automatically formatting code.

### 7.2 Build Configuration

-   **Webpack/Vite**: Configure Webpack or Vite to bundle and optimize code.
-   **Babel**: Configure Babel to transpile JavaScript code to older versions.
-   **Environment Variables**: Use environment variables to configure different environments.

### 7.3 Linting and Formatting

-   **ESLint**: Configure ESLint with recommended React rules.
-   **Prettier**: Configure Prettier to automatically format code.
-   **Husky/lint-staged**: Use Husky and lint-staged to run linters and formatters before committing code.

### 7.4 Deployment Best Practices

-   **Static Hosting**: Host static assets on a CDN.
-   **Server-Side Rendering (SSR)**: Use SSR to improve SEO and initial load time.
-   **Continuous Deployment**: Automate the deployment process using CI/CD.

### 7.5 CI/CD Integration

-   **GitHub Actions/GitLab CI**: Use GitHub Actions or GitLab CI to automate testing, linting, and deployment.
-   **Automated Testing**: Run automated tests on every commit or pull request.
-   **Automated Deployment**: Automatically deploy code to production after successful tests.

By following these best practices, React developers can build high-quality, maintainable, and scalable applications that meet the demands of modern web development. Continual education and adaptation to emerging trends in the React ecosystem are crucial for sustained success.