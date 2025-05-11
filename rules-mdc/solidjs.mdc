---
description: This comprehensive guide outlines best practices for SolidJS development, covering code organization, performance, security, testing, and common pitfalls. It provides actionable guidelines for building maintainable, efficient, and secure SolidJS applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
# SolidJS Best Practices: A Comprehensive Guide

This document provides a detailed overview of best practices for SolidJS development, covering various aspects from code organization to security and performance. It aims to guide developers in building robust, scalable, and maintainable SolidJS applications.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

A well-defined directory structure is crucial for maintaining a large SolidJS application. Here's a recommended structure:


my-solid-app/
├── src/
│   ├── components/
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.module.css
│   │   │   ├── Button.test.tsx
│   │   ├── Card/
│   │   │   ├── Card.tsx
│   │   │   ├── Card.module.css
│   │   ├── index.ts # Exports all components
│   ├── contexts/
│   │   ├── ThemeContext.tsx
│   ├── hooks/
│   │   ├── useTheme.ts
│   ├── utils/
│   │   ├── api.ts
│   │   ├── helpers.ts
│   ├── routes/
│   │   ├── Home.tsx
│   │   ├── About.tsx
│   ├── styles/
│   │   ├── global.css
│   ├── App.tsx
│   ├── index.tsx
│   ├── types.d.ts # Global type definitions
│   ├── vite-env.d.ts # vite type definitions
├── public/
│   ├── favicon.svg
│   ├── assets/
├── .env
├── vite.config.ts
├── tsconfig.json
├── package.json
├── README.md


**Explanation:**

*   **src/:** Contains the source code of your application.
    *   **components/:** Reusable UI components. Each component should have its own directory containing the component file, styles, and tests.
    *   **contexts/:** SolidJS Context providers for managing global state and shared data.
    *   **hooks/:** Custom hooks for reusable logic.
    *   **utils/:** Utility functions such as API calls, data formatting, etc.
    *   **routes/:** Components that define different routes in your application.
    *   **styles/:** Global styles or shared CSS modules.
    *   **App.tsx:** The root component of your application.
    *   **index.tsx:** Entry point for rendering the App component into the DOM.
    *   **types.d.ts:** Global typescript type definitions for the project
    *   **vite-env.d.ts:** vite environment type definitions
*   **public/:** Static assets such as images, fonts, etc.
*   **.env:** Stores environment variables.
*   **vite.config.ts:** Vite configuration file.
*   **tsconfig.json:** TypeScript configuration file.
*   **package.json:** Node.js package file.
*   **README.md:** Project documentation.

### 1.2 File Naming Conventions

*   **Components:** Use PascalCase for component file names (e.g., `Button.tsx`, `UserProfile.jsx`).
*   **Hooks:** Use camelCase prefixed with `use` for hook file names (e.g., `useTheme.ts`, `useFetchData.js`).
*   **Utilities:** Use camelCase for utility file names (e.g., `api.ts`, `formatDate.js`).
*   **Styles:** Use the `.module.css` extension for CSS modules (e.g., `Button.module.css`). If using other styling solutions like styled-components, follow their recommended practices.

### 1.3 Module Organization

*   **Keep components modular:** Each component should have a single responsibility.
*   **Export components:** Export components to make them reusable in other parts of the application (e.g., `export default Button;`).
*   **Create an `index.ts` file:** In each directory (e.g., `components/`) export all the modules in that directory. This allows simpler imports from other components. e.g., `import {Button, Card} from '../components'`

### 1.4 Component Architecture

*   **Functional Components:** Use functional components with Solid's reactivity system for managing state and UI updates. Avoid class components.
*   **Props:** Use props to pass data from parent to child components. Be explicit about prop types using TypeScript.
*   **Context API:** Use Solid's Context API for sharing data between components without prop drilling. Use cases: themes, user authentication, global configuration.
*   **Signals:** Employ SolidJS Signals for managing component-level state. Prefer derived signals or memoization for expensive computations.

### 1.5 Code Splitting Strategies

*   **Route-Based Splitting:** Use dynamic imports to load components only when a route is visited. Leverage Solid Router's lazy loading capabilities with `lazy`.
*   **Component-Based Splitting:** Split large components into smaller chunks that can be loaded on demand. Use `Suspense` to handle loading states.
*   **Library Splitting:** If using large libraries, consider splitting them into smaller chunks or using tree-shaking to remove unused code.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to SolidJS

*   **Renderless Components:** Create components that encapsulate logic without rendering any UI elements. They can be used as hooks or utility functions to share logic between components.
*   **Compound Components:** Build components that work together implicitly, often using Context to share state. (e.g., Tabs, Accordions).
*   **Reactivity-Based Composition:** Compose components based on shared reactive state, rather than inheritance. This allows more flexibility and better performance.
*   **Custom Control Flow Components:** Create reusable components to handle conditional rendering or list iteration with specific optimization strategies. Examples include more performant alternatives to `<For/>` in certain specific use cases.

### 2.2 Recommended Approaches for Common Tasks

*   **Fetching Data:** Use `createResource` for handling asynchronous data fetching. Use `AbortController` for cancelling requests on component unmount or data changes.
*   **Managing Forms:** Use signals to track form input values. Implement validation logic and handle form submission.
*   **Handling Events:** Attach event handlers to elements using JSX attributes (e.g., `onClick`). Use arrow functions or bind methods to ensure correct `this` context.
*   **Global State Management:** Use Solid's Context API or a dedicated state management library (e.g., `solid-zustand`, `solid-js/store`) for sharing data between components.

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Reading and Writing to the Same Signal in an Effect:** This creates an infinite loop. Use the functional update form `setCount(count => count + 1)` or derived signals.
*   **Writing to Signals in Effects for Derived Values:** Prefer derived signals or memoization for expensive computations (using `createMemo`).
*   **Destructuring Props:** This breaks reactivity. Access props directly or use `mergeProps` and `splitProps`.
*   **Using `.map` and Ternary Operators for Control Flow:** Use Solid's control flow components (`<For>`, `<Index>`, `<Show>`) to avoid unnecessary DOM elements.
*   **Mutating Resource Data Directly:** Always create new objects when mutating resource data to prevent unnecessary re-renders. use `createMutable` judiciously only where mutability is needed.
*   **Ignoring Cleanup Functions:** Neglecting `onCleanup` can lead to memory leaks and unexpected behavior when components unmount.

### 2.4 State Management Best Practices

*   **Component-Level State:** Use signals for managing simple, component-specific state.
*   **Context API:** For sharing data between components without prop drilling, especially themes or user authentication.
*   **Global Stores:** Use store solutions or third-party libraries (e.g., `solid-zustand`, `solid-js/store`) for complex global state management.
*   **Immutability:** Treat state as immutable.  Always create new objects/arrays when updating state, especially with resources or stores.
*   **Single Source of Truth:** Ensure that each piece of state has a single, authoritative source. Avoid duplicating state across components.

### 2.5 Error Handling Patterns

*   **`Suspense` for Loading States:** Wrap asynchronous operations like `createResource` with `Suspense` to display loading indicators.
*   **Error Boundaries:** Implement error boundaries to catch errors during rendering and display fallback UI.
*   **Try-Catch Blocks:** Use `try-catch` blocks for handling synchronous errors or errors in event handlers.
*   **Centralized Error Logging:** Implement a centralized error logging mechanism to track and monitor errors in your application.
*   **User-Friendly Error Messages:** Display user-friendly error messages to guide users on how to resolve issues.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Fine-Grained Reactivity:** Solid's fine-grained reactivity ensures that only the necessary parts of the UI are updated when state changes. Leverage this by breaking down components into smaller, more focused units.
*   **Derived Signals and Memos:** Use derived signals and memos to avoid recomputing expensive values unnecessarily.
*   **Control Flow Components:** Use `<For>`, `<Index>`, and `<Show>` for efficient conditional rendering and list iteration.
*   **Lazy Loading:** Load components and modules only when they are needed.
*   **Virtualization:** For rendering large lists, use virtualization techniques to only render the visible items.
*   **Debouncing and Throttling:** Implement debouncing or throttling for event handlers that trigger frequent updates.

### 3.2 Memory Management

*   **Cleanup Functions:** Always use `onCleanup` to release resources when a component is unmounted, such as timers, event listeners, and subscriptions.
*   **Avoid Memory Leaks:** Be mindful of potential memory leaks when using closures or retaining references to DOM elements.
*   **Weak References:** Use `WeakRef` or `WeakMap` for holding references to objects without preventing them from being garbage collected.

### 3.3 Rendering Optimization

*   **Minimize DOM Updates:** Solid's reactivity system minimizes DOM updates, but avoid unnecessary renders by optimizing component structure and state management.
*   **Batch Updates:** Use `batchedUpdates` to batch multiple state updates into a single render cycle (use with caution).
*   **Avoid Inline Styles:** Prefer CSS classes or CSS modules for styling components.
*   **Use CSS Transitions and Animations:** Leverage CSS transitions and animations for smoother UI updates.

### 3.4 Bundle Size Optimization

*   **Tree Shaking:** Use tree shaking to remove unused code from your bundle. Vite automatically performs tree shaking.
*   **Code Splitting:** Split your application into smaller chunks that can be loaded on demand.
*   **Minification and Compression:** Minify and compress your code to reduce bundle size. Vite handles minification by default.
*   **Image Optimization:** Optimize images by compressing them and using appropriate formats (e.g., WebP).
*   **Dependency Analysis:** Analyze your dependencies to identify and remove unnecessary libraries.

### 3.5 Lazy Loading Strategies

*   **Route-Based Lazy Loading:** Load components only when a route is visited using dynamic imports and the `lazy` function from Solid Router.
*   **Component-Based Lazy Loading:** Split large components into smaller chunks that can be loaded on demand.
*   **Conditional Lazy Loading:** Load components based on specific conditions, such as user interactions or device capabilities.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user inputs and escaping data when rendering it in the DOM. Avoid using `dangerouslySetInnerHTML`.
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection by using anti-CSRF tokens in forms and validating them on the server.
*   **SQL Injection:** Prevent SQL injection attacks by using parameterized queries or ORMs when interacting with databases.
*   **Authentication and Authorization:** Secure your application by implementing proper authentication and authorization mechanisms.
*   **Denial-of-Service (DoS):** Protect against DoS attacks by implementing rate limiting, input validation, and resource quotas.

### 4.2 Input Validation

*   **Server-Side Validation:** Always validate user inputs on the server to prevent malicious data from being stored in your database.
*   **Client-Side Validation:** Provide client-side validation for a better user experience, but never rely on it as the sole means of protection.
*   **Sanitize Inputs:** Sanitize user inputs to remove potentially harmful characters or code.
*   **Use Validation Libraries:** Use validation libraries to simplify the input validation process.

### 4.3 Authentication and Authorization Patterns

*   **JSON Web Tokens (JWT):** Use JWTs for managing user authentication and authorization.
*   **OAuth 2.0:** Implement OAuth 2.0 for third-party authentication and authorization.
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
*   **Multi-Factor Authentication (MFA):** Implement MFA for enhanced security.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit using appropriate encryption algorithms.
*   **Data Masking:** Mask sensitive data when displaying it to users or storing it in logs.
*   **Data Redaction:** Redact sensitive data from logs and audit trails.
*   **Data Retention Policies:** Implement data retention policies to ensure that data is only stored for as long as it is needed.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication to encrypt data in transit.
*   **API Keys:** Use API keys for authenticating clients and controlling access to your APIs.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your APIs.
*   **Input Validation:** Validate all API inputs to prevent malicious data from being processed.
*   **Output Encoding:** Encode API outputs to prevent XSS attacks.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test Individual Components:** Unit tests should focus on testing individual components in isolation.
*   **Mock Dependencies:** Mock dependencies to isolate the component being tested.
*   **Test All Scenarios:** Test all possible scenarios and edge cases.
*   **Use Assertions:** Use assertions to verify that the component behaves as expected.

### 5.2 Integration Testing

*   **Test Component Interactions:** Integration tests should focus on testing the interactions between components.
*   **Mock External Services:** Mock external services to isolate the component being tested.
*   **Test Data Flow:** Test the flow of data between components.

### 5.3 End-to-End Testing

*   **Test the Entire Application:** End-to-end tests should focus on testing the entire application from the user's perspective.
*   **Use a Testing Framework:** Use a testing framework such as Cypress or Playwright to automate the testing process.
*   **Test User Flows:** Test common user flows to ensure that the application works as expected.

### 5.4 Test Organization

*   **Colocate Tests:** Place test files alongside the components they are testing.
*   **Use a Consistent Naming Convention:** Use a consistent naming convention for test files (e.g., `Button.test.tsx`).
*   **Organize Tests by Feature:** Organize tests by feature or module.

### 5.5 Mocking and Stubbing

*   **Use Mocking Libraries:** Use mocking libraries such as Jest or Vitest to create mocks and stubs.
*   **Mock External Dependencies:** Mock external dependencies to isolate the component being tested.
*   **Stub API Calls:** Stub API calls to control the data returned by the API.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Incorrectly Using Effects:** Misunderstanding the dependency tracking of `createEffect` can lead to unnecessary re-runs or infinite loops.
*   **Forgetting `onCleanup`:** Not using `onCleanup` for resources and subscriptions causes memory leaks.
*   **Mutating State Directly:** Directly mutating objects or arrays used as state leads to unexpected behavior and broken reactivity.
*   **Over-Optimizing Prematurely:** Focusing on micro-optimizations before addressing fundamental architectural issues.
*   **Not Understanding Solid's Reactivity:**  Failing to grasp the nuances of Solid's reactive system can lead to inefficient code.

### 6.2 Edge Cases to Be Aware Of

*   **SSR Hydration Issues:** When doing SSR, ensure that the initial state on the server matches the client to avoid hydration errors.
*   **Reactivity with Non-Reactive Values:**  Passing plain JS objects or arrays to components without making them reactive can cause issues with updates.
*   **Transitions and Resource Updates:** Interactions between transitions and resource updates can sometimes be tricky, especially around loading states.
*   **Nested Reactivity:** Managing reactivity in complex, deeply nested data structures can be challenging.

### 6.3 Version-Specific Issues

*   **Compatibility:** Consider backwards compatibility when upgrading SolidJS or related libraries.
*   **Deprecations:** Be aware of deprecated features and APIs and migrate to the recommended alternatives.
*   **Bug Fixes:** Stay up-to-date with bug fixes and security patches in newer versions of SolidJS.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:** Test your application in different browsers to ensure compatibility.
*   **Device Compatibility:** Test your application on different devices, such as desktops, tablets, and mobile phones.
*   **Accessibility:** Ensure that your application is accessible to users with disabilities.

### 6.5 Debugging Strategies

*   **Use Browser Developer Tools:** Use the browser's developer tools to inspect the DOM, network requests, and console output.
*   **SolidJS Devtools:** Use the SolidJS Devtools extension to inspect components, signals, and resources.
*   **Logging:** Add logging statements to your code to track the flow of execution and the values of variables.
*   **Debugging Tools:** Use debugging tools such as VS Code's debugger to step through your code and inspect variables.
*   **Reproducible Examples:** Create reproducible examples to isolate and debug issues more effectively.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Vite:** Use Vite as the build tool. SolidJS uses Vite as its default.
*   **VS Code:** Use VS Code as the code editor, along with extensions for TypeScript, ESLint, and Prettier.
*   **SolidJS Devtools:** Use the SolidJS Devtools browser extension to inspect components, signals, and resources.
*   **TypeScript:** Utilize TypeScript for type safety and improved code maintainability.

### 7.2 Build Configuration

*   **Vite Configuration:** Configure Vite to optimize your build for production (e.g., code splitting, minification, compression).
*   **TypeScript Configuration:** Configure TypeScript to enforce strict type checking and code quality.
*   **Environment Variables:** Use environment variables to store sensitive data and configure your application for different environments.

### 7.3 Linting and Formatting

*   **ESLint:** Use ESLint to enforce code style and identify potential errors.
*   **Prettier:** Use Prettier to automatically format your code.
*   **Husky:** Use Husky to run linters and formatters before committing code.

### 7.4 Deployment Best Practices

*   **Choose a Hosting Platform:** Select a suitable hosting platform for your SolidJS application (e.g., Netlify, Vercel, AWS).
*   **Configure a Production Environment:** Configure your hosting platform to use a production environment with appropriate settings.
*   **Automate Deployment:** Automate the deployment process using CI/CD tools.
*   **Monitor Your Application:** Monitor your application for errors and performance issues.

### 7.5 CI/CD Integration

*   **Choose a CI/CD Tool:** Select a CI/CD tool such as GitHub Actions, GitLab CI, or CircleCI.
*   **Configure CI/CD Pipelines:** Configure CI/CD pipelines to automatically build, test, and deploy your application.
*   **Automate Testing:** Automate unit tests, integration tests, and end-to-end tests in your CI/CD pipelines.
*   **Automate Deployment:** Automate the deployment process to deploy new versions of your application with minimal manual intervention.

This document provides a comprehensive set of best practices for SolidJS development. By following these guidelines, developers can build high-quality, maintainable, and secure SolidJS applications.