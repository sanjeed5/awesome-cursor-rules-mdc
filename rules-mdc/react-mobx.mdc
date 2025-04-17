---
description: This rule provides comprehensive best practices for developing React applications with Mobx, covering code organization, performance, testing, and security considerations. It aims to guide developers in building robust and maintainable applications using React-Mobx.
globs: *.js,*.jsx,*.ts,*.tsx
---
# React-Mobx Best Practices

This document outlines best practices for developing React applications with Mobx. Following these guidelines will help you build robust, maintainable, and performant applications.

## 1. Core Principles

- **Model observable state:**  Define the source of truth in your Mobx stores and make them observable.
- **Use derived state (computed properties):**  Derive data from your observable state using `computed` properties to avoid unnecessary re-renders and ensure data consistency.
- **Embrace mutability:** Mobx encourages direct mutation of observable state within actions.
- **Use strict mode:** Enforce predictable state changes by using `use strict` or enabling strict mode in Mobx to ensure all state modifications happen within actions.
- **Keep actions in MobX stores:** Encapsulate state mutations within Mobx actions to ensure data integrity and simplify debugging.
- **Prefer class vs object syntax:** Use class syntax for creating stores as it offers better structure, inheritance, and maintainability.
- **Inject store rather than importing:**  Use dependency injection (e.g., React Context) to provide stores to components instead of importing them directly to improve testability and reduce tight coupling.

## 2. Code Organization and Structure

- **Directory Structure:**
    - `/src`:
        - `/components`: React components (presentational and container components).
        - `/stores`: Mobx stores containing observable state and actions.
        - `/models`: Data models and types definitions.
        - `/services`: API clients and other services.
        - `/utils`: Utility functions.
        - `/hooks`: Custom React hooks.
        - `/constants`: Application-wide constants.
    - `/test`: Unit, integration, and end-to-end tests.

- **File Naming Conventions:**
    - Components: `ComponentName.jsx` or `ComponentName.tsx`
    - Stores: `StoreName.store.js` or `StoreName.store.ts`
    - Models: `ModelName.model.js` or `ModelName.model.ts`
    - Services: `ServiceName.service.js` or `ServiceName.service.ts`
    - Hooks: `useHookName.js` or `useHookName.ts`

- **Module Organization:**
    - Group related components, stores, and models into modules.
    - Use clear and descriptive module names.
    - Avoid circular dependencies between modules.

- **Component Architecture:**
    - **Presentational Components:**  Responsible for rendering UI and receiving data via props.
    - **Container Components:**  Connect presentational components to Mobx stores and manage state updates.
    - Use the `observer` decorator/function from `mobx-react-lite` to make components react to observable changes.

- **Code Splitting Strategies:**
    - Use React's `lazy` and `Suspense` components for lazy loading routes or components.
    - Implement route-based code splitting to load only necessary code for the current route.
    - Consider using `dynamic imports` for on-demand loading of modules.

## 3. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Observer Pattern:** Mobx uses the observer pattern to automatically update components when observable state changes.
    - **Dependency Injection:** Use React Context to inject stores into components, making them more testable and reusable.
    - **Provider Pattern:** Use a Provider component to make stores available to the component tree.

- **Recommended Approaches:**
    - **Form Handling:** Use Mobx to manage form state and validation.
    - **Asynchronous Operations:** Use Mobx actions to handle asynchronous operations and update the state accordingly.
    - **List Rendering:** Optimize list rendering using `key` props and `React.memo`.

- **Anti-patterns and Code Smells:**
    - **Modifying State Outside Actions:** Always modify observable state within Mobx actions to ensure data consistency.
    - **Deeply Nested Observables:** Avoid deeply nested observable objects as they can impact performance.
    - **Computing Data in Components:**  Use `computed` properties in stores to derive data instead of computing it directly in components.
    - **Over-Observing:** Only observe the specific parts of the store that a component needs.

- **State Management Best Practices:**
    - **Single Source of Truth:** Keep the application state in Mobx stores and avoid duplicating data.
    - **Normalization:** Normalize data to reduce redundancy and improve data consistency.
    - **Immutability (with Mutation):**  While Mobx encourages mutation, treat observable state as immutable from the perspective of components (i.e., components shouldn't directly mutate the store).

- **Error Handling Patterns:**
    - Implement centralized error handling using a dedicated error store.
    - Use try-catch blocks within actions to handle potential errors.
    - Display user-friendly error messages to the user.

## 4. Performance Considerations

- **Optimization Techniques:**
    - **`React.memo`:**  Wrap components with `React.memo` to prevent unnecessary re-renders when props haven't changed.
    - **`useMemo` and `useCallback`:**  Use `useMemo` and `useCallback` hooks to memoize values and callbacks, preventing unnecessary re-renders of child components.
    - **Computed Properties:**  Use `computed` properties to derive data from observable state and cache the results.
    - **Optimize List Rendering:**  Use `key` props when rendering lists and consider using virtualized lists for large datasets.

- **Memory Management:**
    - **Dispose of Observers:**  Dispose of observers when components unmount to prevent memory leaks.
    - **Avoid Retaining Large Data Sets:**  Remove or release references to large data sets when they are no longer needed.

- **Rendering Optimization:**
    - **Minimize Re-renders:**  Optimize component rendering by using `React.memo` and avoiding unnecessary state updates.
    - **Use ShouldComponentUpdate (if needed):** In class components, use `shouldComponentUpdate` lifecycle method to control when a component should re-render.  This is less common with `observer`.  Be careful, can lead to bugs.

- **Bundle Size Optimization:**
    - **Code Splitting:** Implement code splitting to reduce the initial bundle size.
    - **Tree Shaking:**  Use a bundler that supports tree shaking to remove unused code.
    - **Minification:**  Minify your code to reduce the bundle size.

- **Lazy Loading Strategies:**
    - **React.lazy and Suspense:** Use React's built-in lazy loading mechanism.
    - **Dynamic Imports:** Use dynamic imports to load modules on demand.

## 5. Security Best Practices

- **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):** Sanitize user inputs to prevent XSS attacks.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection mechanisms.
    - **Injection Attacks:**  Validate user inputs to prevent SQL injection and other injection attacks.

- **Input Validation:**
    - **Server-Side Validation:**  Always validate user inputs on the server-side.
    - **Client-Side Validation:**  Implement client-side validation to provide immediate feedback to the user.

- **Authentication and Authorization Patterns:**
    - **JSON Web Tokens (JWT):**  Use JWTs for authentication and authorization.
    - **Role-Based Access Control (RBAC):**  Implement RBAC to control access to different parts of the application.

- **Data Protection Strategies:**
    - **Encryption:**  Encrypt sensitive data both in transit and at rest.
    - **Data Masking:**  Mask sensitive data to protect user privacy.

- **Secure API Communication:**
    - **HTTPS:**  Use HTTPS for all API communication.
    - **Rate Limiting:**  Implement rate limiting to prevent abuse.

## 6. Testing Approaches

- **Unit Testing Strategies:**
    - Test individual components and stores in isolation.
    - Mock dependencies to isolate the unit under test.
    - Use testing libraries like Jest and React Testing Library.

- **Integration Testing:**
    - Test the interaction between different components and stores.
    - Use a testing environment that closely resembles the production environment.

- **End-to-End Testing:**
    - Test the entire application flow from the user's perspective.
    - Use testing frameworks like Cypress or Selenium.

- **Test Organization:**
    - Organize tests into separate directories for unit, integration, and end-to-end tests.
    - Use descriptive test names.

- **Mocking and Stubbing:**
    - Use mocking libraries to mock dependencies in unit tests.
    - Use stubbing to replace complex dependencies with simpler implementations.

## 7. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Forgetting to decorate components with `observer`.
    - Modifying state outside of actions.
    - Not disposing of observers properly.
    - Over-observing state.

- **Edge Cases:**
    - Handling large datasets efficiently.
    - Dealing with complex asynchronous operations.
    - Managing nested observable objects.

- **Version-Specific Issues:**
    - Be aware of breaking changes in Mobx and React versions.
    - Consult the Mobx and React documentation for migration guides.

- **Compatibility Concerns:**
    - Ensure compatibility with different browsers and devices.
    - Test your application on different platforms.

- **Debugging Strategies:**
    - Use the Mobx devtools to inspect observable state and track changes.
    - Use browser developer tools to debug React components.
    - Add logging statements to track the flow of data and execution.

## 8. Tooling and Environment

- **Recommended Development Tools:**
    - **VS Code:**  A popular code editor with excellent support for React and Mobx.
    - **Mobx Devtools:**  A browser extension for inspecting Mobx state.
    - **React Devtools:**  A browser extension for inspecting React components.

- **Build Configuration:**
    - **Webpack:**  A popular module bundler for React applications.
    - **Parcel:**  A zero-configuration bundler that's easy to use.
    - **Rollup:** A bundler focused on creating libraries and smaller bundles.

- **Linting and Formatting:**
    - **ESLint:**  A JavaScript linter for enforcing code style and preventing errors.
    - **Prettier:**  A code formatter for automatically formatting code.
    - **Husky/lint-staged:** Tools to automatically lint code before commits

- **Deployment Best Practices:**
    - **Continuous Integration/Continuous Deployment (CI/CD):** Automate the build, test, and deployment process.
    - **Caching:**  Use caching to improve performance.
    - **Content Delivery Network (CDN):**  Use a CDN to serve static assets.

- **CI/CD Integration:**
    - Integrate your application with a CI/CD pipeline to automate the build, test, and deployment process.
    - Use tools like Jenkins, Travis CI, or CircleCI.