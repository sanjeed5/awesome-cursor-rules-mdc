---
description: This rule provides comprehensive best practices for developing Qwik applications, covering code organization, performance, security, testing, and common pitfalls. It aims to guide developers in writing maintainable, efficient, and secure Qwik code.
globs: *.ts?(x)
---
# Qwik Library Best Practices

This document outlines the best practices for developing Qwik applications. Adhering to these guidelines will help you write maintainable, efficient, and secure code.

## 1. Code Organization and Structure

- **Directory Structure:**
    - `src/`: Contains the source code of your application.
        - `components/`: Reusable UI components.
        - `routes/`: Defines the application's routes and pages.  Use the file system routing for easy setup. Each file under `routes` will become a route based on the filename.
        - `services/`: Contains business logic and data fetching services.
        - `utils/`: Utility functions and helpers.
        - `styles/`: Global styles and themes.
        - `types/`: Type definitions used throughout the application.
    - `public/`: Static assets like images, fonts, and favicons.
    - `vite.config.ts`: Vite configuration file.
    - `tsconfig.json`: TypeScript configuration file.
    - `.eslintrc.js`: ESLint configuration file.

- **File Naming Conventions:**
    - Components: `ComponentName.tsx` (e.g., `MyButton.tsx`).
    - Services: `serviceName.ts` (e.g., `userService.ts`).
    - Utilities: `utilityName.ts` (e.g., `dateUtils.ts`).
    - Styles: `componentName.css` or `componentName.module.css`.
    - Routes: Filename dictates the route path (e.g., `index.tsx` for `/`, `about.tsx` for `/about`).

- **Module Organization:**
    - Group related components, services, and utilities into modules.
    - Use `index.ts` files to re-export members from a module for cleaner imports.
    - Keep modules focused and avoid creating large, monolithic modules.

- **Component Architecture:**
    - Favor small, reusable components.
    - Use a component-based architecture for building UIs.
    - Separate concerns: presentational components and container components.
        - Presentational components: Focus on rendering UI elements based on props.
        - Container components: Handle data fetching, state management, and business logic.
    - Use Qwik's `component$` and `use*` APIs for creating and managing components.

- **Code Splitting Strategies:**
    - Use Qwik's automatic code splitting capabilities.
    - Ensure routes are lazily loaded by their nature in Qwik.
    - For larger components or modules, consider manually triggering lazy loading.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Composition:** Build complex components by composing smaller, reusable components.
    - **Provider Pattern:** Use Context to share state or services across multiple components.
    - **Hooks:** Use Qwik's `use*` hooks to manage state, effects, and other side effects.
    - **Factories:** For complex object creation, consider using factory functions.

- **Recommended Approaches:**
    - Use Qwik's state management features (`useStore`, `useSignal`) for managing component state.
    - Use Qwik City's routing capabilities for defining routes and navigation.
    - Use Qwik's optimizer and server rendering for performance.

- **Anti-patterns and Code Smells:**
    - **Large Components:** Avoid creating overly large components. Break them down into smaller, more manageable pieces.
    - **Tight Coupling:** Reduce dependencies between components and modules.
    - **Mutating Props:** Never directly modify props passed to a component.
    - **Global State Abuse:** Avoid using global state excessively. Prefer component-level state management.
    - **Ignoring Server Rendering:** Ensure your components are server-renderable for initial page load performance.

- **State Management Best Practices:**
    - Use Qwik's `useStore` for reactive state management.
    - Consider `useSignal` for simpler reactive values.
    - Avoid direct DOM manipulation; rely on Qwik's component rendering.

- **Error Handling Patterns:**
    - Use `try...catch` blocks to handle errors gracefully.
    - Display user-friendly error messages.
    - Log errors for debugging and monitoring.
    - Consider using error boundary components to prevent entire application crashes.

## 3. Performance Considerations

- **Optimization Techniques:**
    - Use Qwik's optimizer to minimize bundle size and improve performance.
    - Leverage server rendering to improve initial page load time.
    - Optimize images and other assets.
    - Use Qwik's built-in lazy loading features.

- **Memory Management:**
    - Avoid memory leaks by properly cleaning up resources.
    - Unsubscribe from subscriptions and event listeners when components unmount.

- **Rendering Optimization:**
    - Use `track` property in `useStore` to selectively trigger updates.
    - Memoize expensive computations using `useComputed$()`
    - Avoid unnecessary re-renders by ensuring component props are stable.

- **Bundle Size Optimization:**
    - Use code splitting to reduce the initial bundle size.
    - Remove unused code and dependencies.
    - Use tree shaking to eliminate dead code.

- **Lazy Loading Strategies:**
    - Use Qwik's built-in lazy loading features for components and routes.
    - Consider using dynamic imports for loading modules on demand.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user inputs and using Qwik's built-in escaping mechanisms.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection by using tokens and validating requests.
    - **Injection Attacks:** Prevent SQL injection and other injection attacks by using parameterized queries and escaping user inputs.

- **Input Validation:**
    - Validate all user inputs on both the client and server sides.
    - Use appropriate data types and formats.
    - Sanitize user inputs to prevent XSS attacks.

- **Authentication and Authorization Patterns:**
    - Use a secure authentication mechanism, such as OAuth or JWT.
    - Implement proper authorization checks to ensure users only have access to the resources they are authorized to access.

- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to secure communication between the client and server.
    - Implement proper access controls to protect data from unauthorized access.

- **Secure API Communication:**
    - Use HTTPS for all API requests.
    - Validate API responses.
    - Implement rate limiting to prevent abuse.

## 5. Testing Approaches

- **Unit Testing Strategies:**
    - Write unit tests for individual components, services, and utilities.
    - Use a testing framework like Jest or Vitest.
    - Mock dependencies to isolate units under test.

- **Integration Testing:**
    - Write integration tests to verify interactions between different parts of the application.
    - Test the integration of components, services, and routes.

- **End-to-end Testing:**
    - Write end-to-end tests to verify the entire application flow.
    - Use a testing framework like Playwright or Cypress.
    - Test user interactions and navigation.

- **Test Organization:**
    - Organize tests into separate directories based on the type of test.
    - Use descriptive test names.
    - Follow a consistent testing style.

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate units under test.
    - Mock external dependencies, such as APIs and databases.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Forgetting to use Qwik's `component$` for creating components.
    - Not using `use*` hooks for managing state and effects.
    - Over-reliance on `globalThis`.

- **Edge Cases:**
    - Handling errors during server rendering.
    - Dealing with different browser environments.
    - Managing application state across multiple pages.

- **Version-Specific Issues:**
    - Be aware of breaking changes between Qwik versions.
    - Consult the Qwik changelog for migration instructions.

- **Compatibility Concerns:**
    - Ensure your application is compatible with different browsers and devices.
    - Use polyfills to support older browsers.

- **Debugging Strategies:**
    - Use browser developer tools for debugging.
    - Use Qwik's built-in debugging features.
    - Log errors and warnings.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - Visual Studio Code with the Qwik extension.
    - Node.js and npm.
    - A modern web browser.

- **Build Configuration:**
    - Use Vite for building and bundling Qwik applications.
    - Configure Vite to optimize the build for production.

- **Linting and Formatting:**
    - Use ESLint and Prettier for linting and formatting code.
    - Configure ESLint and Prettier to follow Qwik's coding style.

- **Deployment Best Practices:**
    - Deploy your application to a serverless environment, such as Netlify or Vercel.
    - Configure your server to serve static assets efficiently.
    - Use a CDN to cache static assets.

- **CI/CD Integration:**
    - Integrate your application with a CI/CD pipeline.
    - Automate testing, building, and deployment.