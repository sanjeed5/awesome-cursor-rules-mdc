---
description: This rule provides comprehensive best practices for developing with Turbopack, covering code organization, performance, security, testing, and tooling to ensure efficient and maintainable applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
# Turbopack Best Practices

This document outlines best practices for developing with Turbopack, focusing on code organization, performance, security, testing, and tooling.

## 1. Core Principles

- **Leverage the Turbo Engine:** Turbopack's core strength lies in its Turbo engine for function-level caching. Understand and utilize this for incremental builds.
- **Embrace Incremental Computation:** Be mindful of how changes affect the build process. Design your application to maximize incremental computation benefits.
- **Use TypeScript:** Turbopack has built-in support for TypeScript and JSX, so use them to improve code quality and developer experience.  Make full use of TypeScript's features like generics, interfaces and types.
- **Follow Next.js Conventions:** When using Turbopack within Next.js, adhere to Next.js's recommended patterns and practices.
- **Avoid direct webpack configurations:** Turbopack aims to abstract away webpack configurations, prefer using Next.js configurations to customize Turbopack.

## 2. Code Organization and Structure

- **Directory Structure:**
    - **`src/`:**  All application source code should reside within the `src/` directory.
    - **`components/`:** Reusable UI components.
    - **`pages/`:**  (Next.js) Pages for routing.
    - **`api/`:** (Next.js) API routes.
    - **`lib/` or `utils/`:** Utility functions and shared logic.
    - **`types/` or `interfaces/`:** TypeScript type definitions and interfaces.
    - **`styles/` or `css/`:** Global styles and CSS modules.
    - **`public/`:** Static assets (images, fonts, etc.).
- **File Naming Conventions:**
    - Use descriptive names for files and directories.
    - Component files: `ComponentName.jsx` or `ComponentName.tsx`.
    - Style files: `ComponentName.module.css` or `ComponentName.module.scss`.
    - Utility files: `utilityName.js` or `utilityName.ts`.
- **Module Organization:**
    - Group related code into modules with clear responsibilities.
    - Export only what is necessary from each module.
    - Use ES modules (import/export) for modularity.
- **Component Architecture:**
    - **Atomic Design:** Consider using Atomic Design principles to create a scalable and maintainable component architecture.
    - **Composition:** Favor composition over inheritance for component reusability.
    - **Separation of Concerns:**  Separate UI logic, data fetching, and state management within components.
    - **Keep components small:** Focus on single responsability principle.
- **Code Splitting Strategies:**
    - **Dynamic Imports:** Use dynamic imports (`import()`) to split code into smaller chunks that are loaded on demand.
    - **Route-Based Splitting:** (Next.js) Each page in the `pages/` directory is automatically code-split.
    - **Component-Based Splitting:** Split large components into smaller, lazily loaded sub-components.
    - **Vendor Splitting:** Turbopack automatically splits vendor code (third-party libraries) into separate chunks. 

## 3. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Higher-Order Components (HOCs):** Reusable logic for components (use with caution, consider hooks).
    - **Render Props:** Sharing code between React components using a prop whose value is a function.
    - **Hooks:**  Reusable stateful logic for functional components.
    - **Context API:**  Share data that is considered "global" for a tree of React components.
- **Recommended Approaches:**
    - **Data Fetching:** Use `getServerSideProps`, `getStaticProps`, or `getInitialProps` (Next.js) for data fetching depending on your requirements.
    - **API Routes:** Use Next.js API routes to create serverless functions for handling API requests.
    - **State Management:** Choose a state management library (Redux, Zustand, Jotai, Recoil) based on the complexity of your application.
- **Anti-patterns:**
    - **Global State Mutation:** Avoid directly mutating global state, use reducers or state management libraries.
    - **Over-Fetching:** Fetch only the data that is needed by a component.
    - **Tight Coupling:**  Reduce dependencies between modules to improve maintainability.
    - **Long Component Files:** Avoid having components larger than 200-300 lines, break down them into smaller components.
- **State Management Best Practices:**
    - **Centralized State:** Manage application state in a central store.
    - **Immutability:** Treat state as immutable to prevent unexpected side effects.
    - **Reducers:** Use reducers to update state based on actions.
    - **Selectors:** Use selectors to derive data from the state.
- **Error Handling Patterns:**
    - **Error Boundaries:**  Use error boundaries to catch JavaScript errors anywhere in a component tree.
    - **Centralized Error Logging:**  Log errors to a central service for monitoring and debugging.
    - **Graceful Degradation:**  Handle errors gracefully and provide informative messages to the user.
    - **Retry mechanisms:** Implement retry mechanisms for failed requests.

## 4. Performance Considerations

- **Optimization Techniques:**
    - **Caching:** Leverage Turbopack's caching capabilities to avoid unnecessary re-builds.
    - **Memoization:** Use `React.memo` or `useMemo` to memoize components and values.
    - **Debouncing and Throttling:** Use debouncing and throttling to reduce the frequency of function calls.
- **Memory Management:**
    - **Avoid Memory Leaks:**  Be careful to clean up event listeners and timers when components unmount.
    - **Garbage Collection:**  Understand how JavaScript garbage collection works and avoid creating unnecessary objects.
- **Rendering Optimization:**
    - **Virtualization:** Use virtualization libraries for rendering large lists.
    - **Code Splitting:** Split the codebase into smaller chunks using dynamic imports for faster initial load times.
    - **Image Optimization:** Optimize images using tools like `next/image` or `cloudinary`.
- **Bundle Size Optimization:**
    - **Tree Shaking:**  Remove unused code from the bundle by using ES modules and configuring your bundler correctly.
    - **Minification:** Minify JavaScript and CSS code to reduce bundle size.
    - **Compression:** Compress assets using gzip or Brotli.
- **Lazy Loading Strategies:**
    - **Component-Level Lazy Loading:**  Lazy load components that are not initially visible.
    - **Image Lazy Loading:**  Lazy load images that are not initially visible.
    - **Route-Based Lazy Loading:**  Lazy load routes that are not frequently visited.

## 5. Security Best Practices

- **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):**  Prevent XSS attacks by sanitizing user input and using appropriate escaping techniques.
    - **Cross-Site Request Forgery (CSRF):**  Prevent CSRF attacks by using CSRF tokens.
    - **SQL Injection:**  Prevent SQL injection attacks by using parameterized queries or ORMs.
    - **Authentication and Authorization Issues:**  Secure your API endpoints with proper authentication and authorization mechanisms.
- **Input Validation:**
    - **Validate all user input:** Sanitize and validate all user input on both the client and server.
    - **Use appropriate data types:** Enforce data types to prevent unexpected values.
- **Authentication and Authorization Patterns:**
    - **JSON Web Tokens (JWT):**  Use JWTs for authentication and authorization.
    - **Role-Based Access Control (RBAC):**  Implement RBAC to control access to resources based on user roles.
    - **OAuth:**  Use OAuth for third-party authentication.
- **Data Protection Strategies:**
    - **Encryption:** Encrypt sensitive data at rest and in transit.
    - **Data Masking:** Mask sensitive data in logs and reports.
    - **Data Retention Policies:**  Implement data retention policies to ensure compliance with regulations.
- **Secure API Communication:**
    - **HTTPS:**  Use HTTPS to encrypt communication between the client and server.
    - **Rate Limiting:**  Implement rate limiting to prevent abuse and denial-of-service attacks.
    - **API Keys:**  Use API keys for authentication.

## 6. Testing Approaches

- **Unit Testing Strategies:**
    - **Test individual components and functions:**  Write unit tests to verify the behavior of individual components and functions.
    - **Use mocking and stubbing:**  Use mocking and stubbing to isolate components from their dependencies.
    - **Test edge cases:**  Test edge cases and error conditions.
- **Integration Testing:**
    - **Test the interaction between components:**  Write integration tests to verify the interaction between components.
    - **Test data flow:**  Test the flow of data between components and services.
- **End-to-End Testing:**
    - **Test the entire application flow:**  Write end-to-end tests to verify the entire application flow.
    - **Use browser automation tools:**  Use browser automation tools like Cypress or Puppeteer.
- **Test Organization:**
    - **Keep tests close to the code:**  Organize tests in the same directory as the code they test.
    - **Use descriptive test names:**  Use descriptive test names that clearly describe what the test is verifying.
- **Mocking and Stubbing:**
    - **Use mocking libraries:**  Use mocking libraries like Jest or Sinon to create mocks and stubs.
    - **Avoid over-mocking:**  Mock only the dependencies that are necessary.

## 7. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - **Incorrectly configured `tsconfig.json`:** Ensure paths and baseUrl are correctly configured for absolute imports.
    - **Misunderstanding caching:** Not utilizing Turbopack's caching efficiently, leading to slower builds.
    - **Not using TypeScript features**: Not leveraging the benefits of Typescript and writing javascript code that misses out on typesafety.
- **Edge Cases:**
    - **Complex dependencies:**  Be aware of how complex dependencies can affect build times.
    - **Large file sizes:**  Optimize large file sizes to improve performance.
- **Version-Specific Issues:**
    - **Breaking changes:**  Be aware of breaking changes in Turbopack and Next.js releases.
    - **Compatibility issues:**  Ensure that your dependencies are compatible with the versions of Turbopack and Next.js that you are using.
- **Compatibility Concerns:**
    - **Browser compatibility:**  Test your application in different browsers to ensure compatibility.
    - **Device compatibility:**  Test your application on different devices to ensure compatibility.
- **Debugging Strategies:**
    - **Use debugging tools:**  Use browser developer tools and debugging tools to identify and fix issues.
    - **Log statements:**  Use log statements to track the flow of execution and identify errors.
    - **Reproducible steps:**  Create reproducible steps to help isolate and fix issues.

## 8. Tooling and Environment

- **Recommended Development Tools:**
    - **VS Code:**  A popular code editor with excellent support for JavaScript, TypeScript, and React.
    - **ESLint:**  A linter that helps you identify and fix code style issues.
    - **Prettier:**  A code formatter that automatically formats your code.
    - **Chrome Developer Tools:** Powerful tools for debugging and profiling web applications.
- **Build Configuration:**
    - **Configure `tsconfig.json`:** Configure `tsconfig.json` to enable TypeScript features and specify compiler options.
    - **Configure ESLint:** Configure ESLint to enforce code style guidelines.
    - **Configure Prettier:** Configure Prettier to automatically format code.
- **Linting and Formatting:**
    - **Use ESLint and Prettier:**  Use ESLint and Prettier to automatically lint and format your code.
    - **Integrate with your editor:**  Integrate ESLint and Prettier with your code editor to automatically lint and format code on save.
- **Deployment Best Practices:**
    - **Use a CI/CD pipeline:**  Use a CI/CD pipeline to automate the build, test, and deployment process.
    - **Deploy to a CDN:**  Deploy static assets to a CDN for faster delivery.
    - **Monitor your application:**  Monitor your application for errors and performance issues.
- **CI/CD Integration:**
    - **Use GitHub Actions, GitLab CI, or CircleCI:**  Use a CI/CD platform to automate the build, test, and deployment process.
    - **Run tests in CI/CD:**  Run tests in your CI/CD pipeline to ensure that code changes do not introduce new bugs.
    - **Automate deployments:**  Automate deployments to production and staging environments.

## 9. Additional Considerations

- **Experimentation:** Turbopack is rapidly evolving. Experiment with new features and configurations to optimize your build process.
- **Community Engagement:** Participate in the Turbopack community to share your experiences and learn from others.
- **Documentation:**  Refer to the official Turbopack documentation for the latest information and guidance.

By following these best practices, you can leverage the full potential of Turbopack to build high-performance, maintainable, and secure applications.