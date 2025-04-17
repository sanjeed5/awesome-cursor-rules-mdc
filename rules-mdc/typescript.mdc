---
description: Enforces best practices for TypeScript development, including coding standards, performance considerations, and common pitfalls. This rule provides actionable guidance for developers to write clean, maintainable, and scalable TypeScript code.
globs: *.ts?(x)
---
# TypeScript Best Practices and Coding Standards

This document outlines best practices and coding standards for developing TypeScript applications. Following these guidelines will help ensure code quality, maintainability, and scalability.

## 1. Code Organization and Structure

- **Directory Structure:**
    - **Feature-based:** Group files related to a specific feature within a dedicated directory.
    
    src/
    ├── feature1/
    │   ├── components/
    │   │   ├── ComponentA.tsx
    │   │   └── ComponentB.tsx
    │   ├── services/
    │   │   └── feature1.service.ts
    │   ├── types.ts
    │   └── feature1.module.ts
    ├── feature2/ 
    │   └── ...
    └── shared/
        ├── components/
        │   └── ReusableComponent.tsx
        ├── services/
        │   └── api.service.ts
        └── types/
            └── global.d.ts
    
    - **Type-based:**  Separate files based on their role (components, services, types, etc.).
    
    src/
    ├── components/
    │   ├── Feature1Component.tsx
    │   └── Feature2Component.tsx
    ├── services/
    │   ├── feature1.service.ts
    │   └── feature2.service.ts
    ├── types/
    │   ├── feature1.types.ts
    │   └── feature2.types.ts
    └── modules/
        ├── feature1.module.ts
        └── feature2.module.ts
    
    - Choose the structure that best fits your project's complexity and team's preferences.  Consistency is key.

- **File Naming Conventions:**
    - Use descriptive and consistent file names.
    - Components: `ComponentName.tsx`
    - Services: `serviceName.service.ts`
    - Types: `typeName.types.ts` or `types.ts` (if grouping related types)
    - Modules: `moduleName.module.ts`
    - Interfaces: `IInterfaceName.ts` (or `interfaceName.interface.ts` if preferred and consistent throughout the codebase)

- **Module Organization:**
    - Use ES Modules (`import`/`export`) for modularity and reusability.
    - Favor named exports over default exports for better discoverability and refactoring.
    - Group related functionality into modules.
    - Avoid circular dependencies.

- **Component Architecture:**
    - Consider using component-based architectures like React, Angular, or Vue.js.
    - Follow component design principles: Single Responsibility Principle, separation of concerns.
    - Use composition over inheritance.
    - Keep components small and focused.

- **Code Splitting Strategies:**
    - Split your application into smaller chunks to improve initial load time.
    - Implement lazy loading for modules and components that are not immediately needed.
    - Use dynamic imports (`import()`).
    - Webpack, Parcel, and other bundlers offer built-in support for code splitting.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Factory Pattern:**  Use factories to create objects with complex initialization logic.
    - **Singleton Pattern:**  Use sparingly, and only when a single instance is truly required.
    - **Observer Pattern:**  Implement reactive patterns for handling events and data changes.
    - **Strategy Pattern:**  Define a family of algorithms and encapsulate each one into a separate class.
    - **Dependency Injection:** Reduce coupling by injecting dependencies into components and services.

- **Recommended Approaches:**
    - **Data Fetching:** Use libraries like `axios` or `fetch` for making API requests.
    - **State Management:** Choose a state management solution appropriate for your application's complexity (e.g., React Context, Redux, Zustand, MobX).
    - **Form Handling:** Use libraries like `react-hook-form` or `formik` for managing form state and validation.

- **Anti-patterns and Code Smells:**
    - **`any` type overuse:** Avoid using `any` as much as possible. Use more specific types or generics.
    - **Long methods/functions:** Break down large functions into smaller, more manageable units.
    - **Deeply nested code:** Refactor deeply nested code to improve readability.
    - **Magic numbers/strings:** Use constants for values that have a specific meaning.
    - **Duplicated code:** Extract common logic into reusable functions or components.
    - **Ignoring errors:** Always handle errors gracefully.  Don't just catch and ignore them.
    - **Over-commenting:** Write self-documenting code and use comments only when necessary to explain complex logic.

- **State Management Best Practices:**
    - Choose a state management library based on project needs: React Context API, Redux, Zustand, MobX.
    - Keep state minimal and derive values where possible.
    - Follow immutable update patterns (especially with Redux).
    - Use selectors to access state.
    - Centralize state logic.

- **Error Handling Patterns:**
    - Use `try...catch` blocks to handle potential errors.
    - Implement a global error handler to catch unhandled exceptions.
    - Use error logging to track errors in production.
    - Use discriminated unions for representing different error states.
    - Implement retry mechanisms for transient errors.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Memoization:** Use memoization techniques (e.g., `React.memo`, `useMemo`) to avoid unnecessary re-renders.
    - **Debouncing and Throttling:** Limit the rate at which functions are executed in response to user input.
    - **Virtualization:** Use virtualization for rendering large lists or tables.
    - **Code Splitting:** Split your code into smaller chunks to reduce initial load time.

- **Memory Management:**
    - Avoid memory leaks by properly cleaning up resources (e.g., event listeners, timers).
    - Use weak references to avoid circular dependencies that can prevent garbage collection.
    - Profile your application to identify memory leaks.

- **Rendering Optimization:**
    - Minimize DOM manipulations.
    - Use CSS transforms and animations instead of JavaScript animations.
    - Optimize images and other assets.
    - Use the `shouldComponentUpdate` lifecycle method or `React.memo` to prevent unnecessary re-renders.

- **Bundle Size Optimization:**
    - Use tree shaking to remove unused code from your bundle.
    - Minify your code to reduce bundle size.
    - Compress your code using gzip or Brotli.
    - Use code splitting to load only the code that is needed for a particular page or component.

- **Lazy Loading Strategies:**
    - Lazy load modules and components that are not immediately needed.
    - Use dynamic imports (`import()`) to load modules on demand.
    - Implement a loading indicator to provide feedback to the user while the module is loading.

## 4. Security Best Practices

- **Common Vulnerabilities and Prevention:**
    - **Cross-Site Scripting (XSS):** Sanitize user input and escape output to prevent XSS attacks.
    - **Cross-Site Request Forgery (CSRF):** Use anti-CSRF tokens to protect against CSRF attacks.
    - **SQL Injection:** Use parameterized queries or ORMs to prevent SQL injection attacks (relevant for backend TypeScript).
    - **Denial of Service (DoS):** Implement rate limiting and other measures to prevent DoS attacks.
    - **Man-in-the-Middle (MitM):** Use HTTPS to encrypt communication between the client and server.

- **Input Validation:**
    - Validate all user input on both the client and server sides.
    - Use strong validation rules to prevent malicious input.
    - Sanitize user input to remove potentially harmful characters.

- **Authentication and Authorization Patterns:**
    - Use a secure authentication mechanism to verify user identities.
    - Implement authorization checks to control access to resources.
    - Use role-based access control (RBAC) to manage user permissions.
    - Use JSON Web Tokens (JWT) for stateless authentication.

- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use strong encryption algorithms.
    - Store passwords securely using a hashing algorithm and salt.
    - Protect API keys and other secrets.

- **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement proper authentication and authorization for API endpoints.
    - Use rate limiting to prevent abuse.
    - Validate API requests and responses.

## 5. Testing Approaches

- **Unit Testing Strategies:**
    - Write unit tests for individual functions and components.
    - Use mocking and stubbing to isolate units of code.
    - Test edge cases and error conditions.
    - Aim for high code coverage.

- **Integration Testing:**
    - Test the interaction between different modules and components.
    - Verify that different parts of the application work together correctly.

- **End-to-End Testing:**
    - Test the entire application from the user's perspective.
    - Use tools like Cypress or Playwright to automate end-to-end tests.

- **Test Organization:**
    - Organize tests in a way that makes it easy to find and run them.
    - Group tests by feature or module.
    - Use descriptive test names.

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate units of code and simulate dependencies.
    - Use mocking libraries like Jest or Sinon.js.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Incorrectly handling asynchronous operations (Promises, async/await).
    - Not handling errors properly.
    - Overusing the `any` type.
    - Ignoring compiler warnings.
    - Not keeping dependencies up to date.

- **Edge Cases:**
    - Handling different browser versions and devices.
    - Dealing with network latency and failures.
    - Handling different time zones and locales.
    - Handling large datasets and complex calculations.

- **Version-Specific Issues:**
    - Be aware of breaking changes in new versions of TypeScript and related libraries.
    - Consult the release notes for each new version to identify potential issues.
    - Use TypeScript's compiler options to target specific ECMAScript versions and maintain backwards compatibility if needed.

- **Compatibility Concerns:**
    - Ensure that your code is compatible with the target browsers and devices.
    - Use polyfills to provide support for older browsers.
    - Test your code on different platforms to identify compatibility issues.

- **Debugging Strategies:**
    - Use a debugger to step through your code and inspect variables.
    - Use console logging to track the flow of execution and identify errors.
    - Use TypeScript's type checking to catch errors early.
    - Use source maps to debug code that has been transpiled or minified.
    - Learn to read and understand stack traces.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **IDE:** Visual Studio Code with the TypeScript extension.
    - **Package Manager:** npm or Yarn.
    - **Bundler:** Webpack, Parcel, or Rollup.
    - **Linter:** ESLint with TypeScript-specific rules.
    - **Formatter:** Prettier.
    - **Testing Framework:** Jest, Mocha, or Jasmine.

- **Build Configuration:**
    - Use a `tsconfig.json` file to configure the TypeScript compiler.
    - Configure compiler options like `target`, `module`, `jsx`, and `strict`.
    - Use TypeScript's project references to organize large projects.

- **Linting and Formatting:**
    - Use ESLint with TypeScript-specific rules to enforce coding standards.
    - Use Prettier to automatically format your code.
    - Integrate linting and formatting into your development workflow using Git hooks or CI/CD pipelines.

- **Deployment Best Practices:**
    - Use a build process to transpile and bundle your code.
    - Minify and compress your code to reduce bundle size.
    - Use a CDN to serve static assets.
    - Implement caching strategies to improve performance.

- **CI/CD Integration:**
    - Integrate your tests and linters into your CI/CD pipeline.
    - Automate the build and deployment process.
    - Use environment variables to configure your application for different environments.