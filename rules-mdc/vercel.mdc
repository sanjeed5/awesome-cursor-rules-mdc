---
description: Enforces Vercel's recommended coding style, optimization strategies, and security best practices. This guide helps developers build performant, secure, and maintainable applications on the Vercel platform.
globs: *.js,*.jsx,*.ts,*.tsx,*.md,*.mdx,*.json,*.yml,*.yaml,*.html,*.css,*.scss,*.sass
---
- Adhere to Vercel's Style Guide: Utilize linting and styling tools configured according to the Vercel Style Guide for consistent code formatting and style.  Use `@file vercel_style_guide.mdc` to include style guide details.
- Enforce Consistent Coding Style:  Integrate ESLint, Prettier, and TypeScript to automatically enforce coding style and prevent stylistic inconsistencies.  Refer to [Vercel's documentation](https://vercel.com/docs) for setup instructions.
- Optimize Codebase Performance: Focus on optimizing the codebase for faster loading times and improved user experience. Includes optimizing cache, assets, and serverless functions.
- Implement Conformance Checks: Utilize Vercel's Conformance tools to automatically check for performance, security, and code health issues.

## 1. Code Organization and Structure:

- **Directory Structure:**
    - Adopt a clear and consistent directory structure (e.g., `components`, `pages`, `lib`, `api`, `styles`, `public`).  Group related files logically to improve maintainability.
    - Use a `src` directory to encapsulate all source code, separating it from configuration files and other project assets.
- **File Naming Conventions:**
    - Use descriptive and consistent file names (e.g., `Button.tsx`, `useUser.ts`, `api/products.js`).
    - Prefer camelCase for JavaScript/TypeScript files and kebab-case for CSS/SCSS files.
- **Module Organization:**
    - Organize code into reusable modules or components. Favor small, focused modules with well-defined interfaces.
    - Use ES modules (`import`/`export`) for modularity and dependency management.
- **Component Architecture:**
    - Employ a component-based architecture (e.g., using React, Vue, or Svelte) to build reusable UI elements.
    - Follow the principles of separation of concerns and single responsibility.
    - Consider using a design system or component library to maintain consistency across the application.
- **Code Splitting Strategies:**
    - Implement code splitting to reduce the initial bundle size and improve loading times.  Use dynamic imports for route-based or component-based splitting.
    - Analyze bundle size using tools like `webpack-bundle-analyzer` to identify large dependencies.

## 2. Common Patterns and Anti-patterns:

- **Design Patterns Specific to Vercel:**
    - Serverless Functions: Use serverless functions for API endpoints and background tasks. Optimize function size and execution time to minimize cold starts.
    - Edge Functions: Use edge functions to perform operations closer to the user, reducing latency for geographically distributed users.
    - Incremental Static Regeneration (ISR): Use ISR to combine the benefits of static generation and server-side rendering.
- **Recommended Approaches for Common Tasks:**
    - Data Fetching: Use `getStaticProps` or `getServerSideProps` in Next.js for data fetching in pages.
    - API Routes: Create API routes in the `pages/api` directory of a Next.js project for handling API requests.
    - Environment Variables: Use environment variables to store sensitive information and configuration settings.
- **Anti-patterns and Code Smells to Avoid:**
    - Large components: Break down large components into smaller, more manageable pieces.
    - Deeply nested components: Avoid excessive nesting of components, which can make code harder to read and maintain.
    - Over-fetching data: Fetch only the data that is needed by a component.
    - Mutating state directly: Avoid mutating state directly, which can lead to unexpected behavior.
- **State Management Best Practices:**
    - Choose a state management solution (e.g., Redux, Zustand, Recoil, React Context) that is appropriate for the application's complexity.
    - Follow recommended patterns for managing state (e.g., reducers, actions, selectors).
    - Avoid storing too much data in the global state.
- **Error Handling Patterns:**
    - Implement comprehensive error handling throughout the application.
    - Use try-catch blocks to catch exceptions.
    - Log errors to a monitoring service.
    - Display user-friendly error messages.

## 3. Performance Considerations:

- **Optimization Techniques:**
    - Minimize bundle size by removing unused code and dependencies (tree shaking).
    - Compress images and other assets.
    - Use a content delivery network (CDN) to serve static assets.
    - Optimize database queries.
    - Cache frequently accessed data.
- **Memory Management:**
    - Avoid memory leaks by properly releasing resources.
    - Use garbage collection to reclaim unused memory.
    - Profile application memory usage to identify potential issues.
- **Rendering Optimization:**
    - Use memoization techniques (e.g., `React.memo`, `useMemo`) to prevent unnecessary re-renders.
    - Virtualize long lists to improve rendering performance.
    - Use code splitting to reduce the initial bundle size.
- **Bundle Size Optimization:**
    - Analyze bundle size with tools like `webpack-bundle-analyzer` or `source-map-explorer`.
    - Remove unused dependencies.
    - Minify code.
    - Compress code with gzip or Brotli.
- **Lazy Loading Strategies:**
    - Implement lazy loading for images, components, and routes.
    - Use the `IntersectionObserver` API to detect when an element is visible in the viewport.

## 4. Security Best Practices:

- **Common Vulnerabilities and How to Prevent Them:**
    - Cross-site scripting (XSS): Sanitize user input to prevent XSS attacks.  Use a library like DOMPurify.
    - Cross-site request forgery (CSRF): Use CSRF tokens to protect against CSRF attacks.
    - SQL injection: Use parameterized queries or an ORM to prevent SQL injection attacks.
    - Authentication and authorization vulnerabilities: Implement strong authentication and authorization mechanisms.
- **Input Validation:**
    - Validate all user input on both the client and server sides.
    - Use a validation library to simplify the validation process.
- **Authentication and Authorization Patterns:**
    - Use a secure authentication protocol (e.g., OAuth 2.0, JWT).
    - Implement role-based access control (RBAC) to restrict access to sensitive resources.
- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to secure communication between the client and server.
    - Protect against data breaches by implementing appropriate security measures.
- **Secure API Communication:**
    - Use HTTPS to encrypt API traffic.
    - Authenticate and authorize API requests.
    - Limit API rate to prevent abuse.

## 5. Testing Approaches:

- **Unit Testing Strategies:**
    - Write unit tests for individual components, functions, and modules.
    - Use a unit testing framework (e.g., Jest, Mocha, Jasmine).
    - Mock external dependencies.
- **Integration Testing:**
    - Write integration tests to verify that different parts of the application work together correctly.
    - Test interactions between components, modules, and APIs.
- **End-to-End Testing:**
    - Write end-to-end tests to simulate user interactions and verify that the application works as expected.
    - Use an end-to-end testing framework (e.g., Cypress, Playwright).
- **Test Organization:**
    - Organize tests into separate directories based on the type of test (e.g., unit, integration, e2e).
    - Use descriptive test names.
- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components and functions during testing.
    - Mock external dependencies to avoid making real API calls.

## 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes Developers Make:**
    - Over-reliance on client-side rendering.
    - Neglecting performance optimization.
    - Ignoring security vulnerabilities.
    - Not writing enough tests.
- **Edge Cases to Be Aware Of:**
    - Handling errors gracefully.
    - Supporting different browsers and devices.
    - Dealing with slow network connections.
- **Version-Specific Issues:**
    - Be aware of breaking changes in Vercel and its dependencies.
    - Test application thoroughly after upgrading Vercel or any dependencies.
- **Compatibility Concerns:**
    - Ensure that the application is compatible with different browsers, devices, and operating systems.
- **Debugging Strategies:**
    - Use browser developer tools to debug client-side code.
    - Use server-side logging to debug server-side code.
    - Use a debugger to step through code and inspect variables.

## 7. Tooling and Environment:

- **Recommended Development Tools:**
    - VS Code with extensions for ESLint, Prettier, and TypeScript.
    - Chrome DevTools for debugging.
    - Vercel CLI for deploying and managing applications.
- **Build Configuration:**
    - Configure build scripts to optimize code and assets.
    - Use environment variables to configure the build process.
- **Linting and Formatting:**
    - Use ESLint and Prettier to enforce consistent code style.
    - Configure linting and formatting rules to match the Vercel Style Guide.
- **Deployment Best Practices:**
    - Deploy application to Vercel using the Vercel CLI or Git integration.
    - Configure environment variables for production.
    - Monitor application performance and logs.
- **CI/CD Integration:**
    - Integrate with a CI/CD pipeline (e.g., GitHub Actions, CircleCI) to automate the build, test, and deployment process.
    - Run tests and linters as part of the CI/CD pipeline.

@file vercel_style_guide.mdc