---
description: This rule provides comprehensive best practices and coding standards for SvelteKit development, covering code structure, performance, security, testing, and common pitfalls. It aims to improve code quality, maintainability, and overall project health.
globs: *.svelte
---
# SvelteKit Best Practices

This guide provides comprehensive best practices for developing SvelteKit applications. It covers various aspects, including code organization, performance considerations, security measures, and common pitfalls to avoid.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

*   **`src/lib`**: Reusable components, utilities, and stores. Organize by feature or domain.
*   **`src/routes`**: Defines the application's routes. Each directory represents a route segment.
*   **`src/routes/+page.svelte`**: Represents a page component for a route.
*   **`src/routes/+layout.svelte`**: Layout component that wraps the page. Useful for consistent UI elements.
*   **`src/routes/+page.server.ts`**: Server-side code for handling data fetching and form submissions for a page.
*   **`src/routes/+layout.server.ts`**: Server-side code for layout data, loaded before any +page.server.ts data.
*   **`src/hooks.server.ts`**: Handle server-side requests and responses.  Useful for authentication and session management.
*   **`static`**: Static assets like images, fonts, and other files served directly.
*   **`tests`**: Directory for all tests, mirroring the source structure for clarity.

Example:


src/
├── lib/
│   ├── components/
│   │   ├── Button.svelte
│   │   └── Card.svelte
│   ├── utils/
│   │   ├── api.ts
│   │   └── helpers.ts
│   └── stores/
│       └── user.ts
├── routes/
│   ├── +
│   │   └── page.svelte
│   ├── about/
│   │   ├── +
│   │   │   └── page.svelte
│   │   └── +page.server.ts
│   └── blog/
│       ├── [slug]/
│       │   ├── +
│       │   │   └── page.svelte
│       │   └── +page.server.ts
│       └── +
│           └── page.svelte
├── hooks.server.ts
└── app.d.ts


### 1.2. File Naming Conventions

*   **Components**: PascalCase (e.g., `MyComponent.svelte`).
*   **Utility functions**: camelCase (e.g., `formatDate.ts`).
*   **Stores**: camelCase (e.g., `userStore.ts`).
*   **Route files**: Follow SvelteKit's routing conventions (`+page.svelte`, `+layout.svelte`, etc.).

### 1.3. Module Organization

*   Group related functionality into modules.
*   Use `src/lib` for reusable modules.
*   Employ clear and descriptive module names.
*   Leverage SvelteKit's `$lib` alias for importing modules from `src/lib`.

### 1.4. Component Architecture

*   **Atomic Design**: Break down UI into small, reusable components (atoms, molecules, organisms, templates, pages).
*   **Component Composition**: Compose complex UIs from simpler components.
*   **Props and Events**: Use props for data input and events for component output.
*   **Slots**: Allow parent components to inject content into child components.

### 1.5. Code Splitting Strategies

*   **Dynamic Imports**: Use dynamic imports (`import()`) to load modules on demand.
*   **Route-Based Splitting**: SvelteKit automatically splits code based on routes.
*   **Component-Level Splitting**: Split large components into smaller chunks that can be loaded independently.
*   **Lazy Loading**: Load non-critical components or modules only when needed.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to SvelteKit

*   **Form Actions**: Use SvelteKit's form actions for handling form submissions on the server.
*   **Load Functions**: Utilize `load` functions in `+page.server.ts` and `+layout.server.ts` for data fetching.
*   **Stores for State Management**: Employ Svelte stores for managing application state.
*   **Server-Side Rendering (SSR)**: Leverage SSR for improved SEO and initial load performance.
*   **Progressive Enhancement**: Design applications to work even with JavaScript disabled.

### 2.2. Recommended Approaches for Common Tasks

*   **Data Fetching**: Use `fetch` API or libraries like `axios` within `load` functions.
*   **Form Handling**: Use SvelteKit's form actions for server-side validation and processing.
*   **Authentication**: Implement authentication using libraries like `lucia-auth` or integrate with OAuth providers.
*   **Authorization**: Implement role-based access control using stores or server-side checks.
*   **Error Handling**: Use `try...catch` blocks and SvelteKit's `handleError` hook for global error handling.

### 2.3. Anti-patterns and Code Smells to Avoid

*   **Over-reliance on Global State**: Avoid using global stores for component-specific state.
*   **Direct DOM Manipulation**: Avoid directly manipulating the DOM; use Svelte's reactivity system instead.
*   **Large, Unmaintainable Components**: Break down large components into smaller, reusable ones.
*   **Ignoring Accessibility**: Ensure components are accessible to users with disabilities (ARIA attributes, semantic HTML).
*   **Unnecessary Re-renders**: Optimize components to avoid unnecessary re-renders.

### 2.4. State Management Best Practices

*   **Use Svelte Stores**: Svelte stores are the recommended way to manage application state.
*   **Divide stores according to feature**: Split stores based on logical domain or feature (e.g. UserStore, CartStore).
*   **Readable and Writable Stores**: Use readable stores for derived state and writable stores for mutable state.
*   **Custom Stores**: Create custom stores for complex state management logic.

### 2.5. Error Handling Patterns

*   **`try...catch` Blocks**: Use `try...catch` blocks to handle errors within components and functions.
*   **`handleError` Hook**: Implement the `handleError` hook in `src/hooks.server.ts` for global error handling.
*   **Error Boundaries**: Use error boundaries to catch errors in component trees and display fallback UI.
*   **Logging**: Log errors to a server for monitoring and debugging.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Lazy Loading**: Load non-critical resources (images, components) only when needed.
*   **Code Splitting**: Split code into smaller chunks to reduce initial load time.
*   **Image Optimization**: Optimize images using tools like `squoosh` or `tinypng`.
*   **Server-Side Rendering (SSR)**: Use SSR for improved initial load performance and SEO.
*   **Caching**: Implement caching strategies for data and assets.
*   **Preload critical assets**: use `<link rel='preload'>` to fetch critical assets earlier

### 3.2. Memory Management

*   **Avoid Memory Leaks**: Properly clean up resources when components unmount.
*   **Use WeakRefs**: Use `WeakRefs` for managing objects that might be garbage collected.
*   **Minimize Object Creation**: Avoid creating unnecessary objects and arrays.
*   **Properly unsubscrive from stores:** Properly unsubscribe to avoid re-renders when the component is unmounted.

### 3.3. Rendering Optimization

*   **Tracked values**: Use the `$:` syntax to track dependencies of reactive statements.
*   **Svelte Compiler**: Utilize Svelte's compiler to optimize component rendering.
*   **Virtual DOM**: Svelte doesn't use a virtual DOM, so updates are generally very efficient.
*   **Avoid Unnecessary Re-renders**: Use `{#each}` blocks with keys to optimize list rendering.
*   **Debouncing and Throttling**: Use debouncing and throttling to limit the frequency of updates.

### 3.4. Bundle Size Optimization

*   **Tree Shaking**: Ensure unused code is removed during the build process.
*   **Minification**: Minify JavaScript and CSS files to reduce bundle size.
*   **Compression**: Compress assets using gzip or Brotli.
*   **Dependency Analysis**: Analyze dependencies to identify and remove unused libraries.

### 3.5. Lazy Loading Strategies

*   **Intersection Observer**: Use the Intersection Observer API to detect when elements are visible and load them lazily.
*   **Dynamic Imports**: Use dynamic imports to load components and modules on demand.
*   **Placeholder UI**: Display a placeholder UI while resources are loading.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS)**: Sanitize user input to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF)**: Use CSRF tokens to protect against CSRF attacks.
*   **SQL Injection**: Use parameterized queries or ORMs to prevent SQL injection.
*   **Authentication and Authorization**: Implement secure authentication and authorization mechanisms.
*   **Insecure Direct Object References (IDOR)**: Implement proper access controls to prevent unauthorized access to resources.

### 4.2. Input Validation

*   **Server-Side Validation**: Always validate user input on the server.
*   **Client-Side Validation**: Use client-side validation for immediate feedback, but never rely on it solely.
*   **Escape User Input**: Escape user input before rendering it in the DOM.
*   **Use Validation Libraries**: Use libraries like `zod` or `yup` for input validation.

### 4.3. Authentication and Authorization Patterns

*   **JWT (JSON Web Tokens)**: Use JWTs for authentication and authorization.
*   **OAuth 2.0**: Integrate with OAuth providers like Google and Facebook for social login.
*   **Role-Based Access Control (RBAC)**: Implement RBAC to control access to resources based on user roles.
*   **Multi-Factor Authentication (MFA)**: Implement MFA for enhanced security.

### 4.4. Data Protection Strategies

*   **Encryption**: Encrypt sensitive data at rest and in transit.
*   **Hashing**: Hash passwords using strong hashing algorithms like bcrypt or Argon2.
*   **Data Masking**: Mask sensitive data in logs and reports.
*   **Regular Security Audits**: Conduct regular security audits to identify and address vulnerabilities.

### 4.5. Secure API Communication

*   **HTTPS**: Use HTTPS for all API communication.
*   **API Keys**: Protect API keys and secrets.
*   **Rate Limiting**: Implement rate limiting to prevent abuse.
*   **Input Validation**: Always validate input to your APIs.
*   **Output Encoding**: Always encode output from your API to prevent injection attacks.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   **Test Individual Components**: Write unit tests for individual components in isolation.
*   **Test Utility Functions**: Write unit tests for utility functions and modules.
*   **Use Mock Data**: Use mock data for testing components and functions.
*   **Test Edge Cases**: Test edge cases and error conditions.
*   **Utilize Svelte Testing Library**: Employ `svelte-testing-library` for testing Svelte components.

### 5.2. Integration Testing

*   **Test Component Interactions**: Write integration tests to ensure components interact correctly.
*   **Test Data Flow**: Test the flow of data between components and modules.
*   **Test API Integrations**: Test integrations with APIs and external services.

### 5.3. End-to-End Testing

*   **Simulate User Interactions**: Write end-to-end tests to simulate user interactions.
*   **Test Full Workflows**: Test complete user workflows from start to finish.
*   **Use Playwright or Cypress**: Use tools like Playwright or Cypress for end-to-end testing.

### 5.4. Test Organization

*   **Mirror Source Structure**: Organize tests in a directory structure that mirrors the source code.
*   **Use Descriptive Names**: Use descriptive names for test files and functions.
*   **Separate Test Environments**: Use separate test environments for unit, integration, and end-to-end tests.

### 5.5. Mocking and Stubbing

*   **Mock API Calls**: Mock API calls to isolate components during testing.
*   **Stub External Dependencies**: Stub external dependencies to control their behavior.
*   **Use Mocking Libraries**: Use libraries like `jest.mock` or `sinon` for mocking and stubbing.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

*   **Incorrectly Using Reactive Statements**: Not understanding how Svelte's reactivity system works.
*   **Ignoring Accessibility**: Neglecting accessibility considerations when building components.
*   **Over-complicating State Management**: Using complex state management solutions when simpler ones would suffice.
*   **Not Handling Errors Properly**: Ignoring errors or not providing informative error messages.
*   **Premature Optimization**: Optimizing code before it is necessary.

### 6.2. Edge Cases to Be Aware Of

*   **Server-Side Rendering Differences**: Be aware of differences between server-side and client-side rendering.
*   **Browser Compatibility**: Test applications in different browsers to ensure compatibility.
*   **Network Latency**: Design applications to handle network latency and unreliable connections.
*   **Memory Constraints**: Consider memory constraints when building large applications.

### 6.3. Version-Specific Issues

*   **Breaking Changes**: Be aware of breaking changes in new versions of SvelteKit.
*   **Deprecated Features**: Avoid using deprecated features.
*   **Upgrade Guides**: Follow upgrade guides when migrating to new versions of SvelteKit.

### 6.4. Compatibility Concerns

*   **Third-Party Libraries**: Ensure third-party libraries are compatible with SvelteKit.
*   **Browser Support**: Check browser support for new features and APIs.
*   **Node.js Version**: Ensure the Node.js version is compatible with SvelteKit.

### 6.5. Debugging Strategies

*   **Browser Developer Tools**: Use browser developer tools for debugging JavaScript and CSS.
*   **Svelte Devtools**: Use the Svelte Devtools extension for inspecting Svelte components.
*   **Logging**: Use `console.log` statements for debugging.
*   **Debuggers**: Use debuggers like VS Code's debugger for step-by-step debugging.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **VS Code**: VS Code is a popular code editor with excellent Svelte and TypeScript support.
*   **Svelte Devtools**: The Svelte Devtools browser extension allows you to inspect Svelte components.
*   **ESLint**: ESLint is a linter that helps enforce code style and best practices.
*   **Prettier**: Prettier is a code formatter that automatically formats code to a consistent style.
*   **TypeScript**: Using typescript is recommended for type safety.

### 7.2. Build Configuration

*   **`svelte.config.js`**: Configure SvelteKit using the `svelte.config.js` file.
*   **Vite**: SvelteKit uses Vite as its build tool.
*   **Environment Variables**: Use environment variables for configuration settings.
*   **Adapters**: Use adapters to deploy SvelteKit applications to different environments.

### 7.3. Linting and Formatting

*   **ESLint**: Configure ESLint to enforce code style and best practices.
*   **Prettier**: Configure Prettier to automatically format code.
*   **Husky**: Use Husky to run linters and formatters before committing code.
*   **Lint Staged**: Use lint-staged to run linters and formatters only on staged files.

### 7.4. Deployment Best Practices

*   **Choose an Adapter**: Use an adapter like `@sveltejs/adapter-node` or `@sveltejs/adapter-static` to deploy the application.
*   **Configure Environment Variables**: Configure environment variables for production environments.
*   **Set Up HTTPS**: Set up HTTPS for secure communication.
*   **Monitor the Application**: Monitor the application for errors and performance issues.

### 7.5. CI/CD Integration

*   **Use a CI/CD Pipeline**: Use a CI/CD pipeline to automate the build, test, and deployment process.
*   **Automated Tests**: Run automated tests in the CI/CD pipeline.
*   **Automated Deployments**: Automate deployments to production environments.
*   **Use tools like GitHub Actions, GitLab CI, or CircleCI**