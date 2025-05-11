---
description: This rule provides comprehensive guidelines for Angular development, covering coding standards, best practices, performance optimization, security considerations, and testing approaches to ensure maintainable, scalable, and high-performing applications.
globs: *.ts,*.html,*.scss,*.css
---
# Angular Best Practices and Coding Standards

This document outlines comprehensive best practices and coding standards for developing Angular applications. Adhering to these guidelines will promote maintainability, scalability, performance, and security.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

- **Feature-based Modules:** Organize your application into feature modules, where each module encapsulates a specific feature or functionality.
- **Shared Module:** Create a `shared` module for commonly used components, directives, pipes, and services.
- **Core Module:** Create a `core` module for application-wide services and singleton instances (e.g., authentication service, configuration service).
- **Lazy-loaded Modules:** Group related features into lazy-loaded modules to improve initial load time.
- **Directory structure example:**

  
  src/
  ├── app/
  │   ├── core/
  │   │   ├── core.module.ts
  │   │   └── auth.service.ts
  │   ├── shared/
  │   │   ├── shared.module.ts
  │   │   └── components/
  │   │       └── ...
  │   ├── features/
  │   │   ├── dashboard/
  │   │   │   ├── dashboard.module.ts
  │   │   │   ├── dashboard.component.ts
  │   │   │   └── dashboard.component.html
  │   │   ├── user-management/
  │   │   │   ├── user-management.module.ts
  │   │   │   └── ...
  │   │   └── ...
  │   ├── app.component.ts
  │   └── app.module.ts
  └── ...
  

### 1.2. File Naming Conventions

- **Consistent Naming:** Use a consistent naming pattern for all files and symbols, following the `feature.type.ts` convention (e.g., `user.component.ts`, `user.service.ts`).
- **Type Abbreviations:** Use abbreviations for file types (e.g., `component.ts` -> `.component.ts`, `service.ts` -> `.service.ts`, `module.ts` -> `.module.ts`).
- **Descriptive Names:** Use descriptive names that clearly indicate the purpose of the file.

### 1.3. Module Organization

- **Single Responsibility:** Each module should have a single responsibility.
- **Declarations:** Declare components, directives, and pipes within the appropriate module.
- **Imports:** Import only the necessary modules in each module.
- **Exports:** Export components, directives, and pipes that need to be used by other modules.
- **forRoot() pattern:** Use the `forRoot()` pattern for modules that provide singleton services to ensure they are only instantiated once in the application.

### 1.4. Component Architecture

- **Smart vs. Dumb Components:** Distinguish between smart (container) components that handle data and logic, and dumb (presentational) components that focus on rendering UI.
- **Component Reusability:** Design components for reusability.
- **Input and Output Properties:** Use `@Input()` and `@Output()` properties to pass data and events between components.
- **Change Detection:** Be mindful of change detection strategies (e.g., `OnPush`) to optimize rendering performance.

### 1.5. Code Splitting Strategies

- **Lazy Loading:** Implement lazy loading for modules to reduce the initial bundle size and improve load time.
- **Route-based Code Splitting:** Split your application into modules based on routes.
- **Feature-based Code Splitting:** Split your application into modules based on features.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to Angular

- **Dependency Injection (DI):** Use DI to manage dependencies between components and services.
- **Observables:** Use RxJS Observables for handling asynchronous data streams.
- **Services:** Use services for encapsulating reusable business logic.
- **Pipes:** Use pipes for transforming data in templates.
- **Directives:** Use directives for manipulating the DOM.

### 2.2. Recommended Approaches for Common Tasks

- **Data Binding:** Use data binding (`{{ }}`, `[]`, `()`) to synchronize data between the component and the template.
- **Event Handling:** Use event binding (`()`) to handle user interactions.
- **Form Handling:** Use reactive forms or template-driven forms to manage user input.
- **HTTP Requests:** Use the `HttpClient` to make HTTP requests to backend APIs.

### 2.3. Anti-patterns and Code Smells to Avoid

- **Deeply Nested Components:** Avoid deeply nested component hierarchies, which can make it difficult to manage state and events.
- **Large Components:** Break large components into smaller, reusable components.
- **Logic in Templates:** Avoid complex logic in templates; move it to the component class.
- **Direct DOM Manipulation:** Avoid direct DOM manipulation; use Angular's data binding and directives instead.
- **Unnecessary Subscriptions:** Unsubscribe from Observables to prevent memory leaks.  Use the `async` pipe in templates to automatically handle subscriptions and unsubscriptions.

### 2.4. State Management Best Practices

- **Choose a State Management Library:** Consider using a state management library like NgRx, Akita, or MobX for complex applications.
- **Centralized State:** Store application state in a central store.
- **Immutability:** Treat state as immutable to simplify change detection and debugging.
- **Actions and Reducers:** Use actions to describe state changes and reducers to update the state.
- **Selectors:** Use selectors to efficiently retrieve data from the store.

### 2.5. Error Handling Patterns

- **Centralized Error Handling:** Implement a centralized error handling mechanism using an `ErrorHandler`.
- **Error Interceptors:** Use HTTP interceptors to handle errors from backend APIs.
- **User-Friendly Error Messages:** Display user-friendly error messages to the user.
- **Logging:** Log errors to a central logging service for debugging and monitoring.
- **Retry mechanism:** Implement retry mechanism for failed requests using RxJS retry operators.

## 3. Performance Considerations

### 3.1. Optimization Techniques

- **Change Detection Optimization:** Use `OnPush` change detection strategy for components that only depend on their input properties.
- **TrackBy Function:** Use the `trackBy` function with `ngFor` to optimize rendering of lists.
- **Virtualization:** Use virtual scrolling for large lists to improve performance.
- **Ahead-of-Time (AOT) Compilation:** Use AOT compilation to compile templates during the build process, improving startup time.
- **Minification and Uglification:** Minify and uglify your code to reduce bundle size.

### 3.2. Memory Management

- **Unsubscribe from Observables:** Unsubscribe from Observables when they are no longer needed to prevent memory leaks.
- **Avoid Circular Dependencies:** Avoid circular dependencies, which can lead to memory leaks.
- **Detach Event Listeners:** Detach event listeners when they are no longer needed.

### 3.3. Rendering Optimization

- **Minimize DOM Manipulation:** Minimize DOM manipulation, as it can be expensive.
- **Use CSS Transforms:** Use CSS transforms instead of modifying layout properties.
- **Debouncing and Throttling:** Use debouncing and throttling to reduce the frequency of event handlers.

### 3.4. Bundle Size Optimization

- **Lazy Loading:** Implement lazy loading for modules to reduce the initial bundle size.
- **Tree Shaking:** Use tree shaking to remove unused code from your bundle.
- **Code Splitting:** Split your application into smaller bundles to improve load time.
- **Image Optimization:** Optimize images to reduce their file size.
- **Vendor Libraries:** Use only necessary vendor libraries, and ensure they are up-to-date for optimized versions.

### 3.5. Lazy Loading Strategies

- **Route-based Lazy Loading:** Load modules when navigating to a specific route.
- **Feature-based Lazy Loading:** Load modules based on feature requirements.
- **Preloading:** Preload modules in the background to improve perceived performance.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

- **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user input and avoiding the use of `innerHTML`.
- **Cross-Site Request Forgery (CSRF):** Prevent CSRF attacks by using anti-CSRF tokens.
- **Injection Attacks:** Prevent injection attacks by validating user input and using parameterized queries.

### 4.2. Input Validation

- **Client-Side Validation:** Perform client-side validation to provide immediate feedback to the user.
- **Server-Side Validation:** Perform server-side validation to ensure data integrity.
- **Whitelist Validation:** Use whitelist validation to allow only specific characters or patterns.

### 4.3. Authentication and Authorization Patterns

- **Authentication:** Implement authentication to verify the user's identity.
- **Authorization:** Implement authorization to control access to resources based on the user's roles and permissions.
- **JSON Web Tokens (JWT):** Use JWTs for secure authentication and authorization.
- **OAuth 2.0:** Use OAuth 2.0 for delegating authorization to third-party applications.

### 4.4. Data Protection Strategies

- **Encryption:** Encrypt sensitive data at rest and in transit.
- **Hashing:** Hash passwords before storing them in the database.
- **Data Masking:** Mask sensitive data to prevent unauthorized access.

### 4.5. Secure API Communication

- **HTTPS:** Use HTTPS for secure communication between the client and the server.
- **API Keys:** Use API keys to authenticate API requests.
- **Rate Limiting:** Implement rate limiting to prevent abuse of your API.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

- **Test Driven Development (TDD):** Write tests before writing code.
- **Component Testing:** Test components in isolation using mock data.
- **Service Testing:** Test services in isolation using dependency injection.
- **Pipe Testing:** Test pipes to ensure they transform data correctly.

### 5.2. Integration Testing

- **Module Testing:** Test modules to ensure they integrate correctly.
- **Component Interaction Testing:** Test the interaction between components.
- **HTTP Testing:** Test HTTP requests and responses.

### 5.3. End-to-End Testing

- **User Interface (UI) Testing:** Test the UI to ensure it behaves as expected.
- **Workflow Testing:** Test complete workflows to ensure they function correctly.
- **Cross-Browser Testing:** Test your application in different browsers.

### 5.4. Test Organization

- **Test Directory Structure:** Mirror your source code directory structure in your test directory.
- **Test File Naming:** Use consistent naming conventions for test files (e.g., `user.component.spec.ts`).
- **Test Suites:** Group related tests into test suites.

### 5.5. Mocking and Stubbing

- **Mock Dependencies:** Mock dependencies to isolate units of code for testing.
- **Stub HTTP Requests:** Stub HTTP requests to avoid making actual API calls during testing.
- **Use Mocking Libraries:** Use mocking libraries like Jasmine or Jest to simplify mocking and stubbing.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

- **Forgetting to Unsubscribe from Observables:** Always unsubscribe from Observables to prevent memory leaks.
- **Using `any` Type:** Avoid using the `any` type, as it bypasses type checking.
- **Not Handling Errors:** Handle errors gracefully to prevent unexpected behavior.
- **Ignoring Performance:** Optimize your code for performance from the beginning.

### 6.2. Edge Cases to Be Aware Of

- **Empty Arrays:** Handle empty arrays correctly.
- **Null Values:** Handle null values correctly.
- **Unexpected Input:** Validate user input to prevent unexpected behavior.

### 6.3. Version-Specific Issues

- **Breaking Changes:** Be aware of breaking changes in new versions of Angular and third-party libraries.
- **Deprecated APIs:** Avoid using deprecated APIs.
- **Migration Guides:** Follow migration guides when upgrading to new versions of Angular.

### 6.4. Compatibility Concerns

- **Browser Compatibility:** Test your application in different browsers to ensure compatibility.
- **Device Compatibility:** Test your application on different devices to ensure compatibility.

### 6.5. Debugging Strategies

- **Browser Developer Tools:** Use browser developer tools to debug your code.
- **Logging:** Use logging to track the flow of execution and identify errors.
- **Debugging Tools:** Use debugging tools like VS Code's debugger to step through your code.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

- **VS Code:** Use VS Code as your code editor.
- **Angular CLI:** Use the Angular CLI for scaffolding, building, and deploying your application.
- **Chrome Developer Tools:** Use Chrome Developer Tools for debugging and performance profiling.

### 7.2. Build Configuration

- **Angular CLI Configuration:** Configure your Angular CLI settings in the `angular.json` file.
- **Environment Variables:** Use environment variables to configure your application for different environments.
- **Build Optimization:** Configure your build process to optimize your code for production.

### 7.3. Linting and Formatting

- **ESLint:** Use ESLint to enforce coding standards.
- **Prettier:** Use Prettier to format your code automatically.
- **Husky:** Use Husky to run linters and formatters before committing code.

### 7.4. Deployment Best Practices

- **Continuous Integration/Continuous Deployment (CI/CD):** Use a CI/CD pipeline to automate the build, test, and deployment process.
- **Deployment Environments:** Use different deployment environments for development, testing, and production.
- **CDN:** Use a content delivery network (CDN) to serve static assets.

### 7.5. CI/CD Integration

- **Jenkins:** Use Jenkins for CI/CD.
- **GitHub Actions:** Use GitHub Actions for CI/CD.
- **Azure DevOps:** Use Azure DevOps for CI/CD.
- **GitLab CI:** Use GitLab CI for CI/CD.

By following these best practices and coding standards, you can build robust, maintainable, and scalable Angular applications.