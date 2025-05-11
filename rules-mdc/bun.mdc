---
description: This rule provides comprehensive guidance on Bun library coding standards, best practices, and common patterns. It includes performance optimization, security considerations, and testing strategies.
globs: *.js,*.ts,*.jsx,*.tsx,*.bun
---
# Bun Library Best Practices

This document outlines the recommended coding standards, best practices, and patterns for developing applications using the Bun library. Following these guidelines ensures maintainability, performance, security, and overall code quality.

## 1. Code Organization and Structure

### 1.1. Directory Structure

A well-defined directory structure is crucial for maintainability. Consider the following structure as a starting point:


project-root/
├── src/                  # Source code
│   ├── components/       # Reusable UI components (if applicable)
│   ├── services/         # Business logic and API interactions
│   ├── utils/            # Utility functions
│   ├── types/            # TypeScript type definitions
│   ├── routes/           # API route handlers
│   ├── middleware/       # Middleware functions
│   ├── models/           # Data models
│   ├── config/           # Configuration files
│   ├── index.ts          # Entry point for the application
├── tests/                # Unit and integration tests
├── public/               # Static assets (e.g., images, CSS)
├── .env                  # Environment variables
├── bun.lockb             # Lockfile for dependencies
├── package.json          # Project metadata and dependencies
├── tsconfig.json         # TypeScript configuration
├── README.md             # Project documentation


*   **src/**: Contains the main source code of the application.
*   **components/**: Houses reusable UI components (if building a web application).
*   **services/**: Encapsulates business logic and interactions with external APIs.
*   **utils/**: Contains utility functions used throughout the application.
*   **types/**: Stores TypeScript type definitions.
*   **routes/**: Defines API route handlers using Bun's built-in HTTP server.
*   **middleware/**: Includes middleware functions for request processing.
*   **models/**: Defines data models used in the application.
*   **config/**: Contains configuration files for different environments.
*   **tests/**: Holds unit and integration tests.
*   **public/**: Stores static assets like images and CSS files.
*   **.env**: Stores environment variables.
*   **bun.lockb**:  Lockfile ensuring consistent dependency versions.
*   **package.json**: Defines project metadata and dependencies.
*   **tsconfig.json**: Configures TypeScript compiler options.
*   **README.md**: Provides project documentation and instructions.

### 1.2. File Naming Conventions

*   Use descriptive and consistent file names.
*   Prefer camelCase for JavaScript/TypeScript files (e.g., `userService.ts`, `apiHelper.js`).
*   Use kebab-case for component directories (e.g., `user-profile`).
*   For React components, use PascalCase for the filename (e.g., `UserProfile.tsx`).

### 1.3. Module Organization

*   Group related functionality into modules.
*   Use clear and concise module names.
*   Export only the necessary functions and classes from each module.
*   Favor explicit imports over global variables.

### 1.4. Component Architecture (if applicable)

*   If building a web application, adopt a component-based architecture (e.g., using React, SolidJS).
*   Divide the UI into small, reusable components.
*   Follow the Single Responsibility Principle (SRP) for each component.
*   Use a consistent component structure (e.g., a folder for each component containing the component file, styles, and tests).

### 1.5. Code Splitting Strategies

*   Use dynamic imports (`import()`) to split code into smaller chunks.
*   Load only the necessary code for each route or component.
*   Consider using a library like `esbuild` (Bun's underlying bundler) or `parcel` to automate code splitting.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Singleton**: Use for managing global resources (e.g., database connections, configuration).
*   **Factory**: Use for creating objects without specifying their concrete classes.
*   **Observer**: Use for implementing event-driven systems.
*   **Middleware**: Use to handle requests and responses in a centralized manner.
*   **Dependency Injection**: Use to decouple components and improve testability.

### 2.2. Recommended Approaches for Common Tasks

*   **API Request Handling**: Utilize `fetch` API provided by Bun for making HTTP requests.
*   **File System Operations**: Use `Bun.file()` and other built-in functions for reading and writing files.
*   **Process Management**: Leverage `Bun.spawn()` or `Bun.serve()` to manage child processes and servers.
*   **Environment Variable Access**: Use `Bun.env` or `process.env` to access environment variables.
*   **Logging**: Implement logging using `console.log` or a dedicated logging library like `pino`.

### 2.3. Anti-patterns and Code Smells

*   **Global State**: Avoid using global variables for application state. Use state management solutions instead.
*   **Long Functions**: Break down long functions into smaller, more manageable functions.
*   **Duplicated Code**: Extract common logic into reusable functions or modules.
*   **Magic Numbers**: Use named constants instead of hardcoded values.
*   **Ignoring Errors**: Always handle errors properly using try-catch blocks or error handling middleware.

### 2.4. State Management Best Practices

*   If your application requires complex state management, consider using a library like Zustand, Valtio, or Jotai.
*   Choose a state management solution that fits the complexity of your application.
*   Keep state updates predictable and consistent.
*   Avoid mutating state directly.

### 2.5. Error Handling Patterns

*   Use try-catch blocks to handle synchronous errors.
*   Use `async/await` with try-catch blocks for asynchronous errors.
*   Implement error handling middleware to catch unhandled exceptions.
*   Log errors with relevant information (e.g., stack trace, request details).
*   Provide informative error messages to the user.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Minimize Dependencies**:  Reduce the number of dependencies to decrease bundle size and install time.
*   **Code Splitting**:  Split code into smaller chunks that can be loaded on demand.
*   **Tree Shaking**:  Remove unused code during the build process.
*   **Caching**:  Cache frequently accessed data to reduce latency.
*   **Compression**:  Compress responses using gzip or Brotli to reduce network traffic.
*   **Efficient Algorithms**: Choose the most efficient algorithms for your tasks.

### 3.2. Memory Management

*   Avoid memory leaks by properly releasing resources.
*   Use weak references to avoid circular dependencies.
*   Monitor memory usage using tools like `bun --inspect`.
*   Be mindful of large data structures and use streams when appropriate.

### 3.3. Rendering Optimization (if applicable)

*   Use virtualization for large lists or tables.
*   Optimize images and other assets.
*   Use memoization to avoid unnecessary re-renders.
*   Profile rendering performance using browser developer tools.

### 3.4. Bundle Size Optimization

*   Use a bundler like `esbuild` to minimize bundle size.
*   Remove unused code and dependencies.
*   Use code splitting to load only the necessary code.
*   Consider using a smaller alternative to large libraries.

### 3.5. Lazy Loading Strategies

*   Use dynamic imports (`import()`) to load modules on demand.
*   Implement lazy loading for images and other assets.
*   Use a library like `react-lazyload` (if using React) to simplify lazy loading.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS)**: Sanitize user input to prevent malicious scripts from being injected into the page.
*   **Cross-Site Request Forgery (CSRF)**: Use CSRF tokens to prevent attackers from forging requests on behalf of authenticated users.
*   **SQL Injection**: Use parameterized queries or an ORM to prevent attackers from injecting malicious SQL code.
*   **Authentication and Authorization**: Implement robust authentication and authorization mechanisms to protect sensitive data.
*   **Denial of Service (DoS)**: Implement rate limiting and other measures to prevent attackers from overwhelming the server.

### 4.2. Input Validation

*   Validate all user input on both the client and server sides.
*   Use a validation library like `zod` or `yup` to define validation schemas.
*   Sanitize user input to remove potentially harmful characters.
*   Escape user input when displaying it on the page.

### 4.3. Authentication and Authorization Patterns

*   Use a secure authentication protocol like OAuth 2.0 or OpenID Connect.
*   Store passwords securely using a hashing algorithm like bcrypt or Argon2.
*   Implement role-based access control (RBAC) to restrict access to sensitive resources.
*   Use JSON Web Tokens (JWT) for authentication and authorization.

### 4.4. Data Protection Strategies

*   Encrypt sensitive data at rest and in transit.
*   Use HTTPS to encrypt communication between the client and server.
*   Store encryption keys securely using a key management system.
*   Regularly back up data to prevent data loss.

### 4.5. Secure API Communication

*   Use HTTPS for all API communication.
*   Implement API authentication using API keys or JWTs.
*   Rate limit API requests to prevent abuse.
*   Validate API requests and responses.
*   Use a firewall to protect the API from unauthorized access.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   Write unit tests for individual functions and classes.
*   Use a testing framework like Jest or Bun's built-in test runner.
*   Aim for high code coverage.
*   Use mocks and stubs to isolate units of code.
*   Test edge cases and error conditions.

### 5.2. Integration Testing

*   Write integration tests to verify the interaction between different modules.
*   Test the integration of the application with external APIs.
*   Use a testing framework like Jest or Mocha.
*   Use a testing database or mock API to isolate the tests.

### 5.3. End-to-End Testing

*   Write end-to-end tests to verify the entire application flow.
*   Use a testing framework like Playwright or Cypress.
*   Run tests in a browser environment.
*   Test the application from the user's perspective.

### 5.4. Test Organization

*   Create a separate `tests` directory for test files.
*   Organize test files in a way that mirrors the source code structure.
*   Use descriptive test names.
*   Follow a consistent testing style.

### 5.5. Mocking and Stubbing

*   Use mocks and stubs to isolate units of code during testing.
*   Use a mocking library like `jest.mock()` or `sinon`.
*   Mock external dependencies to avoid relying on external services.
*   Stub functions to control their behavior during testing.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

*   **Not Using Strict Mode**: Always use strict mode (`'use strict'`) to catch common coding errors.
*   **Ignoring Error Handling**: Always handle errors properly using try-catch blocks or error handling middleware.
*   **Leaking Global Variables**: Avoid creating global variables by using `let` or `const` to declare variables.
*   **Not Understanding Asynchronous JavaScript**:  Understand how asynchronous JavaScript works to avoid common pitfalls like callback hell.
*   **Over-Engineering**: Keep the code simple and avoid unnecessary complexity.

### 6.2. Edge Cases to Be Aware Of

*   **Handling Null and Undefined Values**:  Check for null and undefined values before using them to avoid errors.
*   **Integer Overflow**: Be aware of integer overflow and underflow when performing arithmetic operations.
*   **Unicode Support**:  Properly handle Unicode characters to avoid encoding issues.
*   **Time Zone Handling**:  Handle time zones correctly to avoid date and time discrepancies.

### 6.3. Version-Specific Issues

*   Be aware of breaking changes in new versions of Bun.
*   Test the application with different versions of Bun to ensure compatibility.
*   Use a version manager like `bunx` to manage different Bun versions.

### 6.4. Compatibility Concerns

*   Ensure the application is compatible with different operating systems and browsers.
*   Use polyfills to support older browsers.
*   Test the application on different devices to ensure responsiveness.

### 6.5. Debugging Strategies

*   Use the `bun --inspect` flag to debug the application using Chrome DevTools.
*   Use `console.log` statements to print debugging information.
*   Use a debugger like VS Code's built-in debugger.
*   Use a logging library to log errors and other important events.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **VS Code**: A popular code editor with excellent TypeScript support.
*   **ESLint**: A linter for identifying and fixing code style issues.
*   **Prettier**: A code formatter for automatically formatting code.
*   **Jest**: A testing framework for unit and integration testing.
*   **Playwright/Cypress**: A testing framework for end-to-end testing.
*   **Postman/Insomnia**: API client for testing API endpoints.

### 7.2. Build Configuration

*   Use a build tool like `esbuild` or `webpack` to bundle the application.
*   Configure the build tool to optimize the bundle size and performance.
*   Use environment variables to configure the build process for different environments.

### 7.3. Linting and Formatting

*   Use ESLint to enforce consistent coding style.
*   Use Prettier to automatically format code.
*   Integrate ESLint and Prettier into the development workflow using VS Code extensions or command-line tools.
*   Configure ESLint and Prettier to follow the project's coding style guidelines.

### 7.4. Deployment Best Practices

*   Use a process manager like `pm2` or `systemd` to manage the application in production.
*   Deploy the application to a cloud platform like DigitalOcean, Vercel, or Render.
*   Use a CI/CD pipeline to automate the deployment process.
*   Monitor the application's performance and health using monitoring tools.

### 7.5. CI/CD Integration

*   Use a CI/CD platform like GitHub Actions, GitLab CI, or CircleCI to automate the build, test, and deployment process.
*   Configure the CI/CD pipeline to run tests, lint code, and build the application on every commit.
*   Use environment variables to configure the CI/CD pipeline for different environments.
*   Deploy the application to a staging environment before deploying it to production.

By following these best practices, developers can build high-quality, performant, and secure applications using the Bun library. Remember to adapt these guidelines to the specific needs of your project.