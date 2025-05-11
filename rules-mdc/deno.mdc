---
description: This rule file provides comprehensive guidelines for Deno development, covering best practices for code organization, security, performance, testing, and documentation. Adhering to these standards ensures maintainable, efficient, and secure Deno applications.
globs: *.ts,*.tsx,*.js,*.jsx,*.md
---
- **Use TypeScript**: Always prefer TypeScript for its strong typing, which enhances code quality and maintainability. Provide clear type annotations and use interfaces for better structure.
- **Documentation with JSDoc**: Utilize JSDoc for documenting all exported symbols. Include concise summaries, type information, and examples. Keep documentation up-to-date with code changes.
- **Consistent Naming Conventions**: Follow established naming conventions such as camelCase for variables and functions, PascalCase for classes, and UPPER_SNAKE_CASE for constants. This consistency aids readability and maintainability.
- **Limit Function Parameters**: Functions should ideally take no more than two parameters, using options objects for additional parameters. This makes functions easier to use and test.

## 1. Code Organization and Structure

- **Directory Structure Best Practices**:
    - `src/`: Contains the main source code of the application.
    - `src/components/`: Reusable UI components (if applicable).
    - `src/services/`: Business logic and API interactions.
    - `src/utils/`: Utility functions and helpers.
    - `tests/`: Unit and integration tests.
    - `mod.ts`: The main entry point for the module or application.
    - `deps.ts`: Centralized dependency management file (more details below).
    - `dev_deps.ts`: Centralized development dependency management file (more details below).
    - `deno.json`: Deno configuration file.
- **File Naming Conventions**:
    - Use descriptive names with `.ts`, `.tsx`, `.js`, or `.jsx` extensions.
    - Component files: `ComponentName.tsx` or `component-name.tsx`
    - Service files: `service-name.ts`
    - Utility files: `utils.ts` or `string_utils.ts`
    - Test files: `file_name_test.ts`
- **Module Organization Best Practices**:
    - Favor small, focused modules with clear responsibilities.
    - Avoid circular dependencies.
    - Use `deps.ts` to manage dependencies centrally. This file exports all external dependencies used in the project, allowing for easy version management and updates. Example:

    typescript
    // deps.ts
    export * as log from "https://deno.land/std@0.224.0/log/mod.ts";
    export { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
    export { serve } from "https://deno.land/std@0.224.0/http/server.ts";
    
    - Use `dev_deps.ts` to manage development dependencies centrally. Example:

    typescript
    // dev_deps.ts
    export {
      assert,
      assertEquals,
      assertExists,
      assertNotEquals,
      assertRejects,
      assertStringIncludes,
    } from "https://deno.land/std@0.224.0/assert/mod.ts";
    export {
      FakeTime,
    } from "https://deno.land/std@0.224.0/testing/time.ts";
    

    - Import dependencies using the `deps.ts` file in other modules:

    typescript
    // my_module.ts
    import { log, assertEquals } from "./deps.ts";

    log.info("Hello, Deno!");
    
- **Component Architecture Recommendations**:
    - Follow a component-based architecture for UI development (if applicable).
    - Each component should be self-contained and reusable.
    - Consider using a state management library like `nano-states` or `unstate` for complex applications.
- **Code Splitting Strategies**:
    - Use dynamic imports (`import()`) to load modules on demand.
    - Split code based on routes or features.
    - Leverage Deno's built-in module caching for efficient loading.
    - Use `deno bundle` to create optimized bundles for deployment, but be mindful of over-bundling.

## 2. Common Patterns and Anti-patterns

- **Design Patterns**: 
    - **Module Pattern**: Encapsulating code within a module to control access and prevent global scope pollution.
    - **Factory Pattern**: Creating objects without specifying the exact class to instantiate.
    - **Observer Pattern**: Defining a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
    - **Dependency Injection**:  Making components loosely coupled.  Useful with Deno's module system because you can easily swap out dependencies in tests without relying on complex mocking frameworks.
- **Recommended Approaches for Common Tasks**:
    - **HTTP Server**: Use Deno's built-in `Deno.serve` for creating HTTP servers.
    - **File System Access**: Use `Deno.readTextFile`, `Deno.writeTextFile`, etc., for file system operations.
    - **Environment Variables**: Access environment variables using `Deno.env.get()`.
    - **Process Management**: Use `Deno.run` for executing external commands.
- **Anti-patterns and Code Smells**:
    - **Global Scope Pollution**: Avoid declaring variables in the global scope.
    - **Deeply Nested Callbacks**: Refactor code to use async/await for better readability.
    - **Ignoring Errors**: Always handle potential errors using try/catch blocks or `Result` types.
    - **Over-Engineering**: Keep code simple and avoid unnecessary complexity.
    - **Not Using `deps.ts`**:  Failing to centralize and version control your dependencies will make maintenance difficult.
- **State Management Best Practices**:
    - For simple applications, use local component state.
    - For complex applications, consider using a lightweight state management library or a more robust solution like Redux (via a compatible Deno port).
    - Centralize application state and manage it predictably.
    - Avoid mutating state directly; use immutable updates.
- **Error Handling Patterns**:
    - Use try/catch blocks for synchronous error handling.
    - Use `.catch()` for handling errors in Promises.
    - Create custom error types for specific scenarios.
    - Log errors with sufficient context for debugging.
    - Consider using a `Result` type to explicitly handle success or failure cases.

## 3. Performance Considerations

- **Optimization Techniques**:
    - **Minimize I/O Operations**: Batch file reads/writes when possible.
    - **Efficient Data Structures**: Use appropriate data structures for your use case (e.g., `Map` vs. `Object`).
    - **Asynchronous Operations**: Leverage async/await for non-blocking operations.
    - **WebAssembly**: Utilize WebAssembly modules for performance-critical tasks.
    - **`Deno.bench`**: Use Deno's built-in benchmarking tool to identify performance bottlenecks.  Run with `deno bench`.  Write benchmarks adjacent to the files you are testing.
- **Memory Management Considerations**:
    - Be mindful of memory leaks, especially when working with streams and resources.
    - Use `finally` blocks to ensure resources are properly closed.
    - Avoid creating unnecessary objects and variables.
    - Profile your application to identify memory usage patterns.
- **Rendering Optimization (if applicable)**:
    - Use virtual DOM techniques for efficient UI updates.
    - Optimize images and other assets.
    - Implement lazy loading for offscreen content.
- **Bundle Size Optimization**:
    - Use `deno bundle` to create optimized bundles for deployment.
    - Remove unused code with tree shaking.
    - Minify JavaScript and CSS files.
    - Compress assets with gzip or Brotli.
- **Lazy Loading Strategies**:
    - Use dynamic imports (`import()`) to load modules on demand.
    - Implement lazy loading for images and other assets.
    - Split code based on routes or features.

## 4. Security Best Practices

- **Common Vulnerabilities and Prevention**:
    - **Remote Code Execution**:  Be extremely cautious with `Deno.run`, validate inputs, and never execute untrusted code.
    - **Cross-Site Scripting (XSS)**: Sanitize user inputs to prevent XSS attacks (if applicable).
    - **SQL Injection**: Use parameterized queries or an ORM to prevent SQL injection (if applicable).
    - **Denial of Service (DoS)**: Implement rate limiting and other measures to prevent DoS attacks.
    - **Supply Chain Attacks**: Carefully review and pin dependencies in `deps.ts` and `deno.lock`.
- **Input Validation**:
    - Validate all user inputs to prevent injection attacks and data corruption.
    - Use regular expressions or dedicated validation libraries to enforce input formats.
    - Sanitize inputs to remove potentially malicious characters.
- **Authentication and Authorization**:
    - Use secure authentication mechanisms like OAuth 2.0 or JWT.
    - Implement role-based access control (RBAC) or attribute-based access control (ABAC) for authorization.
    - Store passwords securely using bcrypt or Argon2.
- **Data Protection**:
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all communication.
    - Implement proper access controls to protect data from unauthorized access.
    - Regularly back up data and test recovery procedures.
- **Secure API Communication**:
    - Use HTTPS for all API communication.
    - Validate API requests and responses.
    - Implement rate limiting to prevent abuse.
    - Use API keys or JWT for authentication.

## 5. Testing Approaches

- **Unit Testing**:
    - Write unit tests for individual components and functions.
    - Use Deno's built-in testing framework (`Deno.test`).
    - Mock dependencies to isolate units under test.
    - Aim for high test coverage.
- **Integration Testing**:
    - Write integration tests to verify interactions between different modules.
    - Test API endpoints and data flows.
    - Use a test database or mock API for integration tests.
- **End-to-End Testing**:
    - Write end-to-end tests to verify the entire application flow.
    - Use a testing framework like Playwright or Puppeteer.
    - Test user interactions and UI elements (if applicable).
- **Test Organization**:
    - Create a `tests/` directory to store all tests.
    - Organize tests into subdirectories based on module or feature.
    - Use descriptive test names.
    - Run tests using `deno test`.
- **Mocking and Stubbing**:
    - Use Deno's built-in mocking capabilities or a mocking library like `sinon`.
    - Mock external dependencies and API calls.
    - Use stubs to replace complex functions with simplified versions.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes**:
    - **Incorrect Permissions**: Not requesting necessary permissions (e.g., `--allow-read`, `--allow-net`).
    - **Uncaught Exceptions**: Not handling errors properly.
    - **Global Variable Pollution**: Accidentally declaring variables in the global scope.
    - **Missing `await`**: Forgetting to `await` asynchronous operations.
    - **Inconsistent Dependency Versions**: Not managing dependency versions using `deps.ts`.
- **Edge Cases**:
    - **Handling large files**: Efficiently stream large files to avoid memory issues.
    - **Cross-platform compatibility**: Test your application on different operating systems.
    - **Time zone differences**: Handle time zone conversions carefully.
- **Version-Specific Issues**:
    - Be aware of breaking changes in Deno releases.
    - Consult the Deno release notes for migration guidance.
- **Compatibility Concerns**:
    - Ensure compatibility with different browsers and Deno versions (if applicable).
    - Be mindful of differences between Deno and Node.js APIs.
- **Debugging Strategies**:
    - Use Deno's built-in debugger (`deno inspect`).
    - Use `console.log` statements for basic debugging.
    - Use a code editor with Deno support for advanced debugging features.
    - Inspect network requests and responses using browser developer tools.

## 7. Tooling and Environment

- **Recommended Development Tools**:
    - **VS Code** with the Deno extension.
    - **IntelliJ IDEA** with the Deno plugin.
    - **Deno CLI**.
- **Build Configuration**:
    - Use `deno.json` to configure Deno settings, like import maps, linting rules, and formatting options. Example:

    json
    {
      "imports": {
        "*": "./src/",
        "std/": "https://deno.land/std@0.224.0/"
      },
      "lint": {
        "rules": {
          "no-explicit-any": true
        }
      },
      "fmt": {
        "lineWidth": 120,
        "indentWidth": 2
      },
      "compilerOptions": {
          "jsx": "react-jsx",
          "jsxImportSource": "react"
      }
    }
    
- **Linting and Formatting**:
    - Use `deno lint` to check for code style and potential errors. Configure linting rules in `deno.json`.
    - Use `deno fmt` to format code according to the configured style. Configure formatting options in `deno.json`.
    - Integrate linting and formatting into your workflow using pre-commit hooks or CI/CD pipelines.
- **Deployment**:
    - **Deno Deploy**: For easy serverless hosting of Deno applications.  Requires no configuration files to start.
    - **Docker**:  Containerize your Deno app for consistent deployments.
    - **Cloud Providers**: Deploy to AWS, Google Cloud, or Azure using standard deployment methods.
- **CI/CD Integration**:
    - Use GitHub Actions, GitLab CI, or other CI/CD platforms to automate testing, linting, formatting, and deployment.
    - Configure CI/CD pipelines to run tests on every commit or pull request.
    - Automate deployments to production environments.

## Additional Best Practices

- **Use Underscores in Filenames**: Use `file_server.ts` instead of `file-server.ts`.
- **Write Tests for New Features**: Each module should contain or be accompanied by tests for its public functionality.
- **TODO Comments**: TODO comments should include an issue or the author's GitHub username in parentheses. Example: `// TODO(ry): Add tests.`
- **Be Explicit**: Be explicit, even when it means more code.
- **Inclusive Code**: Follow the guidelines for inclusive code.
- **Minimize Dependencies**: Do not make circular imports.
- **JSDoc**: Provide excellent JSDoc coverage for all exported symbols.
- **Resolve Linting Problems**: Use `// deno-lint-ignore <rule>` to suppress warnings when necessary, but use sparingly.
- **Test Module**: Every module with public functionality `foo.ts` should have a test module `foo_test.ts`.
- **Explicit Unit Tests**: Name unit tests clearly.
- **Top-Level Functions**: Use the `function` keyword for top-level functions, not arrow syntax.
- **std Library**: Do not depend on external code in the standard library.
- **Browser Compatibility**: Document and maintain browser compatibility where appropriate.
- **Prefer `#` over `private`**:  Prefer `#` (private class fields) over the `private` keyword.
- **Naming Conventions**: Use camelCase, PascalCase, and UPPER_SNAKE_CASE consistently for variables, functions, classes, types, and constants.

By adhering to these best practices, you can create maintainable, efficient, and secure Deno applications.