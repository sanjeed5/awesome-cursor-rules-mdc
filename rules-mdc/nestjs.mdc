---
description: This rule provides comprehensive guidance on NestJS best practices, coding standards, and architectural patterns. It aims to help developers build scalable, maintainable, and performant NestJS applications by covering code organization, security, testing, and other essential aspects.
globs: *.ts
---
- **Code Organization and Structure:**
  - **Directory Structure:**
    - Adopt a modular structure that reflects the application's domain or features. A common approach includes organizing code into modules, services, controllers, DTOs, and entities, each in their own directory.
    - Example:
      
      src/
      ├── app.module.ts
      ├── auth/
      │   ├── auth.module.ts
      │   ├── auth.controller.ts
      │   ├── auth.service.ts
      │   ├── strategies/
      │   │   └── jwt.strategy.ts
      │   ├── dtos/
      │   │   └── create-user.dto.ts
      │   └── entities/
      │       └── user.entity.ts
      ├── users/
      │   ├── users.module.ts
      │   ├── users.controller.ts
      │   ├── users.service.ts
      │   └── ...
      ├── core/
      │   ├── filters/
      │   │   └── http-exception.filter.ts
      │   ├── interceptors/
      │   │   └── logging.interceptor.ts
      │   └── ...
      └── main.ts
      
  - **File Naming Conventions:**
    - Use descriptive and consistent naming conventions.  Prefix files based on their role (e.g., `user.controller.ts`, `auth.service.ts`, `create-user.dto.ts`).
    - Use PascalCase for classes and interfaces (e.g., `UserService`, `CreateUserDto`).
    - Use camelCase for instances and variables (e.g., `userService`, `createUserDto`).
  - **Module Organization:**
    - Encapsulate features within modules. Each module should represent a distinct part of the application and handle related functionality.
    - Modules should import necessary dependencies and export components that other modules need.
    - Use the `forRoot` and `forFeature` methods for configuration and feature modules, respectively, especially when dealing with database connections or other shared resources.
  - **Component Architecture:**
    - Follow the SOLID principles for designing components.  Each component (controller, service, etc.) should have a single responsibility.
    - Use dependency injection to manage dependencies between components, making them more testable and maintainable.
    - Controllers should handle request routing and validation, services should implement business logic, and entities should represent data models.
  - **Code Splitting Strategies:**
    - For large applications, consider splitting modules into smaller, more manageable chunks using feature modules or lazy-loaded modules.
    - Use dynamic imports and lazy loading to improve initial load times and reduce bundle size.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Dependency Injection:** Use NestJS's built-in dependency injection container to manage dependencies and promote loose coupling.
    - **Repository Pattern:** Abstract data access logic into repositories to decouple services from specific database implementations.
    - **Unit of Work:**  Use a Unit of Work pattern for managing transactions across multiple repositories.
    - **CQRS (Command Query Responsibility Segregation):** For complex applications, consider using CQRS to separate read and write operations, improving performance and scalability.
  - **Recommended Approaches:**
    - Use DTOs (Data Transfer Objects) for data validation and transformation between layers.
    - Implement global exception filters to handle errors consistently across the application.
    - Use interceptors for logging, caching, and other cross-cutting concerns.
    - Utilize pipes for request validation and data transformation.
    - Use asynchronous operations (`async/await`) for non-blocking I/O operations.
  - **Anti-patterns:**
    - **Tight Coupling:** Avoid creating tightly coupled components that are difficult to test and maintain. Use dependency injection and interfaces to promote loose coupling.
    - **God Classes:** Avoid creating classes with too many responsibilities. Break down large classes into smaller, more manageable components.
    - **Ignoring Errors:** Always handle errors properly using try-catch blocks, exception filters, and logging. Never ignore errors or swallow exceptions.
    - **Hardcoding Configuration:** Avoid hardcoding configuration values directly in the code. Use environment variables or configuration files to manage settings.
  - **State Management:**
    - For simple applications, use services to manage application state.
    - For more complex applications, consider using a state management library like Redux or NgRx (although this is less common on the backend).
    - Avoid storing sensitive data in the client-side state. Store it securely on the server.
  - **Error Handling:**
    - Implement a global exception filter to catch unhandled exceptions and return appropriate error responses to the client.
    - Use custom exceptions to represent specific error conditions in the application.
    - Log errors with sufficient detail to facilitate debugging.
    - Return consistent error responses with appropriate HTTP status codes.

- **Performance Considerations:**
  - **Optimization Techniques:**
    - Use caching to reduce database load and improve response times.  NestJS provides built-in support for caching using interceptors.
    - Optimize database queries by using indexes, avoiding N+1 queries, and using efficient data retrieval methods.
    - Use connection pooling to reduce the overhead of establishing database connections.
    - Profile the application to identify performance bottlenecks and optimize accordingly.
  - **Memory Management:**
    - Avoid memory leaks by properly managing resources and releasing unused objects.
    - Use streams for handling large files or data streams.
    - Use object pooling to reuse frequently created objects.
  - **Rendering Optimization (Server-Side Rendering):**
    - Not directly applicable to NestJS, as it's primarily a backend framework. However, if using SSR, optimize rendering performance by caching rendered pages and using efficient templating engines.
  - **Bundle Size Optimization:**
    - Use tree shaking to remove unused code from the bundle.
    - Minify and compress code to reduce bundle size.
    - Use code splitting to load only the necessary code for each route or module.
  - **Lazy Loading:**
    - Use lazy loading to load modules or features on demand, improving initial load times.
    - Implement code splitting to create smaller bundles that can be loaded independently.

- **Security Best Practices:**
  - **Common Vulnerabilities:**
    - **SQL Injection:** Prevent SQL injection by using parameterized queries or an ORM that automatically escapes user inputs.
    - **Cross-Site Scripting (XSS):** Protect against XSS by sanitizing user inputs and encoding outputs.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection using tokens or other mechanisms.
    - **Authentication and Authorization Flaws:** Secure authentication and authorization by using strong passwords, multi-factor authentication, and role-based access control.
    - **Insecure Direct Object References (IDOR):** Prevent IDOR by validating user access to resources before granting access.
  - **Input Validation:**
    - Validate all user inputs to prevent malicious data from entering the system. Use DTOs and validation pipes to enforce input constraints.
    - Sanitize user inputs to remove or escape potentially harmful characters.
    - Validate file uploads to prevent malicious files from being uploaded.
  - **Authentication and Authorization:**
    - Use JWT (JSON Web Tokens) for authentication and authorization.
    - Implement role-based access control (RBAC) to restrict access to resources based on user roles.
    - Use secure password hashing algorithms (e.g., bcrypt) to store passwords securely.
    - Implement rate limiting to prevent brute-force attacks.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to encrypt communication between the client and server.
    - Store secrets securely using environment variables or a secrets management system.
  - **Secure API Communication:**
    - Use API keys or OAuth 2.0 for API authentication and authorization.
    - Implement request validation and rate limiting to protect APIs from abuse.
    - Use a secure API gateway to manage API traffic and enforce security policies.

- **Testing Approaches:**
  - **Unit Testing:**
    - Write unit tests for individual components (services, controllers, etc.) to verify their functionality in isolation.
    - Use mocking and stubbing to isolate components from their dependencies.
    - Follow the Arrange-Act-Assert pattern for writing clear and concise unit tests.
  - **Integration Testing:**
    - Write integration tests to verify the interaction between multiple components or modules.
    - Test the integration between the application and external systems (e.g., databases, APIs).
  - **End-to-End Testing:**
    - Write end-to-end tests to verify the application's functionality from a user's perspective.
    - Use tools like Puppeteer or Cypress to automate end-to-end tests.
  - **Test Organization:**
    - Organize tests into separate directories that mirror the application's directory structure.
    - Use descriptive names for test files and test cases.
  - **Mocking and Stubbing:**
    - Use mocking frameworks (e.g., Jest, Sinon.js) to create mock objects and stub methods.
    - Use dependency injection to make components easily testable.

- **Common Pitfalls and Gotchas:**
  - **Frequent Mistakes:**
    - **Not using DTOs for validation:** Always use DTOs and validation pipes to ensure data integrity.
    - **Ignoring environment variables:** Use environment variables for configuration to avoid hardcoding values.
    - **Not handling exceptions properly:** Implement global exception filters to catch unhandled exceptions and return appropriate error responses.
    - **Overlooking security vulnerabilities:** Be aware of common security vulnerabilities and take steps to mitigate them.
  - **Edge Cases:**
    - **Handling large file uploads:** Use streams and appropriate buffering techniques to handle large file uploads efficiently.
    - **Dealing with concurrent requests:** Use appropriate locking mechanisms or transaction management to handle concurrent requests safely.
  - **Version-Specific Issues:**
    - Be aware of breaking changes between NestJS versions and update code accordingly.
    - Consult the NestJS documentation for migration guides and compatibility information.
  - **Compatibility Concerns:**
    - Ensure compatibility with different Node.js versions and operating systems.
    - Test the application on different browsers and devices to ensure cross-platform compatibility.
  - **Debugging Strategies:**
    - Use the NestJS debugger to step through code and inspect variables.
    - Use logging to track the flow of execution and identify errors.
    - Use profiling tools to identify performance bottlenecks.

- **Tooling and Environment:**
  - **Recommended Development Tools:**
    - **IDE:** Visual Studio Code with NestJS extensions.
    - **CLI:** Nest CLI for generating and managing NestJS projects.
    - **Database:** PostgreSQL, MySQL, MongoDB, or other compatible database.
    - **Testing:** Jest, Supertest.
  - **Build Configuration:**
    - Use `tsconfig.json` to configure the TypeScript compiler.
    - Use Webpack or Parcel to bundle and optimize the application.
  - **Linting and Formatting:**
    - Use ESLint and Prettier to enforce code style and formatting rules.
    - Configure pre-commit hooks to automatically lint and format code before committing.
  - **Deployment:**
    - Deploy to platforms like Heroku, AWS, Google Cloud, or Azure.
    - Use Docker to containerize the application and simplify deployment.
  - **CI/CD Integration:**
    - Integrate with CI/CD pipelines (e.g., Jenkins, Travis CI, GitHub Actions) to automate testing and deployment.
    - Use environment-specific configuration files for different environments (development, staging, production).