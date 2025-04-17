---
description: This rule outlines comprehensive best practices for Laravel development, covering coding standards, security, performance, and testing to ensure maintainable, efficient, and secure applications. It provides guidelines for code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and tooling.
globs: *.php
---
- Adhere to PSR coding standards (PSR-1, PSR-2, PSR-12).
- Use meaningful and descriptive variable, function, and class names.
- Organize routes effectively, leveraging resource controllers and route groups.
- Use Eloquent ORM for database interactions; avoid raw SQL queries where possible (unless necessary for performance).
- Implement caching strategies using Laravel's caching system for frequently accessed data.
- Maintain a clear and consistent project structure following Laravel's conventions (app, config, database, public, resources, routes, storage).
- Ensure code simplicity and readability to enhance maintainability.  Follow the single responsibility principle.
- Keep Laravel and its packages up-to-date to mitigate security vulnerabilities and leverage new features.

### 1. Code Organization and Structure:

   - **Directory Structure Best Practices:**
     - Follow Laravel's default directory structure: `app` (core application logic), `config` (configuration files), `database` (migrations and seeds), `public` (static assets), `resources` (views and assets), `routes` (route definitions), `storage` (file storage).
     - Organize `app` directory using subdirectories like `Models`, `Controllers`, `Services`, `Repositories`, `Exceptions`, `Policies`, `Providers`, and `Http/Middleware`.
     - Consider using modules for larger applications to encapsulate features.

   - **File Naming Conventions:**
     - Use PascalCase for class names (e.g., `UserController`).
     - Use camelCase for variable and function names (e.g., `$userName`, `getUserName()`).
     - Use snake_case for database table names and column names (e.g., `users`, `user_id`).
     - Use kebab-case for route names (e.g., `user.profile`).
     - Use descriptive names that clearly indicate the purpose of the file or class.

   - **Module Organization:**
     - For larger applications, use packages or modules to organize code into reusable components.
     - Implement module-specific service providers, routes, and configurations.
     - Isolate module dependencies to prevent conflicts.

   - **Component Architecture:**
     - Use Blade components for reusable UI elements.
     - Create custom components for complex logic and rendering.
     - Pass data to components using attributes and slots.
     - Encapsulate component logic in dedicated classes.

   - **Code Splitting Strategies:**
     - Use lazy loading for non-critical features.
     - Split large controllers into smaller, more manageable classes.
     - Extract complex logic into service classes or repositories.

### 2. Common Patterns and Anti-patterns:

   - **Design Patterns Specific to Laravel:**
     - **Repository Pattern:** Abstract data access logic from controllers.
     - **Service Pattern:** Encapsulate business logic into reusable classes.
     - **Observer Pattern:** Implement event-driven behavior for model changes.
     - **Factory Pattern:** Create test data and seed databases.
     - **Strategy Pattern:**  Define a family of algorithms and make them interchangeable.

   - **Recommended Approaches for Common Tasks:**
     - Use Eloquent ORM for database interactions, including relationships and aggregations.
     - Use Laravel's validation system for request data validation.
     - Use middleware for authentication, authorization, and request modification.
     - Use queues for background processing and asynchronous tasks.
     - Use events and listeners for decoupling components.

   - **Anti-patterns and Code Smells to Avoid:**
     - **God Classes:** Avoid creating large classes with too many responsibilities.
     - **Spaghetti Code:** Avoid complex and unstructured code that is difficult to understand and maintain.
     - **Copy-Paste Programming:** Avoid duplicating code; instead, create reusable components or functions.
     - **Ignoring Exceptions:** Always handle exceptions properly to prevent unexpected behavior.
     - **Over-Engineering:** Don't overcomplicate solutions with unnecessary complexity.
     - **Mass Assignment Vulnerability:** Use guarded or fillable attributes to protect against mass assignment vulnerabilities.

   - **State Management Best Practices:**
     - Use sessions for storing user-specific data.
     - Use cookies for storing client-side data.
     - Use the cache for storing frequently accessed data.
     - Use databases for persistent data storage.
     - Consider using Laravel's broadcasting feature for real-time updates.

   - **Error Handling Patterns:**
     - Use try-catch blocks to handle exceptions gracefully.
     - Use Laravel's exception handler to log and report errors.
     - Implement custom exception classes for specific error scenarios.
     - Provide informative error messages to users.

### 3. Performance Considerations:

   - **Optimization Techniques:**
     - Use caching to reduce database queries and improve response times.
     - Use eager loading to reduce N+1 query problems.
     - Use queues for background processing.
     - Optimize database queries with indexes and query optimization techniques.
     - Minimize the use of loops and conditional statements in performance-critical code.

   - **Memory Management:**
     - Avoid storing large amounts of data in memory.
     - Use garbage collection to free up memory.
     - Use streams for processing large files.

   - **Rendering Optimization:**
     - Use Blade's caching features to cache rendered views.
     - Use CSS and JavaScript minification to reduce file sizes.
     - Use image optimization techniques to reduce image sizes.

   - **Bundle Size Optimization:**
     - Use Laravel Mix to bundle and minify assets.
     - Remove unused CSS and JavaScript code.
     - Use code splitting to load only the necessary code for each page.

   - **Lazy Loading Strategies:**
     - Use lazy loading for images and other non-critical assets.
     - Use route model binding with eager loading to reduce database queries.

### 4. Security Best Practices:

   - **Common Vulnerabilities and How to Prevent Them:**
     - **SQL Injection:** Use Eloquent ORM and prepared statements to prevent SQL injection attacks.
     - **Cross-Site Scripting (XSS):** Sanitize user input and escape output to prevent XSS attacks.
     - **Cross-Site Request Forgery (CSRF):** Use CSRF protection tokens to prevent CSRF attacks.
     - **Mass Assignment:** Use guarded or fillable attributes to protect against mass assignment vulnerabilities.
     - **Authentication and Authorization:** Use Laravel's built-in authentication and authorization features.

   - **Input Validation:**
     - Use Laravel's validation system to validate all user input.
     - Sanitize user input to remove potentially harmful characters.
     - Validate file uploads to prevent malicious files from being uploaded.

   - **Authentication and Authorization Patterns:**
     - Use Laravel's built-in authentication system for user authentication.
     - Use policies to define authorization rules.
     - Use gates to authorize access to specific resources.
     - Implement two-factor authentication for enhanced security.

   - **Data Protection Strategies:**
     - Encrypt sensitive data using Laravel's encryption features.
     - Store passwords using bcrypt hashing.
     - Protect API keys and other sensitive configuration data.

   - **Secure API Communication:**
     - Use HTTPS for all API communication.
     - Use API tokens for authentication.
     - Implement rate limiting to prevent abuse.
     - Validate API requests and responses.

### 5. Testing Approaches:

   - **Unit Testing Strategies:**
     - Test individual units of code in isolation.
     - Use mock objects to isolate dependencies.
     - Write tests for all critical code paths.

   - **Integration Testing:**
     - Test the interaction between different components of the application.
     - Test database interactions and external API calls.

   - **End-to-End Testing:**
     - Test the entire application from end to end.
     - Use browser automation tools to simulate user interactions.

   - **Test Organization:**
     - Organize tests into logical groups.
     - Use descriptive test names.
     - Follow the arrange-act-assert pattern.

   - **Mocking and Stubbing:**
     - Use mock objects to isolate dependencies.
     - Use stubbing to replace complex dependencies with simpler implementations.

### 6. Common Pitfalls and Gotchas:

   - **Frequent Mistakes Developers Make:**
     - Not using dependency injection properly.
     - Writing complex logic in views.
     - Not using caching effectively.
     - Ignoring security vulnerabilities.
     - Not writing tests.

   - **Edge Cases to Be Aware Of:**
     - Handling large file uploads.
     - Dealing with concurrent requests.
     - Handling database connection errors.
     - Handling time zone conversions.

   - **Version-Specific Issues:**
     - Be aware of breaking changes between Laravel versions.
     - Consult the Laravel upgrade guide when upgrading to a new version.

   - **Compatibility Concerns:**
     - Ensure compatibility with different PHP versions and extensions.
     - Ensure compatibility with different database systems.

   - **Debugging Strategies:**
     - Use Laravel's debugging tools (e.g., debugbar, telescope).
     - Use logging to track application behavior.
     - Use Xdebug for step-by-step debugging.

### 7. Tooling and Environment:

   - **Recommended Development Tools:**
     - PHPStorm or VS Code with PHP extensions.
     - Composer for dependency management.
     - MySQL or PostgreSQL for database management.
     - Docker for containerization.

   - **Build Configuration:**
     - Use Laravel Mix to compile assets.
     - Use environment variables to configure the application.
     - Use a build script to automate the build process.

   - **Linting and Formatting:**
     - Use PHP CS Fixer to enforce coding standards.
     - Use ESLint and Prettier for JavaScript and CSS linting and formatting.

   - **Deployment Best Practices:**
     - Use a deployment tool like Envoyer or Deployer.
     - Use zero-downtime deployment strategies.
     - Use a CDN for static assets.

   - **CI/CD Integration:**
     - Use a CI/CD pipeline to automate testing and deployment.
     - Use tools like Jenkins, GitLab CI, or GitHub Actions.