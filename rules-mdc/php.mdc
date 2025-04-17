---
description: This rule provides guidelines for PHP coding best practices, covering code structure, security, performance, and testing to improve code quality and maintainability.
globs: *.php
---
# PHP Best Practices and Coding Standards

This document outlines best practices and coding standards for PHP development to ensure code quality, maintainability, security, and performance.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

*   **`src/`:**  Contains the core source code of your application or library.
*   **`public/`:**  The document root of your application, containing publicly accessible files like `index.php`, CSS, JavaScript, and images. All requests should be routed through this directory.
*   **`config/`:** Stores configuration files for different environments (development, testing, production).
*   **`tests/`:** Contains unit, integration, and functional tests.
*   **`vendor/`:**  Managed by Composer, containing third-party libraries and dependencies.  Do not modify contents manually.
*   **`cache/`:** Directory used for caching data.
*   **`logs/`:** Directory for storing application logs.  Use descriptive names for log files.
*   **`data/`:**  Stores data files (e.g., SQLite databases).
*   **`resources/`:** Contains view templates, language files, and other resources.
*   **`migrations/`:**  Database migration files for version control of your database schema.


project/
├── src/
│   ├── Controller/
│   │   └── UserController.php
│   ├── Model/
│   │   └── User.php
│   └── ...
├── public/
│   └── index.php
├── config/
│   └── config.php
├── tests/
│   ├── Unit/
│   │   └── UserTest.php
│   └── ...
├── vendor/
│   └── ...
├── cache/
│   └── ...
├── logs/
│   └── app.log
├── data/
│   └── database.sqlite
├── resources/
│   ├── views/
│   │   └── user.php
│   └── lang/
│       └── en.php
└── migrations/
    └── 2024_01_01_create_users_table.php


### 1.2. File Naming Conventions

*   **Classes:** Use PascalCase (e.g., `UserController.php`). The filename should match the class name.
*   **Functions and methods:** Use camelCase (e.g., `getUserById()`).
*   **Variables:** Use camelCase (e.g., `$userName`).
*   **Constants:** Use UPPER_SNAKE_CASE (e.g., `MAX_USER_AGE`).
*   **Configuration files:** Use lowercase with underscores (e.g., `database_config.php`).
*   **View files:** Use lowercase with hyphens (e.g., `user-profile.php`).

### 1.3. Module Organization

*   **Separate concerns:** Organize code into modules based on functionality (e.g., authentication, user management, payment processing).
*   **Use namespaces:** Group related classes and interfaces within namespaces to avoid naming conflicts and improve code organization (e.g., `namespace App\Controllers;`). Follow PSR-4 autoloading standards.
*   **Dependency Injection:** Use dependency injection to decouple modules and make them easier to test and maintain.
*   **Interfaces and Abstract Classes:** Define interfaces for modules to ensure loose coupling and allow for different implementations.

### 1.4. Component Architecture

*   **Components:** Independent, reusable pieces of functionality that can be composed to build larger applications.
*   **Thin Controllers:**  Keep controllers light, delegate business logic to service classes or model layer. 
*   **Repositories:** Abstract data access logic from the rest of the application. This allows you to easily switch between different data sources (e.g., databases, APIs) without modifying the core application code.
*   **Services:**  Encapsulate complex business logic.

### 1.5. Code Splitting

*   **Lazy loading:** Load modules or components only when they are needed to improve initial load time.
*   **Composer autoloading:** Composer efficiently handles autoloading of classes based on namespaces, splitting code into separate files.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Singleton:**  Ensures that a class has only one instance and provides a global point of access to it.  Use sparingly, as overuse can lead to tight coupling and make testing difficult.
*   **Factory:**  Creates objects without specifying the exact class to create. Useful for abstracting object creation.
*   **Observer:**  Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. Useful for event handling.
*   **Strategy:**  Defines a family of algorithms, encapsulates each one, and makes them interchangeable.  Lets the algorithm vary independently from clients that use it.
*   **Model-View-Controller (MVC):** Separates application logic (Model), user interface (View), and input handling (Controller). Most PHP frameworks are based on MVC.
*   **Repository Pattern:** Abstract data access logic. Useful when you want to be able to easily switch data sources or when your data access logic is complex.

### 2.2. Recommended Approaches

*   **Database interactions:**  Use prepared statements with parameterized queries to prevent SQL injection.
*   **Templating:** Use a templating engine like Twig or Blade to separate presentation logic from PHP code. Helps to maintain code clean and readable.
*   **Routing:**  Use a routing component to map URLs to controllers and actions. Simplifies URL handling.
*   **Error handling:** Implement a consistent error handling strategy using exceptions and logging.
*   **Dependency Injection:** Use a dependency injection container to manage dependencies and promote loose coupling.  Frameworks often include DI containers.

### 2.3. Anti-patterns and Code Smells

*   **Global state:** Avoid using global variables as they can lead to unpredictable behavior and make code difficult to test.
*   **Tight coupling:** Minimize dependencies between classes and modules to improve reusability and maintainability.
*   **Long methods/functions:** Break down large methods into smaller, more focused functions.
*   **Copy-pasted code:**  Extract common functionality into reusable functions or classes.
*   **Ignoring errors:**  Always handle exceptions and log errors appropriately.
*   **Over-use of static methods:**  Limit the use of static methods, which can make testing more difficult.
*   **Code duplication:** Follow the DRY (Don't Repeat Yourself) principle.

### 2.4. State Management

*   **Sessions:** Use PHP sessions for managing user authentication and temporary data. Configure session security settings (e.g., `session.cookie_httponly`, `session.cookie_secure`).
*   **Cookies:**  Use cookies for storing small amounts of data on the client-side.  Be mindful of cookie size limits and security.
*   **Caching:**  Use caching mechanisms (e.g., Memcached, Redis, file-based caching) to improve performance by storing frequently accessed data.
*   **Database:**  Store persistent data in a database.  Use appropriate data types and indexes for optimal performance.
*   **Statelessness:**  Design APIs to be stateless whenever possible to improve scalability.

### 2.5. Error Handling

*   **Exceptions:**  Use exceptions to handle errors and exceptional situations. Throw exceptions when something goes wrong.
*   **Try-catch blocks:** Wrap code that might throw an exception in try-catch blocks to handle errors gracefully.
*   **Custom exception classes:** Create custom exception classes to represent specific error conditions in your application.
*   **Logging:**  Log errors, warnings, and informational messages using a logging library (e.g., Monolog).  Configure logging levels.
*   **Error reporting:**  Configure PHP error reporting settings (`error_reporting`, `display_errors`) appropriately for different environments.  Disable displaying errors in production.
*   **Global Exception Handler:** Set a global exception handler to catch uncaught exceptions and log them.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Opcode caching:** Use an opcode cache like OPcache (built into PHP) to store compiled PHP code in memory, reducing the need to recompile code on each request.
*   **Caching:** Implement caching strategies at different levels (e.g., database caching, object caching, page caching).
*   **Database optimization:** Optimize database queries, use indexes, and avoid N+1 query problems.
*   **Code profiling:** Use a profiler (e.g., Xdebug) to identify performance bottlenecks in your code.
*   **Minimize file I/O:**  Reduce the number of file read/write operations.
*   **Use appropriate data structures:** Choose the right data structures for your needs (e.g., arrays, linked lists, hash tables).
*   **Avoid resource-intensive operations:**  Minimize the use of functions that consume a lot of resources (e.g., regular expressions).
*   **Use PHP 7.x/8.x features:**  Take advantage of performance improvements in newer versions of PHP.

### 3.2. Memory Management

*   **Unset variables:** Unset variables that are no longer needed to free up memory.
*   **Avoid large arrays:**  Avoid storing large amounts of data in arrays.  Use iterators or generators if possible.
*   **Garbage collection:**  PHP automatically manages memory using garbage collection.  Ensure that your code doesn't create memory leaks (e.g., circular references).
*   **Limit session size:** Be mindful of the amount of data stored in sessions, as large sessions can consume a lot of memory.

### 3.3. Rendering Optimization

*   **Minimize HTTP requests:** Reduce the number of HTTP requests by combining CSS and JavaScript files, using CSS sprites, and inlining small images.
*   **Optimize images:** Compress images and use appropriate image formats (e.g., JPEG, PNG, WebP).
*   **Use browser caching:** Configure browser caching to store static assets (e.g., CSS, JavaScript, images) in the browser cache.
*   **Content Delivery Network (CDN):**  Use a CDN to serve static assets from geographically distributed servers.
*   **Gzip compression:** Enable Gzip compression to reduce the size of HTTP responses.
*   **Minify HTML, CSS, and JavaScript:** Remove unnecessary characters from HTML, CSS, and JavaScript files.

### 3.4. Bundle Size Optimization (Less relevant for traditional PHP, but important for frontend assets)

*   **Code splitting:** Split frontend code into smaller chunks that can be loaded on demand. (Relevant if using a frontend framework like React, Vue, or Angular alongside PHP).
*   **Tree shaking:** Remove unused code from your JavaScript bundles. (Relevant if using a frontend framework).

### 3.5. Lazy Loading

*   **Lazy-load images:** Load images only when they are visible in the viewport.
*   **Lazy-load modules:** Load modules or components only when they are needed.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **SQL injection:**  Occurs when untrusted data is used to construct SQL queries. Prevent by using prepared statements with parameterized queries.
*   **Cross-Site Scripting (XSS):**  Occurs when malicious scripts are injected into a website. Prevent by sanitizing user input and escaping output.
*   **Cross-Site Request Forgery (CSRF):**  Occurs when a malicious website tricks a user into performing unwanted actions on another website. Prevent by using CSRF tokens.
*   **Remote Code Execution (RCE):**  Occurs when an attacker can execute arbitrary code on the server. Prevent by avoiding dangerous functions (e.g., `eval()`, `system()`) and carefully validating user input.
*   **File Inclusion:** Occurs when an attacker can include arbitrary files on the server. Prevent by carefully validating user input and restricting file access.
*   **Session Hijacking:** Occurs when an attacker steals a user's session ID. Prevent by using secure session management practices.
*   **Denial of Service (DoS):** Occurs when an attacker overwhelms a server with requests, making it unavailable to legitimate users. Prevent by implementing rate limiting and other security measures.

### 4.2. Input Validation

*   **Sanitize user input:**  Remove or encode potentially dangerous characters from user input.
*   **Validate user input:**  Verify that user input conforms to expected formats and values.
*   **Use whitelisting:**  Allow only known good values for user input.  Avoid blacklisting, which can be easily bypassed.
*   **Escape output:**  Escape output to prevent XSS attacks.

### 4.3. Authentication and Authorization

*   **Use strong passwords:**  Enforce strong password policies and use a secure password hashing algorithm (e.g., bcrypt, Argon2).
*   **Salt passwords:**  Add a unique salt to each password before hashing it.
*   **Store passwords securely:** Store password hashes in a secure database.
*   **Implement access control:** Restrict access to resources based on user roles and permissions.
*   **Use multi-factor authentication (MFA):**  Add an extra layer of security by requiring users to provide multiple forms of authentication.
*   **Secure cookies:**  Set the `HttpOnly` and `Secure` flags on cookies to prevent XSS and man-in-the-middle attacks.
*   **Use HTTPS:**  Encrypt all communication between the client and server using HTTPS.

### 4.4. Data Protection

*   **Encrypt sensitive data:** Encrypt sensitive data at rest and in transit.
*   **Use encryption keys securely:** Store encryption keys in a secure location and rotate them regularly.
*   **Comply with data privacy regulations:** Comply with data privacy regulations like GDPR and CCPA.
*   **Implement data masking:** Mask sensitive data when it is not needed for processing.
*   **Regularly back up data:** Regularly back up data to protect against data loss.

### 4.5. Secure API Communication

*   **Use HTTPS:** Enforce HTTPS for all API endpoints.
*   **Authenticate API requests:** Use API keys, OAuth 2.0, or other authentication mechanisms to verify the identity of API clients.
*   **Authorize API requests:**  Restrict access to API endpoints based on user roles and permissions.
*   **Rate limit API requests:** Implement rate limiting to prevent abuse.
*   **Validate API input:** Validate all API input to prevent injection attacks.
*   **Sanitize API output:** Sanitize all API output to prevent XSS attacks.
*   **Log API requests:** Log all API requests for auditing and security monitoring.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test individual units of code:**  Focus on testing individual functions, methods, or classes in isolation.
*   **Use a testing framework:**  Use a unit testing framework like PHPUnit or Codeception.
*   **Write testable code:**  Design code to be easily testable (e.g., use dependency injection).
*   **Follow the Arrange-Act-Assert pattern:** Arrange the test data, act on the code being tested, and assert the expected results.
*   **Test edge cases:** Test edge cases and boundary conditions to ensure that code handles unexpected input correctly.

### 5.2. Integration Testing

*   **Test interactions between components:** Focus on testing how different components of the application interact with each other.
*   **Use a testing framework:** Use an integration testing framework like PHPUnit or Codeception.
*   **Test database interactions:** Test database interactions to ensure that data is stored and retrieved correctly.
*   **Test API integrations:** Test API integrations to ensure that data is exchanged correctly.

### 5.3. End-to-End Testing

*   **Test the entire application flow:** Focus on testing the entire application flow from start to finish.
*   **Use a testing framework:** Use an end-to-end testing framework like Selenium or Cypress.
*   **Automate tests:**  Automate end-to-end tests to ensure that the application continues to work correctly as it evolves.

### 5.4. Test Organization

*   **Create a `tests/` directory:** Store all test files in a `tests/` directory.
*   **Mirror the source code structure:**  Organize test files to mirror the source code structure (e.g., `tests/Unit/UserControllerTest.php`).
*   **Use namespaces:**  Use namespaces to organize test classes.
*   **Use descriptive test names:**  Use descriptive test names that clearly indicate what is being tested.

### 5.5. Mocking and Stubbing

*   **Use mocks to isolate units of code:**  Use mocks to replace dependencies with controlled test doubles.
*   **Use stubs to provide canned responses:** Use stubs to provide canned responses from dependencies.
*   **Use a mocking framework:** Use a mocking framework like Mockery or Prophecy.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Not escaping output:**  Failing to escape output, leading to XSS vulnerabilities.
*   **Using `mysql_*` functions:** Using deprecated `mysql_*` functions instead of `mysqli_*` or PDO.
*   **Not using prepared statements:** Not using prepared statements with parameterized queries, leading to SQL injection vulnerabilities.
*   **Ignoring errors:** Ignoring errors and warnings, making it difficult to debug problems.
*   **Not using namespaces:** Not using namespaces, leading to naming conflicts.
*   **Over-complicating code:**  Writing overly complex code that is difficult to understand and maintain.
*   **Not following coding standards:** Not following coding standards, leading to inconsistent code.

### 6.2. Edge Cases

*   **Handling different character encodings:**  Handling different character encodings correctly.
*   **Dealing with time zones:** Dealing with time zones correctly.
*   **Handling large file uploads:** Handling large file uploads without running out of memory.
*   **Dealing with concurrency:** Handling concurrency correctly in multi-threaded applications (less common in typical web applications).

### 6.3. Version-Specific Issues

*   **Deprecated features:**  Being aware of deprecated features and replacing them with their recommended alternatives.
*   **Compatibility issues:**  Ensuring that code is compatible with different versions of PHP.

### 6.4. Compatibility Concerns

*   **Database drivers:**  Ensuring that the correct database drivers are installed and configured.
*   **Web server configuration:** Configuring the web server (e.g., Apache, Nginx) correctly to serve PHP applications.
*   **Operating system compatibility:**  Ensuring that the application works correctly on different operating systems.

### 6.5. Debugging Strategies

*   **Use a debugger:** Use a debugger (e.g., Xdebug) to step through code and inspect variables.
*   **Log messages:**  Insert log messages to track the execution flow and identify problems.
*   **Use `var_dump()` or `print_r()`:** Use `var_dump()` or `print_r()` to inspect variables and data structures (use with caution in production).
*   **Read error messages:** Carefully read error messages and warnings to understand the cause of problems.
*   **Use a linter:** Use a linter to identify potential errors and code style violations.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **IDE:** PhpStorm, VS Code with PHP extensions.
*   **Debugger:** Xdebug.
*   **Package manager:** Composer.
*   **Database client:** Dbeaver, phpMyAdmin.
*   **Version control:** Git.
*   **Virtualization:** Docker, Vagrant.
*   **Profiler:** Xdebug, Blackfire.io

### 7.2. Build Configuration

*   **Use Composer:**  Use Composer to manage dependencies.
*   **Define dependencies in `composer.json`:**  List all dependencies in the `composer.json` file.
*   **Use semantic versioning:** Use semantic versioning to specify dependency versions.
*   **Use a `composer.lock` file:**  Commit the `composer.lock` file to version control to ensure that everyone is using the same versions of dependencies.
*   **Configure autoloading:**  Configure autoloading to automatically load classes.

### 7.3. Linting and Formatting

*   **Use a linter:** Use a linter (e.g., PHPStan, Psalm) to identify potential errors and code style violations.
*   **Use a code formatter:** Use a code formatter (e.g., PHP CS Fixer) to automatically format code according to coding standards.
*   **Configure linting and formatting rules:**  Configure linting and formatting rules to enforce coding standards.
*   **Integrate linting and formatting into the development workflow:**  Integrate linting and formatting into the development workflow to automatically check code for errors and style violations.

### 7.4. Deployment

*   **Use version control:** Use version control to track changes to code.
*   **Automate deployments:**  Automate deployments using a deployment tool (e.g., Deployer, Capistrano).
*   **Use environment variables:**  Use environment variables to configure application settings for different environments.
*   **Separate code and configuration:** Separate code and configuration to make it easier to deploy applications to different environments.
*   **Use a build process:**  Use a build process to prepare code for deployment (e.g., run linters, formatters, and tests).
*   **Test deployments:** Test deployments in a staging environment before deploying to production.

### 7.5. CI/CD

*   **Use a CI/CD platform:** Use a CI/CD platform (e.g., Jenkins, Travis CI, CircleCI, GitHub Actions) to automate the build, test, and deployment process.
*   **Configure CI/CD pipelines:** Configure CI/CD pipelines to run tests, linters, and formatters automatically on every commit.
*   **Automate deployments:**  Automate deployments using CI/CD pipelines.
*   **Use infrastructure as code:**  Use infrastructure as code (e.g., Terraform, Ansible) to manage infrastructure.

This document aims to provide a comprehensive overview of PHP best practices and coding standards. By following these guidelines, developers can create high-quality, maintainable, secure, and performant PHP applications.