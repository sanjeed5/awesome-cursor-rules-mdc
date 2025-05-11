---
description: Comprehensive coding standards, best practices, and architectural guidelines for Python backend development with the Bottle microframework, designed to improve code quality, maintainability, and security.
globs: *.py
---
# Bottle Framework Best Practices

This document provides comprehensive coding standards, best practices, and architectural guidelines for Python backend development using the Bottle microframework. Adhering to these guidelines will improve code quality, maintainability, security, and overall project success.

## 1. Code Organization and Structure

Effective code organization is crucial for maintainability and scalability.  Here's how to structure your Bottle applications:

### 1.1. Directory Structure Best Practices

A well-organized directory structure makes it easier to navigate and understand the project.


my_bottle_app/
├── app.py          # Main application entry point
├── routes/
│   ├── __init__.py   # Make 'routes' a package
│   ├── home.py       # Route definitions for the home page
│   ├── api.py        # Route definitions for the API
│   └── users.py      # Route definitions for user management
├── models/
│   ├── __init__.py
│   ├── user.py       # User model definition
│   └── product.py    # Product model definition
├── views/
│   ├── __init__.py
│   ├── home.tpl      # Template for the home page
│   └── user_list.tpl # Template for listing users
├── config.py       # Application configuration
├── db.py           # Database connection and setup
├── utils/
│   ├── __init__.py
│   ├── auth.py       # Authentication utilities
│   └── helpers.py    # General helper functions
├── tests/
│   ├── __init__.py
│   ├── test_routes.py  # Tests for route handlers
│   └── test_models.py  # Tests for data models
├── requirements.txt  # Project dependencies
└── README.md


### 1.2. File Naming Conventions

*   **Python Files:** Use lowercase with underscores (snake_case) for Python file names (e.g., `user_model.py`, `api_routes.py`).
*   **Template Files:** Use lowercase with underscores, and the `.tpl` extension (e.g., `product_details.tpl`).
*   **Configuration Files:** Use a descriptive name like `config.py`.
*   **Test Files:** Prefix test files with `test_` (e.g., `test_user_routes.py`).

### 1.3. Module Organization

*   **Separate Concerns:** Group related functionality into separate modules (e.g., routes, models, utilities).
*   **Use Packages:** Utilize Python packages (directories with `__init__.py` files) to further organize modules into logical groups.
*   **Avoid Circular Imports:** Carefully manage dependencies between modules to prevent circular import errors.  Structure your code so that lower-level modules (e.g., models, utilities) don't depend on higher-level modules (e.g., routes).

### 1.4. Component Architecture

*   **MVC (Model-View-Controller):** While Bottle is simple, aim for an MVC-like structure:
    *   **Models:** Represent data and business logic (e.g., database interactions).
    *   **Views:** Render the user interface (using Bottle's templating or another engine).
    *   **Controllers (Routes):** Handle user requests, interact with models, and select the appropriate view.
*   **Service Layer (Optional):**  For more complex applications, introduce a service layer between routes and models to encapsulate business logic and improve testability.

### 1.5. Code Splitting Strategies

*   **Functional Decomposition:** Break down complex functions into smaller, more manageable functions with single responsibilities.
*   **Route Grouping:**  Group related routes into separate modules or route blueprints for better organization.
*   **Configuration Management:** Move configuration settings to a separate module for easy management and environment-specific configurations.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Singleton (Carefully):**  Use sparingly for resources like database connections. Ensure thread safety if using a singleton in a multithreaded environment.
*   **Factory:**  Useful for creating model instances or other objects in a controlled manner.
*   **Middleware (Plugins):**  Bottle's plugin system allows you to create reusable middleware components for tasks like authentication, logging, and request processing.

### 2.2. Recommended Approaches for Common Tasks

*   **Database Access:** Use an ORM (SQLAlchemy) for database interaction.  This provides an abstraction layer, improves code readability, and helps prevent SQL injection vulnerabilities.
*   **Templating:** Leverage Bottle's built-in templating engine for simple projects. For more complex templating needs, consider Jinja2 or Mako.
*   **Request Handling:**  Use Bottle's decorators (`@route`, `@get`, `@post`) for defining routes and handling HTTP requests.
*   **Error Handling:** Implement custom error handlers for different HTTP status codes to provide informative error messages to the client.

### 2.3. Anti-patterns and Code Smells

*   **Fat Routes:** Avoid putting too much logic directly in route handlers.  Delegate complex tasks to models, services, or helper functions.
*   **Global State:** Minimize the use of global variables.  Prefer dependency injection or passing data as arguments.
*   **Hardcoding Configuration:**  Never hardcode sensitive information like API keys or database credentials. Use environment variables or a configuration file.
*   **Ignoring Exceptions:**  Don't catch exceptions and do nothing.  Log the error and take appropriate action (e.g., return an error response to the client).
*   **SQL Injection Vulnerabilities:** Avoid string concatenation when building SQL queries.  Use parameterized queries provided by your ORM or database driver.

### 2.4. State Management

*   **Stateless Architecture:**  Prefer a stateless architecture where each request is independent and doesn't rely on server-side session state.  This improves scalability.
*   **Sessions (if needed):**  If session state is required, use a session middleware (e.g., Beaker) or a token-based authentication system (JWT).
*   **Cookies:**  Use cookies for storing non-sensitive information on the client-side.

### 2.5. Error Handling

*   **Specific Exception Handling:** Catch specific exceptions rather than generic `Exception` to handle different error scenarios appropriately.
*   **Logging:** Use the `logging` module to log errors and other important events.  Configure different logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) for different environments.
*   **Custom Error Pages:** Create custom error pages for different HTTP status codes (404, 500, etc.) to provide a better user experience.
*   **Return Informative Error Responses:** When an error occurs, return a JSON response with a clear error message and status code.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Caching:** Implement caching for frequently accessed data to reduce database load and improve response times. Use Bottle's caching mechanism or an external caching system like Redis or Memcached.
*   **Database Optimization:** Optimize database queries by using indexes, avoiding unnecessary joins, and selecting only the required columns.
*   **Gzip Compression:** Enable Gzip compression for HTTP responses to reduce the amount of data transferred to the client.
*   **Asynchronous Operations:** Use asynchronous tasks (e.g., Celery) for long-running operations like sending emails or processing large datasets. This prevents blocking the main thread and improves responsiveness.
*   **Profiling:** Use profiling tools to identify performance bottlenecks in your code.

### 3.2. Memory Management

*   **Avoid Memory Leaks:**  Be careful when dealing with large datasets or long-running processes to avoid memory leaks.  Use generators or iterators to process data in chunks.
*   **Resource Management:**  Release resources (e.g., database connections, file handles) as soon as they are no longer needed.  Use `try...finally` blocks or context managers (`with` statement) to ensure resources are released even if an exception occurs.

### 3.3. Rendering Optimization

*   **Template Caching:**  Enable template caching to improve rendering performance.  Bottle's built-in templating engine supports caching.
*   **Minify CSS and JavaScript:**  Minify CSS and JavaScript files to reduce their size and improve loading times.
*   **Optimize Images:**  Optimize images for the web by using appropriate image formats, compression levels, and dimensions.

### 3.4. Bundle Size Optimization (Applicable if using Bottle to serve static assets)

*   **Code Splitting (For Single Page Apps):** If building a SPA, consider code-splitting your javascript into smaller bundles.
*   **Tree Shaking:** Use a bundler (like Webpack or Parcel) that supports tree shaking to remove unused code from your JavaScript bundles.

### 3.5. Lazy Loading

*   **Lazy Imports:**  Import modules only when they are needed. This can reduce startup time.
*   **Database Connection on Demand:**  Establish database connections only when they are required, rather than at application startup.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

*   **SQL Injection:**  Use parameterized queries with an ORM like SQLAlchemy to prevent SQL injection vulnerabilities.
*   **Cross-Site Scripting (XSS):** Sanitize user input before displaying it in templates to prevent XSS attacks.  Use Bottle's HTML escaping functions or a templating engine that automatically escapes HTML.
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection by generating and validating CSRF tokens for state-changing requests. Bottle doesn't have built-in CSRF protection, so you'll need to use a library or implement it manually.
*   **Authentication Bypass:** Ensure proper authentication and authorization mechanisms are in place to prevent unauthorized access to resources.
*   **Denial of Service (DoS):**  Implement rate limiting and input validation to prevent DoS attacks.

### 4.2. Input Validation

*   **Validate All User Input:**  Validate all user input on both the client-side and server-side.
*   **Use Data Type Validation:**  Ensure that input data matches the expected data types.
*   **Sanitize Input:**  Sanitize input data to remove potentially harmful characters or code.
*   **Limit Input Length:**  Limit the length of input fields to prevent buffer overflows.

### 4.3. Authentication and Authorization

*   **Use a Strong Authentication Scheme:** Implement a secure authentication scheme like OAuth 2.0 or JWT.
*   **Hash Passwords:**  Hash passwords using a strong hashing algorithm (e.g., bcrypt or Argon2) before storing them in the database. Never store passwords in plain text.
*   **Implement Role-Based Access Control (RBAC):**  Implement RBAC to control access to resources based on user roles.
*   **Use HTTPS:**  Always use HTTPS to encrypt communication between the client and server.

### 4.4. Data Protection

*   **Encrypt Sensitive Data:**  Encrypt sensitive data (e.g., credit card numbers, social security numbers) both in transit and at rest.
*   **Use Environment Variables:**  Store sensitive configuration data (e.g., API keys, database credentials) in environment variables rather than hardcoding them in the code.
*   **Regularly Update Dependencies:** Keep your dependencies up to date to patch security vulnerabilities.
*   **Secure File Uploads:** Validate file types and sizes during upload.  Store uploaded files outside the web root and serve them through a controlled endpoint.

### 4.5. Secure API Communication

*   **API Keys:** Use API keys for authentication and authorization of API clients.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your API.
*   **Input Validation:** Thoroughly validate all data received through the API.
*   **HTTPS Only:** Require HTTPS for all API communication.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Individual Components:** Unit tests should focus on testing individual components (e.g., functions, classes) in isolation.
*   **Use a Testing Framework:** Use a testing framework like `pytest` or `unittest` to write and run unit tests.
*   **Mock Dependencies:**  Mock external dependencies (e.g., database connections, API calls) to isolate the component being tested.
*   **Test Edge Cases:**  Test edge cases and boundary conditions to ensure that the component handles unexpected input correctly.
*   **Follow the Arrange-Act-Assert Pattern:**  Structure your tests using the Arrange-Act-Assert pattern to make them more readable and maintainable.

### 5.2. Integration Testing

*   **Test Interactions Between Components:** Integration tests should focus on testing the interactions between different components of the application.
*   **Use a Test Database:**  Use a separate test database to avoid affecting the production database.
*   **Test API Endpoints:**  Test API endpoints to ensure that they return the correct data and status codes.

### 5.3. End-to-End Testing

*   **Simulate User Interactions:** End-to-end tests should simulate user interactions with the application to ensure that the entire system works correctly.
*   **Use a Browser Automation Tool:** Use a browser automation tool like Selenium or Playwright to automate end-to-end tests.
*   **Test Common User Flows:**  Test common user flows to ensure that the application meets the needs of its users.

### 5.4. Test Organization

*   **Separate Test Files:**  Create separate test files for each module or component.
*   **Organize Tests by Feature:**  Organize tests into directories based on the feature they are testing.
*   **Use Descriptive Test Names:**  Use descriptive test names that clearly indicate what the test is verifying.

### 5.5. Mocking and Stubbing

*   **Use Mocking for External Dependencies:** Use mocking to simulate the behavior of external dependencies (e.g., databases, APIs) in your tests.
*   **Use Stubbing for Complex Calculations:** Use stubbing to replace complex calculations with precomputed values.
*   **Use a Mocking Library:**  Use a mocking library like `unittest.mock` or `pytest-mock` to simplify the process of creating mocks and stubs.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Not Sanitizing User Input:**  Failing to sanitize user input can lead to XSS vulnerabilities.
*   **Hardcoding Secrets:** Hardcoding sensitive information like API keys or database passwords directly in the code makes them vulnerable to exposure.
*   **Incorrect Error Handling:** Improper exception handling can result in unhandled errors and application crashes.
*   **Ignoring Security Updates:** Neglecting dependency updates leaves you exposed to known vulnerabilities.

### 6.2. Edge Cases

*   **Unicode Handling:** Ensure your application handles Unicode characters correctly, especially when dealing with user input or data from external sources.
*   **Time Zones:** Be mindful of time zones when dealing with dates and times.  Store dates and times in UTC and convert them to the user's local time zone when displaying them.
*   **Concurrency Issues:** If your application handles concurrent requests, be aware of potential race conditions and synchronization issues.

### 6.3. Version-Specific Issues

*  **Bottle Versions:** Stay informed about changes and bug fixes of specific Bottle versions you are using.  Check the Bottle changelog for changes that may impact your application.

### 6.4. Compatibility Concerns

*   **Python Version Compatibility:**  Ensure your application is compatible with the Python version you are using.  Bottle supports Python 3.6+.
*   **Dependency Conflicts:**  Be aware of potential dependency conflicts between different libraries. Use a virtual environment to isolate your project's dependencies.

### 6.5. Debugging Strategies

*   **Use a Debugger:** Use a debugger like `pdb` or the built-in debugger in your IDE to step through your code and inspect variables.
*   **Logging:** Use the `logging` module to log errors and other important events.  Configure different logging levels for different environments.
*   **Error Handling:**  Implement custom error handlers to catch exceptions and provide informative error messages.
*   **Testing:**  Write unit tests and integration tests to catch bugs early in the development process.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **IDE:**  Use a powerful IDE like Visual Studio Code, PyCharm, or Sublime Text for code editing, debugging, and testing.
*   **Virtual Environment Manager:**  Use `venv`, `virtualenv`, or `conda` to create virtual environments for isolating project dependencies.
*   **Package Manager:** Use `pip` or `conda` to install and manage Python packages.
*   **Database Client:** Use a database client like DBeaver or pgAdmin to connect to and manage your databases.

### 7.2. Build Configuration

*   **Use `requirements.txt`:** Use a `requirements.txt` file to specify your project's dependencies. Generate this file using `pip freeze > requirements.txt`.
*   **Specify Versions:** Specify exact versions of your dependencies in `requirements.txt` to ensure consistent builds across different environments.
*   **Use a Build Tool (Optional):** For more complex projects, consider using a build tool like Make or Invoke to automate build tasks.

### 7.3. Linting and Formatting

*   **Use a Linter:** Use a linter like `flake8` or `pylint` to enforce coding style and identify potential errors.
*   **Use a Formatter:** Use a code formatter like `black` or `autopep8` to automatically format your code according to PEP 8.
*   **Configure Your IDE:** Configure your IDE to automatically run the linter and formatter when you save a file.

### 7.4. Deployment Best Practices

*   **Use a Production Web Server:** Use a production-ready web server like Gunicorn or uWSGI to serve your application.
*   **Use a Reverse Proxy:** Use a reverse proxy like Nginx or Apache to handle static assets, SSL termination, and load balancing.
*   **Use a Process Manager:** Use a process manager like Supervisor or systemd to ensure that your application is always running.
*   **Monitor Your Application:** Use a monitoring tool like Prometheus or Grafana to monitor your application's performance and identify potential issues.

### 7.5. CI/CD Integration

*   **Use a CI/CD Pipeline:** Use a CI/CD pipeline to automate the process of building, testing, and deploying your application.
*   **Use a CI/CD Tool:** Use a CI/CD tool like Jenkins, GitLab CI, GitHub Actions, or CircleCI to create your CI/CD pipeline.
*   **Run Tests Automatically:** Configure your CI/CD pipeline to run tests automatically on every commit.
*   **Automate Deployment:** Automate the deployment process so that new versions of your application are deployed automatically when the tests pass.

By following these best practices, you can build robust, maintainable, and secure Bottle applications that meet the needs of your users and your organization.