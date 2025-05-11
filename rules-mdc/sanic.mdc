---
description: This rule file outlines best practices and coding standards for developing Sanic applications, covering code organization, performance, security, testing, and common pitfalls.
globs: *.py
---
- This document outlines the best practices for developing applications using the Sanic framework. Following these guidelines will lead to more maintainable, performant, and secure applications.

## 1. Code Organization and Structure

- **Directory Structure:** Organize your project into logical directories. A recommended structure is:
  
  project_root/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py          # Main application file
  │   ├── routes.py        # Defines application routes
  │   ├── models.py        # Data models (if using an ORM)
  │   ├── views.py         # Request handlers and view logic
  │   ├── config.py        # Application configuration
  │   ├── utils.py         # Utility functions and helpers
  │   ├── middlewares.py   # Sanic middlewares
  │   ├── exceptions.py    # Custom exception definitions
  │   └── services/        # Business logic and service layers
  ├── tests/             # Unit and integration tests
  │   ├── __init__.py
  │   ├── conftest.py     # Pytest configuration file
  │   ├── unit/
  │   └── integration/
  ├── migrations/        # Database migration scripts
  ├── requirements.txt   # Project dependencies
  ├── .env               # Environment variables
  └── README.md          # Project documentation
  

- **File Naming Conventions:**
    - Use descriptive and consistent naming.
    - `routes.py`, `models.py`, `views.py`, `config.py`, etc. are good examples.
    - For service modules, use `service_name_service.py` (e.g., `user_service.py`).
    - Test files should mirror the source file names (e.g., `test_routes.py` for `routes.py`).

- **Module Organization:**
    - Keep modules focused and cohesive.
    - Avoid large, monolithic modules.
    - Group related functionality into separate modules and packages.
    - Use relative imports within the `app` package.

- **Component Architecture:**
    - Adopt a layered architecture (e.g., presentation, business logic, data access).
    - Use dependency injection to decouple components.
    - Employ service objects for complex business logic.

- **Code Splitting:**
    - Defer loading non-critical modules using dynamic imports.
    - Split large route handlers into smaller, manageable functions.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Dependency Injection:** Use dependency injection to manage dependencies and improve testability.
    - **Service Layer:** Encapsulate business logic in service objects.
    - **Repository Pattern:** Abstract data access logic using repositories.
    - **Middleware Pattern:** Use middleware for request pre-processing and response post-processing.

- **Recommended Approaches:**
    - **Configuration:** Use environment variables for configuration and load them with a library like `python-dotenv`.
    - **Database Access:** Utilize an ORM (e.g., SQLAlchemy) for interacting with databases.
    - **Asynchronous Tasks:** Use `asyncio` or a task queue (e.g., Celery) for background tasks.
    - **Response Handling:** Standardize response formats and error handling using custom exception handlers.

- **Anti-patterns:**
    - **Global State:** Avoid using global variables to store application state; use dependency injection instead.
    - **Tight Coupling:** Reduce coupling between components through interfaces and dependency injection.
    - **Long Functions:** Break down large functions into smaller, more manageable units.
    - **Ignoring Exceptions:** Always handle exceptions properly; avoid bare `except` clauses.

- **State Management:**
    - Use application context or dependency injection to manage shared state.
    - Avoid using global variables.
    - Utilize a dedicated state management library (if needed).

- **Error Handling:**
    - Use custom exception classes for specific error conditions.
    - Implement global exception handlers for uncaught exceptions.
    - Log errors with detailed context information.
    - Return informative error messages to the client.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Asynchronous Operations:** Use `async` and `await` for I/O-bound operations to prevent blocking the event loop.
    - **Connection Pooling:** Use connection pooling for database connections to reduce overhead.
    - **Caching:** Implement caching strategies to reduce database load.
    - **Gzip Compression:** Enable Gzip compression for responses to reduce network bandwidth usage.

- **Memory Management:**
    - Avoid creating unnecessary objects.
    - Use generators for large datasets to reduce memory consumption.
    - Profile your application to identify memory leaks.

- **Rendering Optimization:**
    - Use efficient template engines (e.g., Jinja2) and cache rendered templates.
    - Minimize the amount of data passed to templates.
    - Use response streaming for very large responses.

- **Bundle Size Optimization:**
    - Not directly applicable to Sanic (as it's a backend framework), but if serving static files, minimize and compress them.

- **Lazy Loading:**
    - Defer loading non-critical modules until they are needed.
    - Use lazy imports to improve startup time.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **SQL Injection:** Prevent SQL injection by using parameterized queries or an ORM.
    - **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection for state-changing requests.
    - **Authentication and Authorization Issues:** Secure your application with proper authentication and authorization mechanisms.

- **Input Validation:**
    - Validate all user input to prevent injection attacks.
    - Use data validation libraries (e.g., `marshmallow`) to enforce data types and constraints.
    - Escape special characters in user input.

- **Authentication and Authorization:**
    - Use a secure authentication protocol (e.g., OAuth 2.0, JWT).
    - Implement role-based access control (RBAC) to manage user permissions.
    - Protect sensitive endpoints with authentication and authorization middleware.

- **Data Protection:**
    - Use HTTPS to encrypt data in transit.
    - Store sensitive data (e.g., passwords) securely using hashing and salting.
    - Use encryption for data at rest.

- **Secure API Communication:**
    - Use HTTPS for all API endpoints.
    - Implement rate limiting to prevent abuse.
    - Validate API requests and responses against a schema.

## 5. Testing Approaches

- **Unit Testing:**
    - Use a testing framework like `pytest` or `unittest`.
    - Write unit tests for individual components (e.g., models, services, utilities).
    - Mock external dependencies to isolate components.

- **Integration Testing:**
    - Test the interaction between different components.
    - Use a test database for integration tests.
    - Verify that requests are handled correctly and data is persisted.

- **End-to-End Testing:**
    - Use a tool like `Selenium` or `Playwright` to simulate user interactions.
    - Test the entire application flow from the client to the database.
    - Verify that the application behaves as expected in a production-like environment.

- **Test Organization:**
    - Organize tests into separate directories (e.g., `unit`, `integration`).
    - Use descriptive test names.
    - Follow the arrange-act-assert pattern.

- **Mocking and Stubbing:**
    - Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`) to isolate components.
    - Create stubs for external dependencies that are difficult to test directly.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - **Blocking the Event Loop:** Avoid performing long-running or I/O-bound operations in the main event loop.
    - **Incorrect Use of `async` and `await`:** Ensure that all asynchronous operations are properly awaited.
    - **Not Handling Exceptions:** Always handle exceptions properly to prevent application crashes.
    - **Security Vulnerabilities:** Be aware of common security vulnerabilities and take steps to prevent them.

- **Edge Cases:**
    - **Handling Large Requests:** Properly handle large requests to prevent denial-of-service attacks.
    - **Graceful Shutdown:** Implement graceful shutdown to avoid data loss.
    - **Concurrency Issues:** Be aware of potential concurrency issues when working with shared resources.

- **Version-Specific Issues:**
    - Consult the Sanic documentation and release notes for version-specific issues and migration guides.

- **Compatibility Concerns:**
    - Ensure that your dependencies are compatible with the version of Sanic you are using.
    - Test your application with different versions of Python to ensure compatibility.

- **Debugging Strategies:**
    - Use the built-in debugger or an IDE with debugging support.
    - Log detailed information about requests and responses.
    - Use profiling tools to identify performance bottlenecks.
    - Check Sanic application and server logs when troubleshooting issues.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **IDE:** VS Code, PyCharm
    - **Package Manager:** pip, uv
    - **Virtual Environment:** venv, virtualenv
    - **Testing Framework:** pytest
    - **Linting:** flake8, pylint
    - **Formatting:** black, autopep8
    - **Debugging:** pdb, ipdb

- **Build Configuration:**
    - Use a `requirements.txt` file to manage dependencies.
    - Use a `Makefile` or `tox.ini` file to automate build tasks.
    - Use a `Dockerfile` to containerize your application.

- **Linting and Formatting:**
    - Use a linter (e.g., `flake8`, `pylint`) to enforce code style.
    - Use a formatter (e.g., `black`, `autopep8`) to automatically format your code.
    - Configure your IDE to run linters and formatters automatically.

- **Deployment Best Practices:**
    - Use a process manager like `Gunicorn` or `uvicorn` to serve your application.
    - Deploy your application behind a reverse proxy like `Nginx`.
    - Use a containerization platform like `Docker` to package and deploy your application.
    - Monitor your application using a monitoring tool like `Prometheus` or `Sentry`.

- **CI/CD Integration:**
    - Use a CI/CD platform like `GitHub Actions`, `GitLab CI`, or `Jenkins` to automate the build, test, and deployment process.
    - Configure your CI/CD pipeline to run linters, formatters, and tests.
    - Use a deployment strategy like blue-green deployment or canary deployment.

By adhering to these best practices, you can develop robust, scalable, and maintainable applications using the Sanic framework. Remember to stay updated with the latest Sanic documentation and community recommendations.