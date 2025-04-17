---
description: Comprehensive best practices and coding standards for the Tornado framework. This rule provides guidelines on code organization, performance, security, testing, and common pitfalls when developing Tornado applications.
globs: *.py
---
## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

A well-structured directory layout enhances code maintainability and scalability. Consider the following structure:


project_name/
├── app/
│   ├── __init__.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base_handler.py  # Base class for handlers
│   │   ├── home_handler.py  # Specific request handlers
│   │   ├── auth_handler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py        # Data models
│   │   ├── packet.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── authentication_service.py # Business Logic
│   │   ├── packet_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── database.py    # Database connection and setup
│   │   ├── helpers.py     # Utility functions
│   ├── config.py      # Application configuration
│   ├── application.py # Tornado Application setup
│   └── main.py        # Entry point for the application
├── tests/
│   ├── __init__.py
│   ├── test_handlers.py  # Unit tests for handlers
│   ├── test_models.py    # Unit tests for models
│   └── test_services.py  # Unit tests for services
├── static/           # Static files (CSS, JavaScript, images)
├── templates/        # HTML templates
├── docs/             # Documentation
├── requirements.txt  # Project dependencies
├── .env             # Environment variables (development)
└── README.md


### 1.2. File Naming Conventions

*   Use descriptive and consistent file names.
*   Handlers: `home_handler.py`, `user_handler.py`
*   Models: `user.py`, `product.py`
*   Utilities: `database.py`, `string_utils.py`
*   Tests: `test_handlers.py`, `test_models.py`

### 1.3. Module Organization

*   Group related functionalities into modules.
*   Keep modules focused and avoid creating overly large modules.
*   Use clear and concise names for modules.
*   Use packages (directories with `__init__.py`) to further organize modules.

### 1.4. Component Architecture

*   **MVC (Model-View-Controller):** A common architectural pattern.
    *   **Models:** Represent data and business logic.
    *   **Views:** (Templates) Display data to the user.
    *   **Controllers (Handlers):** Handle user requests and interact with models.
*   **Microservices:** Decompose the application into smaller, independent services.
*   **Hexagonal Architecture (Ports and Adapters):** Isolates the core application logic from external dependencies.

### 1.5. Code Splitting

*   **Split large handler files:** Decompose complex handlers into smaller, reusable functions or classes.
*   **Modularize functionality:** Extract common logic into separate modules or services.
*   **Lazy loading:** Import modules only when needed to reduce startup time. Avoid doing heavy imports at the top of the file. Consider `import tornado.ioloop` inside of a method if it's only used there.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Singleton:** For managing global resources like database connections.
*   **Factory:** For creating instances of objects without specifying their concrete classes.
*   **Observer:** For implementing event-driven architectures.
*   **Template Method:** For defining a skeleton algorithm in a base class and letting subclasses implement specific steps.
*   **Middleware:** Implement custom request processing logic (authentication, logging).

### 2.2. Recommended Approaches

*   **Asynchronous Operations:** Leverage Tornado's asynchronous capabilities (using `async` and `await`) for non-blocking I/O operations. This includes database queries, API calls, and file system operations.
*   **Use `IOLoop.add_callback` for thread safety:** When interacting with Tornado objects from other threads.
*   **Configuration Management:** Utilize environment variables for storing configuration settings, adhering to the Twelve-Factor App methodology.
*   **Logging:** Implement robust logging using Tornado's built-in logging capabilities.
*   **Routing:** Utilize Tornado's routing system effectively by defining URL patterns and corresponding request handlers.
*   **Authentication/Authorization:** Employ decorators like `@tornado.web.authenticated` for securing routes.
*   **Use a BaseHandler class:**  Encapsulate common logic, such as authentication checks, error handling, and template rendering.

### 2.3. Anti-patterns and Code Smells

*   **Blocking I/O in Handlers:** Performing synchronous operations within request handlers will block the I/O loop, severely impacting performance. Always use asynchronous alternatives.
*   **Ignoring Exceptions:** Catching exceptions without proper handling or logging can mask critical errors.
*   **Overly Complex Handlers:** Large, monolithic handlers are difficult to maintain and test. Break them down into smaller, focused functions or classes.
*   **Hardcoding Configuration:** Embedding configuration values directly in the code makes it difficult to manage and deploy the application across different environments. Use environment variables or configuration files.
*   **Global State:** Over-reliance on global variables can lead to unpredictable behavior and make the code harder to reason about. Use dependency injection or other techniques to manage state explicitly.
*   **Ignoring Thread Safety:** Tornado applications are single-threaded, but interactions with external resources might not be. Using `IOLoop.run_in_executor` is key to thread safety. If using WSGI containers, this becomes extremely important.

### 2.4. State Management

*   **Stateless Handlers:** Design handlers to be stateless to improve scalability and fault tolerance.
*   **Session Management:** Use signed cookies or external session stores (Redis, Memcached) for managing user sessions.
*   **Caching:** Implement caching mechanisms (in-memory, Redis, Memcached) to reduce database load and improve response times.
*   **Avoid Global Variables:** Minimize the use of global variables to prevent unintended side effects.

### 2.5. Error Handling

*   **Centralized Error Handling:** Create a custom error handler (e.g., a base handler with error handling logic) to handle exceptions consistently across the application.
*   **Logging:** Log all exceptions and errors with sufficient context (request details, user information).
*   **Custom Error Pages:** Provide user-friendly error pages for common HTTP errors (404, 500).
*   **Exception Propagation:** Allow exceptions to propagate to the Tornado error handler to ensure proper logging and handling.
*   **Use `try...except` Blocks:** Wrap potentially failing code blocks in `try...except` blocks to gracefully handle errors.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Asynchronous I/O:** Use asynchronous I/O operations to avoid blocking the I/O loop.
*   **Connection Pooling:** Use connection pooling for database connections to reduce connection overhead.  Libraries like SQLAlchemy provide this.
*   **Caching:** Implement caching strategies to reduce database load and improve response times. Consider using `functools.lru_cache` for frequently called, pure functions.
*   **Gzip Compression:** Enable Gzip compression for responses to reduce bandwidth usage.
*   **Optimize Database Queries:** Ensure that database queries are efficient and properly indexed.
*   **Minimize External Dependencies:** Reduce the number of external dependencies to minimize startup time and improve performance.
*   **WebSockets**: Implement websockets using Tornado's native support for persistent connections to offload work from HTTP handlers.

### 3.2. Memory Management

*   **Avoid Memory Leaks:** Be mindful of memory leaks, especially when dealing with long-lived connections or background tasks. Use tools like `memory_profiler` to identify and fix memory leaks.
*   **Use Generators:** Use generators for processing large datasets to avoid loading the entire dataset into memory at once.
*   **Object Pooling:** Use object pooling for frequently created objects to reduce object creation overhead.
*   **Limit Object Sizes:** Avoid creating excessively large objects that can consume significant memory.

### 3.3. Rendering Optimization (If applicable)

*   **Template Caching:** Cache compiled templates to reduce rendering overhead. Tornado automatically caches templates by default.
*   **Minimize Template Logic:** Keep template logic simple and move complex calculations to the handler.
*   **Use Template Inheritance:** Use template inheritance to reuse common template elements and reduce duplication.
*   **Optimize Static Files:** Minimize, compress, and cache static files (CSS, JavaScript, images).

### 3.4. Bundle Size Optimization (If applicable)

*   **Minify CSS and JavaScript:** Use tools like UglifyJS or CSSNano to minify CSS and JavaScript files.
*   **Combine CSS and JavaScript:** Combine multiple CSS and JavaScript files into fewer files to reduce HTTP requests.
*   **Use a CDN:** Serve static files from a Content Delivery Network (CDN) to improve loading times.

### 3.5. Lazy Loading

*   **Lazy-load Modules:** Import modules only when needed to reduce startup time.
*   **Lazy-load Images:** Load images only when they are visible in the viewport.
*   **Lazy-load Data:** Fetch data only when it is needed.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
*   **SQL Injection:** Use parameterized queries or an ORM to prevent SQL injection attacks.
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection to prevent malicious websites from performing unauthorized actions on behalf of logged-in users. Tornado includes CSRF protection features.
*   **Authentication and Authorization Flaws:** Implement robust authentication and authorization mechanisms to prevent unauthorized access to resources.
*   **Session Hijacking:** Use secure cookies (HTTPOnly, Secure) and session management techniques to prevent session hijacking.
*   **Denial of Service (DoS):** Implement rate limiting and other measures to prevent DoS attacks.

### 4.2. Input Validation

*   **Validate All Input:** Validate all user input to prevent malicious data from entering the application.
*   **Use Whitelisting:** Use whitelisting to define the allowed characters and formats for input fields.
*   **Sanitize Input:** Sanitize input to remove or escape potentially dangerous characters.
*   **Escape Output:** Escape output to prevent XSS attacks.

### 4.3. Authentication and Authorization

*   **Use Strong Passwords:** Enforce strong password policies and use a secure hashing algorithm (bcrypt, Argon2) to store passwords.
*   **Implement Two-Factor Authentication (2FA):** Use 2FA to add an extra layer of security to user accounts.
*   **Use Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
*   **Principle of Least Privilege:** Grant users only the minimum privileges required to perform their tasks.
*   **JSON Web Tokens (JWT):** Use JWT for stateless authentication.

### 4.4. Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data at rest and in transit.
*   **Use HTTPS:** Use HTTPS to encrypt communication between the client and server.
*   **Store Secrets Securely:** Store secrets (API keys, database passwords) securely using environment variables or a secrets management system.
*   **Regularly Rotate Secrets:** Rotate secrets regularly to reduce the risk of compromise.

### 4.5. Secure API Communication

*   **Use HTTPS:** Enforce HTTPS for all API endpoints.
*   **Implement API Authentication:** Use API keys, JWT, or OAuth 2.0 to authenticate API clients.
*   **Rate Limiting:** Implement rate limiting to prevent abuse and DoS attacks.
*   **Input Validation:** Validate all API input to prevent malicious data from entering the application.
*   **Output Sanitization:** Sanitize API output to prevent XSS attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Individual Components:** Unit tests should focus on testing individual components (handlers, models, services) in isolation.
*   **Use Mocking and Stubbing:** Use mocking and stubbing to isolate the component under test from its dependencies.
*   **Test Edge Cases:** Test edge cases and boundary conditions to ensure that the component handles unexpected inputs correctly.
*   **Follow Arrange-Act-Assert Pattern:** Structure unit tests using the Arrange-Act-Assert pattern.
*   **Use `unittest` or `pytest`:** Python's built-in `unittest` module or the popular `pytest` framework are commonly used.

### 5.2. Integration Testing

*   **Test Interactions Between Components:** Integration tests should focus on testing the interactions between different components of the application.
*   **Use a Test Database:** Use a separate test database to avoid polluting the production database.
*   **Test API Endpoints:** Test API endpoints to ensure that they function correctly and return the expected results.
*   **Test Database Interactions:** Test database interactions to ensure that data is being read and written correctly.

### 5.3. End-to-End Testing

*   **Test the Entire Application:** End-to-end tests should test the entire application from the user's perspective.
*   **Use a Testing Framework:** Use a testing framework like Selenium or Cypress to automate end-to-end tests.
*   **Test User Flows:** Test common user flows to ensure that the application functions correctly.

### 5.4. Test Organization

*   **Organize Tests by Module:** Organize tests into separate files or directories for each module.
*   **Use Descriptive Test Names:** Use descriptive test names to make it clear what each test is testing.
*   **Keep Tests Independent:** Ensure that tests are independent of each other and can be run in any order.

### 5.5. Mocking and Stubbing

*   **Use Mocking Libraries:** Use mocking libraries like `unittest.mock` or `pytest-mock` to create mock objects and stubs.
*   **Mock External Dependencies:** Mock external dependencies (databases, APIs) to isolate the component under test.
*   **Stub Complex Logic:** Stub complex logic to simplify the test and focus on the specific behavior being tested.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Blocking the I/O Loop:** Performing synchronous operations in handlers.
*   **Not Handling Exceptions Properly:** Ignoring or mishandling exceptions.
*   **Incorrectly Using `async` and `await`:**  Forgetting to `await` asynchronous calls.
*   **Not Understanding Thread Safety:** Improperly accessing Tornado objects from threads.
*   **Over-complicating handlers:** Making handlers too long and hard to understand.

### 6.2. Edge Cases

*   **Handling Long-lived Connections:**  Managing WebSocket connections and preventing memory leaks.
*   **Dealing with Slow Clients:**  Implementing timeouts and connection limits.
*   **Handling Unexpected Input:**  Validating and sanitizing user input.
*   **Dealing with Network Errors:** Implementing retry logic and error handling.

### 6.3. Version-Specific Issues

*   **Compatibility with Python Versions:**  Ensuring compatibility with supported Python versions.
*   **API Changes:** Being aware of API changes between Tornado versions.
*   **Deprecated Features:**  Avoiding the use of deprecated features.

### 6.4. Compatibility Concerns

*   **WSGI Compatibility:** Understanding the limitations of WSGI integration.
*   **Asynchronous Libraries:**  Ensuring compatibility with other asynchronous libraries.
*   **Database Drivers:**  Using asynchronous database drivers compatible with Tornado.

### 6.5. Debugging Strategies

*   **Logging:** Use extensive logging to track the flow of execution and identify errors.
*   **Debugging Tools:** Use debugging tools like `pdb` or IDE debuggers to step through the code and inspect variables.
*   **Profiling:** Use profiling tools to identify performance bottlenecks.
*   **Monitoring:** Use monitoring tools to track the application's health and performance in production.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **IDE:** VS Code, PyCharm
*   **Virtual Environment:** `venv`, `virtualenvwrapper`, `conda`
*   **Package Manager:** `pip`, `uv`
*   **Debugging Tools:** `pdb`, IDE debuggers
*   **Profiling Tools:** `cProfile`, `memory_profiler`
*   **Linting and Formatting:** `flake8`, `pylint`, `black`
*   **Testing Frameworks:** `unittest`, `pytest`

### 7.2. Build Configuration

*   **Use `requirements.txt`:**  Specify project dependencies in a `requirements.txt` file.
*   **Use `setup.py` or `pyproject.toml`:** For packaging and distributing the application.
*   **Use a Build System:** Consider using a build system like `make` or `tox` to automate build tasks.

### 7.3. Linting and Formatting

*   **Use a Linter:** Use a linter like `flake8` or `pylint` to enforce coding standards and identify potential errors.
*   **Use a Formatter:** Use a formatter like `black` to automatically format the code according to PEP 8.
*   **Configure Editor:** Configure the editor to automatically run the linter and formatter on save.

### 7.4. Deployment

*   **Use a Production-Ready Server:**  Deploy the application on a production-ready server like Gunicorn or uWSGI.
*   **Use a Reverse Proxy:** Use a reverse proxy like Nginx or Apache to handle SSL termination, load balancing, and caching.
*   **Use a Process Manager:** Use a process manager like Supervisor or systemd to manage the application process.
*   **Use a Containerization Technology:** Consider using containerization technologies like Docker to package and deploy the application.

### 7.5. CI/CD Integration

*   **Use a CI/CD System:** Use a CI/CD system like Jenkins, GitLab CI, GitHub Actions, or CircleCI to automate the build, test, and deployment processes.
*   **Automate Testing:** Automate unit, integration, and end-to-end tests as part of the CI/CD pipeline.
*   **Automate Deployment:** Automate the deployment process to reduce the risk of human error.
*   **Implement Rollback Strategy:** Implement a rollback strategy to quickly revert to a previous version in case of errors.