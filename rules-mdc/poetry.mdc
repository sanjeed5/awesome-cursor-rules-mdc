---
description: This rule set provides comprehensive guidance on best practices for using Poetry in Python projects, including dependency management, project structure, and coding standards. It covers various aspects such as code organization, performance considerations, security, testing, and tooling.
globs: *.py
---
# Poetry Python Library Best Practices

This document outlines best practices for utilizing Poetry, a dependency management and packaging tool for Python. Following these guidelines will result in more maintainable, scalable, and robust Python projects.

## 1. Code Organization and Structure

A well-organized project structure is critical for maintainability and collaboration. Poetry provides the tools to enforce a standardized structure.

### 1.1 Directory Structure Best Practices

*   **Root Directory:** The root directory contains the `pyproject.toml` file, `README.md`, and potentially other top-level configuration files (e.g., `.gitignore`, `mypy.ini`).
*   **Source Directory:** The main source code should reside in a dedicated directory. The typical layout is to place your package in a top-level directory, e.g., `my_package/`, or within a `src/` directory (e.g., `src/my_package/`). Using `src/` allows for cleaner separation of concerns, facilitating testing and package building.
    *   **Choosing Between Flat and `src` Layouts:**
        *   **Flat Layout (default):** `my_project/my_package/__init__.py`
        *   **`src` Layout:** `my_project/src/my_package/__init__.py`. The `src` layout avoids potential import ambiguity when running scripts from the project root and provides a clearer separation between project-level files and package code.
*   **Tests Directory:** Tests should be in a `tests/` directory, mirroring the source structure. This helps in easily locating and running tests. Include an `__init__.py` in the directory to make it a package.
*   **Docs Directory (Optional):** If the project requires extensive documentation, a `docs/` directory can be included to house documentation files.
*   **Examples Directory (Optional):** Provides example usage scenarios to help end-users easily start using your library.

Example directory structure (using the `src` layout):


my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore


### 1.2 File Naming Conventions

*   **Python Files:** Use lowercase names with underscores (snake_case) for Python files (e.g., `my_module.py`, `data_processing.py`).
*   **Test Files:**  Test files should mirror the module being tested, prefixed with `test_`.  For example, `test_my_module.py` tests `my_module.py`.
*   **`__init__.py`:**  Use an empty `__init__.py` file or add initialization logic if needed. The presence of `__init__.py` makes the directory a Python package.
*   **Configuration Files:**  Use descriptive names for configuration files (e.g., `config.py`, `settings.toml`).

### 1.3 Module Organization

*   **Grouping Functionality:** Group related functions and classes into modules. Each module should represent a specific functionality or component.
*   **Avoiding Circular Dependencies:**  Design modules to minimize dependencies on each other. Circular dependencies can lead to import errors and make code harder to understand.
*   **Explicit Imports:** Use explicit imports (e.g., `from my_package.module1 import function_x`) instead of wildcard imports (`from my_package.module1 import *`). Explicit imports improve code readability and reduce namespace pollution.
*   **Relative Imports (Where Appropriate):** Within a package, use relative imports (e.g., `from . import module2` or `from .. import another_package`) to reference other modules in the same package. This makes the package more self-contained.

### 1.4 Component Architecture

*   **Layered Architecture:** For larger projects, consider a layered architecture (e.g., presentation layer, business logic layer, data access layer). This promotes separation of concerns and makes the codebase more modular.
*   **Microservices (If Applicable):** If the project is complex, consider splitting it into microservices, each managed as a separate Poetry project.
*   **Dependency Injection:**  Employ dependency injection to decouple components.  This enhances testability and flexibility.
*   **Interfaces:** Define clear interfaces between components. This allows for easier swapping of implementations without affecting other parts of the system.

### 1.5 Code Splitting Strategies

*   **By Functionality:** Split code into modules based on functionality (e.g., `data_processing.py`, `api_client.py`, `ui_components.py`).
*   **By Component:**  If using a component-based architecture, split code into modules representing individual components.
*   **Lazy Loading:** Use lazy loading for modules that are not immediately required at startup. This can improve application startup time.
*   **Dynamic Imports:** Use dynamic imports (e.g., `importlib.import_module`) for optional dependencies or features. Handle potential `ImportError` exceptions gracefully.

## 2. Common Patterns and Anti-patterns

Understanding common patterns and anti-patterns is essential for writing clean, efficient, and maintainable code.

### 2.1 Design Patterns

*   **Factory Pattern:** Use factory patterns to create objects without specifying their concrete classes. This promotes loose coupling and makes it easier to switch between different implementations.
*   **Strategy Pattern:** Use strategy patterns to define a family of algorithms and make them interchangeable at runtime. This allows for flexible algorithm selection based on different conditions.
*   **Observer Pattern:**  Implement the observer pattern for event handling and decoupled communication between components.
*   **Singleton Pattern (Use Sparingly):** Avoid overuse of the singleton pattern.  If used, ensure thread safety.

### 2.2 Recommended Approaches for Common Tasks

*   **Dependency Injection:** Using dependency injection to manage dependencies improves testability and flexibility.
*   **Configuration Management:** Using configuration files (e.g., TOML, YAML) to store application settings simplifies management and deployment.
*   **Logging:** Using a logging library (e.g., `logging`) for application logging enables easy debugging and monitoring.
*   **Data Validation:** Use a data validation library (e.g., `pydantic`, `marshmallow`) to validate input data and ensure data integrity.
*   **API Clients:** Encapsulate API interactions within dedicated modules, enabling reuse and reducing code duplication. Use libraries like `requests` or `httpx` for HTTP requests. Consider asynchronous clients with `asyncio` for enhanced performance in I/O bound scenarios.

### 2.3 Anti-patterns and Code Smells

*   **Global State:** Avoid using global variables excessively. Global state makes code harder to reason about and test.
*   **Magic Numbers:** Avoid using magic numbers in code. Define constants with descriptive names instead.
*   **Duplicated Code:** Avoid code duplication. Extract common code into functions or classes.
*   **Long Functions:** Avoid writing long functions. Break them down into smaller, more manageable functions.
*   **Over-Engineering:** Don't over-engineer solutions. Keep the code simple and focused on the problem at hand.
*   **Ignoring Exceptions:** Never leave `except:` blocks empty. Always handle exceptions appropriately.
*   **Wildcard Imports:** Avoid wildcard imports (`from module import *`). They pollute the namespace and make it harder to understand where symbols come from.

### 2.4 State Management

*   **Stateless Components:** Prefer stateless components whenever possible. Stateless components are easier to test and reason about.
*   **Immutable Data:** Use immutable data structures (e.g., `namedtuple`, `dataclasses` with `frozen=True`) to avoid accidental state modifications.
*   **Explicit State:** Make state transitions explicit and well-defined. Use a state management library (e.g., `transitions`) for complex state machines.
*   **Centralized State (If Needed):** For complex applications, consider using a centralized state management solution (e.g., Redux-like pattern). However, be mindful of the increased complexity.

### 2.5 Error Handling

*   **Specific Exceptions:** Catch specific exceptions instead of using a broad `except Exception:`.
*   **Logging Errors:** Log error messages with relevant context (e.g., function name, input arguments).
*   **Raising Exceptions:** Raise exceptions when errors occur to signal failures to the caller.
*   **Context Managers:** Use context managers (`with` statement) to ensure proper resource management (e.g., closing files, releasing locks).
*   **Retry Mechanism:**  Implement a retry mechanism for transient errors (e.g., network errors, API rate limits). Use libraries like `tenacity` to easily implement retry logic with customizable backoff strategies.
*   **Error Boundaries:**  Design error boundaries in your application that prevent errors in one part of the system from crashing the entire application.  This can be achieved through exception handling at appropriate levels and the use of circuit breaker patterns.

## 3. Performance Considerations

Optimizing performance is crucial for providing a smooth user experience and reducing resource consumption.

### 3.1 Optimization Techniques

*   **Profiling:** Use profiling tools (e.g., `cProfile`, `line_profiler`) to identify performance bottlenecks.
*   **Algorithm Optimization:** Choose efficient algorithms and data structures for critical tasks.
*   **Caching:** Implement caching for frequently accessed data. Use libraries like `cachetools` or `diskcache` for caching.
*   **Asynchronous Programming:** Use asynchronous programming (`asyncio`) for I/O-bound operations (e.g., network requests, file I/O). This can significantly improve concurrency and responsiveness.
*   **Multiprocessing:** Use multiprocessing (`multiprocessing`) for CPU-bound operations to take advantage of multiple cores. However, be mindful of the increased complexity of inter-process communication.
*   **Vectorization (NumPy):** Use NumPy for vectorized operations on arrays. Vectorization can significantly speed up numerical computations.
*   **Just-In-Time (JIT) Compilation:** Consider using a JIT compiler (e.g., Numba) for computationally intensive functions. JIT compilation can significantly improve performance.

### 3.2 Memory Management

*   **Garbage Collection:** Understand how Python's garbage collection works. Avoid creating circular references that can prevent garbage collection.
*   **Generators and Iterators:** Use generators and iterators for processing large datasets. Generators and iterators consume memory on demand, avoiding the need to load the entire dataset into memory.
*   **Data Structure Efficiency:** Choose efficient data structures for storing data. Consider using specialized data structures (e.g., `collections.deque`, `heapq`) when appropriate.
*   **Memory Profiling:** Use memory profiling tools (e.g., `memory_profiler`) to identify memory leaks and excessive memory consumption.
*   **Object Reuse:**  Reuse objects when possible to minimize object creation and destruction overhead. This can be particularly relevant for long-running processes.

### 3.3 Bundle Size Optimization (If Applicable)

*   **Tree Shaking:** If using a bundler, enable tree shaking to remove unused code from the bundle.
*   **Code Minification:**  Minify code to reduce the bundle size. Tools like `Terser` can be used for code minification.
*   **Image Optimization:** Optimize images to reduce their size without compromising quality.
*   **Dependency Optimization:**  Analyze dependencies to identify and remove unnecessary dependencies. Tools like `depcheck` can help with dependency optimization.
*   **Compression:**  Use compression algorithms (e.g., gzip, Brotli) to compress the bundle before deployment. This reduces the download size and improves loading time.

### 3.4 Lazy Loading

*   **Module Lazy Loading:** Use dynamic imports to lazy load modules that are not immediately required.
*   **Data Lazy Loading:** Use generators or iterators to lazy load data from databases or files.
*   **Component Lazy Loading:**  Load UI components or application features on demand, improving initial startup time.

## 4. Security Best Practices

Security should be a top priority in any software project. Following security best practices can help protect your application from vulnerabilities.

### 4.1 Common Vulnerabilities and Prevention

*   **Injection Attacks:** Prevent injection attacks (e.g., SQL injection, command injection) by validating and sanitizing user inputs. Use parameterized queries or ORMs to prevent SQL injection.
*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by escaping user-generated content before displaying it in the browser.
*   **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by using anti-CSRF tokens.
*   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms to protect sensitive resources.
*   **Dependency Vulnerabilities:** Regularly scan dependencies for vulnerabilities and update them to the latest versions. Tools like `Safety` or Snyk can be used to detect and remediate dependency vulnerabilities.
*   **Sensitive Data Exposure:** Avoid storing sensitive data (e.g., passwords, API keys) in plaintext. Encrypt sensitive data at rest and in transit.
*   **Denial of Service (DoS):**  Implement rate limiting and input validation to prevent DoS attacks.

### 4.2 Input Validation

*   **Validate All Inputs:** Validate all user inputs, including form data, URL parameters, and API requests.
*   **Use a Validation Library:** Use a validation library (e.g., `pydantic`, `marshmallow`) to define validation schemas and validate data against them.
*   **Sanitize Inputs:** Sanitize inputs to remove potentially harmful characters or code.
*   **Limit Input Length:** Limit the length of input fields to prevent buffer overflows.
*   **Check Data Types:** Verify that input data is of the expected data type.

### 4.3 Authentication and Authorization

*   **Strong Passwords:** Enforce strong password policies (e.g., minimum length, complexity requirements).
*   **Hashing Passwords:** Hash passwords using a strong hashing algorithm (e.g., bcrypt, Argon2) and store them securely.
*   **Salting Passwords:** Salt passwords to prevent rainbow table attacks.
*   **Two-Factor Authentication (2FA):** Implement two-factor authentication for increased security.
*   **Role-Based Access Control (RBAC):** Use role-based access control to restrict access to sensitive resources based on user roles.
*   **JSON Web Tokens (JWT):** Use JSON Web Tokens for stateless authentication and authorization.
*   **OAuth 2.0:**  Use OAuth 2.0 for secure delegation of authorization.

### 4.4 Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit using strong encryption algorithms (e.g., AES, TLS).
*   **Data Masking:** Mask sensitive data when displaying it to users.
*   **Data Anonymization:** Anonymize data when it is not necessary to identify individual users.
*   **Regular Backups:** Perform regular backups of data to prevent data loss.
*   **Secure Storage:** Store data in a secure location with appropriate access controls.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication to encrypt data in transit.
*   **API Keys:** Protect API keys and store them securely. Rotate API keys regularly.
*   **Rate Limiting:** Implement rate limiting to prevent API abuse.
*   **Input Validation:** Validate all API requests to prevent injection attacks.
*   **Output Sanitization:** Sanitize API responses to prevent XSS attacks.
*   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms for API access.
*   **CORS:** Configure Cross-Origin Resource Sharing (CORS) to restrict API access to authorized domains.

## 5. Testing Approaches

Thorough testing is essential for ensuring the quality and reliability of software.

### 5.1 Unit Testing

*   **Test-Driven Development (TDD):** Consider using test-driven development (TDD) to write tests before writing code.
*   **Test Coverage:** Aim for high test coverage to ensure that all parts of the code are tested.
*   **Test Independence:** Write tests that are independent of each other. Avoid sharing state between tests.
*   **Assertion Libraries:** Use assertion libraries (e.g., `unittest`, `pytest`) to write clear and concise assertions.
*   **Edge Cases:** Test edge cases and boundary conditions to ensure that the code handles them correctly.

### 5.2 Integration Testing

*   **Component Integration:** Test the integration between different components of the system.
*   **API Integration:** Test the integration with external APIs.
*   **Database Integration:** Test the integration with databases.
*   **Mock External Dependencies:** Use mocks or stubs to isolate the system under test from external dependencies.
*   **Real-World Scenarios:** Design integration tests to simulate real-world usage scenarios.

### 5.3 End-to-End Testing

*   **User Interface Testing:** Test the user interface to ensure that it is working as expected.
*   **User Flows:** Test complete user flows to ensure that the system is functioning correctly from start to finish.
*   **Automated Testing:** Automate end-to-end tests to ensure that they are run regularly.
*   **Browser Automation:** Use browser automation tools (e.g., Selenium, Playwright) to automate user interface tests.
*   **Continuous Integration:** Integrate end-to-end tests into the continuous integration pipeline.

### 5.4 Test Organization

*   **Mirror Source Structure:** Organize tests in a directory structure that mirrors the source code structure.
*   **Descriptive Test Names:** Use descriptive names for test functions and test classes.
*   **Grouping Tests:** Group related tests into test classes or modules.
*   **Test Suites:** Create test suites to group related tests together.
*   **Test Runners:** Use test runners (e.g., `unittest`, `pytest`) to discover and run tests.

### 5.5 Mocking and Stubbing

*   **Mock External Dependencies:** Use mocks to simulate the behavior of external dependencies (e.g., databases, APIs).
*   **Stub Internal Dependencies:** Use stubs to replace internal dependencies with simpler implementations.
*   **Isolate the System Under Test:** Use mocks and stubs to isolate the system under test from external factors.
*   **Verify Interactions:** Use mocks to verify that the system under test is interacting with its dependencies in the expected way.
*   **Mocking Libraries:** Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`) to create mocks and stubs.

## 6. Common Pitfalls and Gotchas

Being aware of common pitfalls can save time and prevent headaches.

### 6.1 Frequent Mistakes

*   **Forgetting to Update `poetry.lock`:** Failing to update the `poetry.lock` file after adding or updating dependencies can lead to inconsistent environments.
*   **Using Wildcard Dependencies:** Using wildcard dependencies can lead to unexpected updates that break the code.
*   **Ignoring Virtual Environments:** Not using virtual environments can lead to conflicts between different projects.
*   **Committing Virtual Environment Directories:** Committing the virtual environment directory to version control is unnecessary and can lead to problems.
*   **Using Incompatible Python Versions:** Using incompatible Python versions can lead to errors.

### 6.2 Edge Cases

*   **Platform-Specific Dependencies:** Handling platform-specific dependencies requires careful consideration of package availability and compatibility.
*   **Optional Dependencies:** Managing optional dependencies that may not be available on all systems.
*   **Conflicting Dependencies:** Resolving conflicts between dependencies with overlapping requirements can be challenging.
*   **Cyclic Dependencies:** Cyclic dependencies can lead to import errors and make code harder to understand.
*   **Large Datasets:** Handling large datasets requires careful consideration of memory management and performance.

### 6.3 Version-Specific Issues

*   **Dependency Compatibility:** Ensuring that dependencies are compatible with the target Python version and each other.
*   **Deprecated Features:** Handling deprecated features and migrating to newer versions of dependencies.
*   **Security Updates:** Staying up-to-date with security updates for Poetry and its dependencies.

### 6.4 Compatibility Concerns

*   **Operating System Compatibility:** Ensuring that the code works correctly on different operating systems.
*   **Python Interpreter Compatibility:** Ensuring that the code works correctly with different Python interpreters (e.g., CPython, PyPy).
*   **Package Compatibility:** Ensuring that packages are compatible with each other.
*   **Hardware Compatibility:** Considering hardware compatibility, especially when dealing with platform-specific dependencies or performance-critical code.
*   **Backward Compatibility:** Striving to maintain backward compatibility to minimize disruption for existing users when introducing new features or changes.

### 6.5 Debugging Strategies

*   **Logging:** Using logging to track the flow of execution and identify errors.
*   **Debuggers:** Using debuggers (e.g., `pdb`, `ipdb`) to step through code and inspect variables.
*   **Unit Tests:** Writing unit tests to isolate and test individual components.
*   **Print Statements (Use Sparingly):** Using print statements to display the values of variables (use sparingly and remove them before committing code).
*   **Profiling:** Using profiling tools to identify performance bottlenecks.
*   **Remote Debugging:**  Utilize remote debugging techniques to debug code running on remote servers or containers. This can be particularly helpful for cloud deployments.
*   **Post-Mortem Debugging:**  Leverage Python's post-mortem debugging capabilities to analyze errors after they have occurred. This can provide valuable insights into the root cause of the problem.

## 7. Tooling and Environment

Using the right tools and configuring the environment correctly can significantly improve developer productivity.

### 7.1 Recommended Development Tools

*   **Integrated Development Environment (IDE):** Use an IDE (e.g., VS Code, PyCharm) with Python support for code completion, syntax highlighting, and debugging.
*   **Linters:** Use linters (e.g., `flake8`, `pylint`) to enforce code style and identify potential errors.
*   **Formatters:** Use formatters (e.g., `black`, `autopep8`) to automatically format code according to style guidelines.
*   **Type Checkers:** Use type checkers (e.g., `mypy`) to catch type errors early on.
*   **Version Control System (VCS):** Use a version control system (e.g., Git) to track changes to the code.

### 7.2 Build Configuration

*   **`pyproject.toml`:**  The cornerstone of Poetry projects, this file defines project metadata, dependencies, and build configuration.
*   **`.gitignore`:**  Excludes unnecessary files and directories (e.g., virtual environment directories, build artifacts) from version control.
*   **`Makefile` (Optional):** Use a `Makefile` to automate common tasks (e.g., running tests, building documentation).
*   **Dockerfiles:** Use Dockerfiles to create container images for deployment. This ensures consistent environments across different systems.
*   **Configuration Files:** Use configuration files (e.g., TOML, YAML) to store application settings and environment variables.

### 7.3 Linting and Formatting

*   **`flake8` Configuration:** Configure `flake8` to enforce coding style guidelines and identify potential errors. Configure in `pyproject.toml` if the tool supports it, otherwise use a separate config file.
*   **`black` Configuration:** Configure `black` to automatically format code according to style guidelines. Configure in `pyproject.toml`.
*   **`isort` Configuration:** Configure `isort` to automatically sort imports according to style guidelines. Configure in `pyproject.toml`.
*   **Pre-commit Hooks:** Use pre-commit hooks to automatically run linters and formatters before committing code. This helps to maintain code quality and consistency.

### 7.4 Deployment

*   **Containerization:** Containerize the application using Docker to ensure consistent environments across different systems.
*   **Cloud Platforms:** Deploy the application to a cloud platform (e.g., AWS, Azure, Google Cloud) for scalability and reliability.
*   **Continuous Delivery:** Use continuous delivery pipelines to automate the deployment process.
*   **Configuration Management:** Use configuration management tools (e.g., Ansible, Chef, Puppet) to manage infrastructure configuration.
*   **Monitoring:** Implement monitoring to track the performance and health of the application.

### 7.5 CI/CD Integration

*   **Continuous Integration (CI):** Integrate the code with a continuous integration system (e.g., Jenkins, GitLab CI, GitHub Actions) to automate testing and building.
*   **Continuous Delivery (CD):** Integrate the CI system with a continuous delivery system to automate the deployment process.
*   **Automated Testing:** Automate unit tests, integration tests, and end-to-end tests in the CI/CD pipeline.
*   **Automated Code Analysis:** Integrate code analysis tools (e.g., linters, formatters, type checkers) into the CI/CD pipeline.
*   **Automated Deployment:** Automate the deployment process to ensure that code is deployed consistently and reliably.

By adhering to these best practices, developers can create Python projects with Poetry that are more maintainable, scalable, secure, and reliable.