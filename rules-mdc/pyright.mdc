---
description: This rule provides comprehensive best practices for using pyright and BasedPyright in Python projects, covering code organization, patterns, performance, security, testing, common pitfalls, and tooling.
globs: *.py
---
# pyright and BasedPyright Best Practices: A Comprehensive Guide

This guide provides comprehensive best practices for using pyright and BasedPyright in Python projects. These practices cover code organization, design patterns, performance optimization, security considerations, testing strategies, common pitfalls, and recommended tooling.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

*   **Flat vs. Nested:** Choose a directory structure that balances simplicity and maintainability. For small projects, a flat structure might suffice. For larger projects, a nested structure that mirrors the module hierarchy is often better. For example:

    
    my_project/
    ├── src/
    │   ├── my_module/
    │   │   ├── __init__.py
    │   │   ├── file1.py
    │   │   ├── file2.py
    │   ├── main.py
    ├── tests/
    │   ├── my_module/
    │   │   ├── test_file1.py
    │   │   ├── test_file2.py
    │   ├── conftest.py
    ├── pyrightconfig.json
    ├── pyproject.toml (or setup.py/setup.cfg)
    ├── README.md
    

*   **`src` Layout:** Use the `src` layout to separate application code from project metadata. This helps prevent accidental imports of test or configuration files.

### 1.2 File Naming Conventions

*   **Python Naming:** Follow standard Python naming conventions (PEP 8):
    *   Modules: `lowercase_with_underscores.py`
    *   Classes: `PascalCase`
    *   Functions and variables: `lowercase_with_underscores`
    *   Constants: `UPPERCASE_WITH_UNDERSCORES`
*   **Test Files:**  Name test files consistently, e.g., `test_<module_name>.py` or `<module_name>_test.py`.

### 1.3 Module Organization

*   **Cohesion:** Group related functions and classes within a single module.
*   **Coupling:** Minimize dependencies between modules to improve maintainability.
*   **`__init__.py`:** Use `__init__.py` files to define packages and control namespace imports.  Consider explicit relative imports within packages (e.g., `from . import module` instead of `import module`).

### 1.4 Component Architecture

*   **Layered Architecture:** For complex applications, consider a layered architecture (e.g., presentation, business logic, data access).  This promotes separation of concerns and testability.
*   **Microservices:** For very large projects, consider breaking the application into microservices.

### 1.5 Code Splitting Strategies

*   **By Feature:** Split code into modules or packages based on features or functionality (e.g., `auth`, `users`, `products`).
*   **By Layer:**  Split code based on architectural layers (e.g., data access, business logic, presentation). This aligns with Layered architecture.
*   **Lazy Loading:** Defer loading modules or components until they are needed. This can improve startup time and reduce memory usage. Use the `importlib` module or dynamic imports for lazy loading.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **Factory Pattern:** Use factory functions or classes to create objects, especially when the object creation logic is complex or needs to be configurable.
*   **Strategy Pattern:**  Use the strategy pattern to encapsulate different algorithms or behaviors and switch between them at runtime.  This promotes flexibility and testability.
*   **Observer Pattern:** Implement the observer pattern for event handling and decoupling components.
*   **Singleton Pattern:** Use sparingly and only when a single instance of a class is truly required. Consider dependency injection as an alternative.

### 2.2 Recommended Approaches

*   **Type Annotations:**  Embrace type annotations throughout your codebase. This is essential for effective static analysis with pyright and improves code readability and maintainability.
*   **Configuration:** Externalize configuration data (e.g., database connection strings, API keys) using environment variables or configuration files.  Use libraries like `python-dotenv` or `dynaconf`.
*   **Logging:** Implement comprehensive logging using the `logging` module.  Configure logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) appropriately.
*   **Dependency Injection:** Use dependency injection to decouple components and improve testability. Libraries like `injector` can simplify dependency injection.

### 2.3 Anti-patterns and Code Smells

*   **Ignoring Pyright Errors:**  Treat pyright errors as critical issues that need to be addressed.  Do not silence errors without understanding the underlying problem.
*   **Overuse of `Any`:**  Avoid using `Any` excessively. It defeats the purpose of static typing.  Strive to provide specific type annotations.
*   **Magic Numbers:**  Avoid hardcoding numerical values or strings directly in your code. Use named constants instead.
*   **Global State:**  Minimize the use of global variables. Global state can make code harder to understand and test.
*   **Deeply Nested Code:**  Avoid deeply nested conditional statements or loops.  Refactor complex code into smaller, more manageable functions.

### 2.4 State Management

*   **Immutability:**  Favor immutable data structures where possible. This can simplify reasoning about state and prevent unintended side effects.  Use libraries like `attrs` or `dataclasses` to create immutable classes.
*   **Centralized State:** For complex applications, consider using a centralized state management solution (e.g., using Redux-like patterns or libraries).

### 2.5 Error Handling

*   **Exceptions:** Use exceptions for exceptional situations.  Raise specific exception types that accurately describe the error.
*   **`try...except`:** Use `try...except` blocks to handle exceptions gracefully.  Avoid catching generic `Exception` unless absolutely necessary.
*   **Logging Errors:** Log exceptions with sufficient context (e.g., traceback, relevant variable values). This is crucial for debugging.
*   **Resource Management:** Use `try...finally` or the `with` statement to ensure resources are properly released, even if an exception occurs.
*   **Retry logic:** Implement retry mechanisms for network requests or other operations that may transiently fail. Use libraries like `tenacity`.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Profiling:** Use profiling tools (e.g., `cProfile`) to identify performance bottlenecks in your code.
*   **Efficient Data Structures:** Choose appropriate data structures for your tasks. For example, use sets for membership testing, and dictionaries for fast lookups.
*   **Algorithm Optimization:** Optimize algorithms to reduce time complexity. Consider using libraries like NumPy for numerical computations.
*   **Caching:** Implement caching to store frequently accessed data and avoid redundant computations. Use libraries like `functools.lru_cache` or `cachetools`.
*   **Just-In-Time (JIT) Compilation:** Explore using JIT compilers like Numba to accelerate numerical code.

### 3.2 Memory Management

*   **Generators:** Use generators to process large datasets without loading them into memory all at once.
*   **Context Managers:** Use context managers (`with` statement) to ensure that resources are properly released, preventing memory leaks.
*   **Object Reuse:** Reuse objects where possible to reduce memory allocation overhead.
*   **Avoid Circular References:** Circular references can prevent garbage collection. Use weak references (`weakref` module) to break cycles when needed.

### 3.3 Bundle Size Optimization (Applicable for web applications using pyright in the backend)

*   **Code Minification:** Minify JavaScript and CSS code to reduce bundle size.
*   **Tree Shaking:** Use tree shaking to remove unused code from your bundles.
*   **Code Splitting:** Split your code into smaller chunks that can be loaded on demand.
*   **Image Optimization:** Optimize images to reduce their file size.
*   **Compression:** Enable compression (e.g., gzip or Brotli) on your web server.

### 3.4 Lazy Loading

*   **Dynamic Imports:** Use dynamic imports (`import()`) to load modules or components on demand.
*   **Conditional Imports:** Conditionally import modules based on certain conditions.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

*   **SQL Injection:** Use parameterized queries or ORM libraries to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks. (Relevant if the backend code generates HTML output)
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection for web forms.
*   **Authentication and Authorization Flaws:** Use secure authentication and authorization mechanisms.
*   **Denial-of-Service (DoS):** Implement rate limiting and other measures to prevent DoS attacks.
*   **Dependency Vulnerabilities:** Regularly scan your dependencies for known vulnerabilities using tools like `pip-audit` or `safety`.

### 4.2 Input Validation

*   **Validate All Input:** Validate all user input, including data from forms, API requests, and command-line arguments.
*   **Whitelisting:** Use whitelisting to specify allowed characters or patterns.
*   **Data Type Validation:** Ensure that input data is of the expected type.
*   **Length Validation:** Validate the length of input strings.
*   **Range Validation:** Validate that numerical values are within the expected range.

### 4.3 Authentication and Authorization

*   **Strong Passwords:** Enforce strong password policies.
*   **Hashing:** Hash passwords securely using libraries like `bcrypt` or `argon2`.
*   **Salt:** Use a unique salt for each password.
*   **Two-Factor Authentication (2FA):** Implement 2FA for enhanced security.
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
*   **JSON Web Tokens (JWT):** Use JWTs for secure authentication and authorization in APIs.

### 4.4 Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Masking:** Mask sensitive data in logs and other non-production environments.
*   **Access Control:** Implement strict access control policies to limit access to sensitive data.
*   **Regular Backups:** Perform regular backups of your data.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **API Keys:** Protect API keys and store them securely.
*   **Rate Limiting:** Implement rate limiting to prevent abuse.
*   **Input Validation:** Validate all API requests.
*   **Output Sanitization:** Sanitize API responses to prevent XSS attacks.

## 5. Testing Approaches

### 5.1 Unit Testing

*   **`unittest` or `pytest`:** Use the `unittest` or `pytest` framework for unit testing.
*   **Test Coverage:** Aim for high test coverage (e.g., 80% or higher).
*   **Test-Driven Development (TDD):** Consider using TDD to write tests before writing code.
*   **Arrange-Act-Assert:** Structure your unit tests using the Arrange-Act-Assert pattern.
*   **Mocking and Stubbing:** Use mocking and stubbing to isolate units of code and test them in isolation.

### 5.2 Integration Testing

*   **Test Interactions:** Test the interactions between different components or modules.
*   **Database Testing:** Test the integration with databases.
*   **API Testing:** Test the integration with external APIs.
*   **Real Dependencies:** Use real dependencies (e.g., a real database) for integration tests when possible.

### 5.3 End-to-End Testing

*   **Simulate User Flows:** Simulate real user flows to test the entire application from end to end.
*   **UI Testing:** Use UI testing frameworks (e.g., Selenium, Playwright) to test the user interface.
*   **Browser Automation:** Automate browser interactions to simulate user behavior.

### 5.4 Test Organization

*   **Separate Test Directory:** Keep your tests in a separate `tests` directory.
*   **Mirror Module Structure:** Mirror the module structure in your test directory.
*   **`conftest.py`:** Use `conftest.py` to define fixtures and configuration for your tests.

### 5.5 Mocking and Stubbing

*   **`unittest.mock` or `pytest-mock`:** Use the `unittest.mock` module or the `pytest-mock` plugin for mocking and stubbing.
*   **Patching:** Use patching to replace objects or functions with mocks during testing.
*   **Context Managers:** Use context managers to manage mock objects.
*   **Side Effects:** Define side effects for mock objects to simulate different scenarios.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   **Incorrect Type Annotations:** Using incorrect or incomplete type annotations.
*   **Ignoring Pyright Errors:** Ignoring or silencing pyright errors without understanding the root cause.
*   **Overuse of `Any`:** Overusing the `Any` type, which defeats the purpose of static typing.
*   **Incorrect Configuration:** Incorrectly configuring pyright settings.
*   **Not Updating Stubs:** Failing to update stubs (`.pyi` files) when the code changes.

### 6.2 Edge Cases

*   **Dynamic Typing:** Dealing with dynamic typing features of Python.
*   **Type Inference Limitations:** Understanding the limitations of pyright's type inference capabilities.
*   **Complex Generics:** Handling complex generic types.
*   **Meta-programming:** Dealing with meta-programming techniques.

### 6.3 Version-Specific Issues

*   **Python Version Compatibility:** Ensuring compatibility with different Python versions.
*   **Pyright Version Compatibility:** Being aware of changes and bug fixes in different pyright versions.

### 6.4 Compatibility Concerns

*   **Third-Party Libraries:** Dealing with third-party libraries that may not have complete or accurate type annotations.
*   **Integration with Other Tools:** Ensuring compatibility with other tools in your development workflow.

### 6.5 Debugging Strategies

*   **Read Pyright Output Carefully:** Carefully examine pyright's error messages and warnings.
*   **Use a Debugger:** Use a debugger (e.g., `pdb` or the debugger in your IDE) to step through your code and inspect variables.
*   **Simplify the Code:** Simplify the code to isolate the source of the error.
*   **Consult Documentation:** Consult the pyright documentation and community resources.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Visual Studio Code:** VS Code with the Pylance extension (or the BasedPyright extension for BasedPyright features).
*   **pyright CLI:** The pyright command-line tool for static analysis.
*   **pre-commit:** pre-commit for automated code formatting and linting.
*   **mypy:** Consider using mypy in conjunction with pyright, as some checks can be performed by mypy that pyright does not offer.

### 7.2 Build Configuration

*   **`pyrightconfig.json` or `pyproject.toml`:** Configure pyright settings using a `pyrightconfig.json` or `pyproject.toml` file.
*   **Include and Exclude Paths:** Specify include and exclude paths to control which files are analyzed by pyright.
*   **Type Checking Mode:** Set the type checking mode to `basic`, `strict`, or `off`.
*   **Python Version:** Specify the target Python version.
*   **Enable Strict Rules:** Enable strict rules to enforce more stringent type checking.

### 7.3 Linting and Formatting

*   **`flake8` or `ruff`:** Use linters like `flake8` or `ruff` to enforce code style guidelines.
*   **`black`:** Use `black` for automatic code formatting.
*   **pre-commit Hooks:** Integrate linters and formatters into your pre-commit workflow.

### 7.4 Deployment Best Practices

*   **Containerization:** Use Docker to containerize your application.
*   **Infrastructure as Code (IaC):** Use IaC tools (e.g., Terraform, Ansible) to manage your infrastructure.
*   **Continuous Integration/Continuous Deployment (CI/CD):** Implement a CI/CD pipeline for automated testing and deployment.
*   **Monitoring:** Monitor your application's performance and health in production.

### 7.5 CI/CD Integration

*   **GitHub Actions or GitLab CI:** Use CI/CD platforms like GitHub Actions or GitLab CI.
*   **Automated Testing:** Run automated tests in your CI/CD pipeline.
*   **Static Analysis:** Run pyright and other static analysis tools in your CI/CD pipeline.
*   **Deployment Automation:** Automate the deployment process in your CI/CD pipeline.

## BasedPyright Specific Enhancements:

*   **reportUnreachable:** Reports errors on unreachable code, enhancing error detection in conditional blocks.
*   **reportAny:** Bans the `Any` type completely, encouraging more precise type annotations and reducing potential runtime errors.
*   **reportPrivateLocalImportUsage:** Restricts private imports within local code, promoting explicit re-exports and clearer module interfaces.
*   **reportImplicitRelativeImport:** Flags incorrect relative imports, preventing module loading issues in various execution contexts.
*   **reportInvalidCast:** Prevents non-overlapping casts, ensuring type safety during casting operations.
*   **reportUnsafeMultipleInheritance:** Discourages multiple inheritance from classes with constructors, reducing the risk of unexpected behavior in complex class hierarchies.
*   **Pylance Feature Re-implementations:** Enables features previously exclusive to Pylance such as import suggestions, semantic highlighting, and improved docstrings for compiled modules, offering more robust IDE support across different editors.
*   **Inline TypedDict Support:** Reintroduces the support for defining TypedDicts inline, offering convenient type annotation syntax.
*   **Better CI Integration:** Leverages GitHub Actions and GitLab CI for seamless error reporting in pull requests and merge requests.
*   **Strict Defaults:** Defaults to a stricter type checking mode (all) and assumes code can run on any operating system (All), promoting more comprehensive type analysis.

By following these best practices, you can leverage pyright and BasedPyright to improve the quality, maintainability, and security of your Python projects.