---
description: This rule file provides comprehensive best practices for using Pylint to ensure high-quality, maintainable, and secure Python code. It covers code organization, common patterns, performance, security, testing, and tooling.
globs: *.py
---
# Pylint Best Practices: A Comprehensive Guide

This document outlines the best practices for using Pylint, a widely used static code analysis tool for Python. Following these guidelines will help you write cleaner, more maintainable, and secure code.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

*   **Project Root:** Contains core project files (`README.md`, `LICENSE`, `.gitignore`, `setup.py` or `pyproject.toml`, `.pylintrc`).
*   **Source Directory (`src/` or project_name/`):** Holds the main application code.
    *   Organize code into modules and subpackages based on functionality (e.g., `src/models`, `src/views`, `src/controllers`, `src/utils`).
    *   Use clear and descriptive names for modules and packages.
*   **Tests Directory (`tests/`):** Contains unit, integration, and end-to-end tests.
    *   Mirror the source directory structure for easier navigation (e.g., `tests/models`, `tests/views`).
*   **Docs Directory (`docs/`):** Stores project documentation (using Sphinx, MkDocs, etc.).
*   **Scripts Directory (`scripts/`):** Contains utility scripts for development, deployment, etc.
*   **Configuration Directory (`config/`):** Stores configuration files for different environments (development, staging, production).

Example:


my_project/
├── .gitignore
├── .pylintrc
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── views/
│   │   ├── __init__.py
│   │   ├── user_view.py
│   │   └── item_view.py
│   └── utils/
│       ├── __init__.py
│       └── helper_functions.py
├── tests/
│   ├── __init__.py
│   ├── models/
│   │   ├── test_user.py
│   │   └── test_item.py
│   └── views/
│       ├── __init__.py
│       └── test_user_view.py
└── docs/
    └── ...


### 1.2 File Naming Conventions

*   Use lowercase with underscores for module and package names (e.g., `user_model.py`, `data_processing/`).
*   Class names should use CamelCase (e.g., `UserModel`).
*   Constants should be in uppercase with underscores (e.g., `MAX_USERS`).
*   Test files should be named `test_*.py`.

### 1.3 Module Organization

*   Each module should have a single, well-defined responsibility.
*   Avoid circular dependencies between modules.
*   Use `__init__.py` files to define packages and control imports.
*   Keep modules relatively small (e.g., less than 500 lines of code).
*   Follow the "Explicit is better than implicit" principle. Use explicit imports rather than `from module import *`.

### 1.4 Component Architecture

*   **Layered Architecture:** Separate concerns into distinct layers (e.g., presentation, business logic, data access).
*   **Microservices Architecture:** Decompose the application into small, independent services.
*   **Hexagonal Architecture (Ports and Adapters):** Decouple the core logic from external dependencies.
*   **MVC (Model-View-Controller):** A common pattern, separating data (Model), presentation (View), and user input handling (Controller).

Choose an architecture that suits the complexity and scale of your project. Ensure that pylint can analyze each component effectively by providing clear interfaces and minimal inter-component dependencies.

### 1.5 Code Splitting Strategies

*   **Functional Decomposition:** Split code into functions based on specific tasks.
*   **Class-Based Decomposition:** Group related functions and data into classes.
*   **Module-Based Decomposition:** Organize classes and functions into modules based on their domain.
*   **Package-Based Decomposition:** Group related modules into packages.
*   **Vertical Slicing**: Create independent, deployable features from end-to-end (UI to database)

Apply these strategies iteratively as your codebase grows.  Keep pylint's static analysis capabilities in mind; avoid excessively complex functions or classes that can become difficult for pylint to analyze effectively.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **Factory Pattern:** Use factory functions or classes to create objects, promoting loose coupling and testability.
*   **Singleton Pattern:** Ensure that a class has only one instance, providing a global point of access.
*   **Observer Pattern:** Define a one-to-many dependency between objects, so that when one object changes state, all its dependents are notified and updated automatically.
*   **Strategy Pattern:** Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
*   **Dependency Injection:** Provide dependencies to a component rather than having it create them internally, increasing flexibility and testability.

### 2.2 Recommended Approaches for Common Tasks

*   **Configuration Management:** Use libraries like `ConfigParser`, `Dynaconf` or `pydantic-settings` to manage application configuration.
*   **Logging:** Use the `logging` module for structured logging.
*   **Data Validation:** Use libraries like `Pydantic` or `Marshmallow` to validate data.
*   **API Requests:** Use the `requests` library for making HTTP requests.
*   **Date and Time Handling:** Use the `datetime` module for date and time operations.
*   **Use Context Managers (`with` statements) whenever working with external resources like files, network connections, etc. to ensure proper cleanup.

### 2.3 Anti-patterns and Code Smells

*   **Long Methods/Functions:** Functions or methods exceeding a reasonable length (e.g., 50 lines) are hard to understand and maintain. Split them into smaller, more focused units.
*   **Duplicate Code:** Avoid copy-pasting code. Extract common logic into reusable functions or classes.
*   **Magic Numbers:** Use named constants instead of hardcoded values.
*   **Feature Envy:** A method accesses the data of another object more than its own. Move the method to the object it uses the most.
*   **God Class/Object:** A class that knows or does too much. Split it into smaller, more specialized classes.
*   **Spaghetti Code:** Code with a complex and tangled control structure, making it difficult to follow the logic. Refactor it into a more structured design.
*   **Shotgun Surgery:** Whenever you make one kind of change, you have to make many small changes to a lot of different classes. Consolidate changes into fewer classes.
*   **Primitive Obsession:** Using primitive data types to represent domain concepts. Create dedicated classes instead.
*   **Data Clumps:** Groups of variables that appear together in many places. Create a class to encapsulate them.

### 2.4 State Management

*   **Minimize Mutable State:** Favor immutable data structures to reduce complexity and potential errors.
*   **Centralized State Management:** Use patterns like Redux or Context API (if applicable) to manage application state in a central location.
*   **Clear State Transitions:** Define clear and predictable state transitions to avoid unexpected behavior.
*   **Avoid Global State:** Global state can make code harder to reason about and test. Use dependency injection or other techniques to manage dependencies.

### 2.5 Error Handling

*   **Specific Exception Handling:** Catch specific exceptions rather than generic `Exception` to handle errors appropriately.
*   **Use `try...except...finally`:** Ensure that resources are released properly, even if an exception occurs.
*   **Avoid Bare `except` Clauses:**  Always specify the exception type you are catching.
*   **Logging Exceptions:** Log detailed information about exceptions for debugging purposes.
*   **Raise Exceptions Responsibly:** Raise exceptions when an error occurs that cannot be handled locally. Provide meaningful error messages.
*   **Use Custom Exceptions:**  Create custom exception classes for specific error scenarios in your application, improving clarity and maintainability.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Profiling:** Use profiling tools like `cProfile` to identify performance bottlenecks.
*   **Algorithmic Optimization:** Choose efficient algorithms and data structures for critical tasks.
*   **Caching:** Use caching techniques to store frequently accessed data and reduce computation time (e.g., `functools.lru_cache`).
*   **List Comprehensions and Generators:** Use list comprehensions and generators for concise and efficient code.
*   **Avoid Premature Optimization:** Optimize code only after identifying actual performance bottlenecks.
*   **Use vectorized operations with NumPy when dealing with numerical computations to leverage performance.

### 3.2 Memory Management

*   **Avoid Memory Leaks:** Ensure that objects are properly deallocated when they are no longer needed.
*   **Use Generators:** Use generators to process large datasets in chunks, reducing memory usage.
*   **Minimize Object Creation:** Avoid creating unnecessary objects to reduce memory overhead.
*   **Use Data Structures Wisely:** Choose appropriate data structures based on memory usage and performance characteristics.
*   **Use the `del` keyword when you are completely finished with a variable that is holding a large data structure, allowing the garbage collector to reclaim the memory.

### 3.3 Rendering Optimization (If Applicable)

*   This is more relevant to UI applications or cases where Pylint is integrated into a tool with a visual output.
*   **Virtualization/Windowing**: Only render what is visible on the screen, especially for long lists or tables.
*   **Debouncing/Throttling**: Limit the frequency of updates to the display.

### 3.4 Bundle Size Optimization (If Applicable)

*   This mainly applies if you are building a web application or a desktop app where Pylint analysis is integrated as part of the build process.  Minimize dependencies, tree shake (remove unused code), and compress the output.

### 3.5 Lazy Loading

*   **Lazy Loading Modules:** Use lazy loading to load modules only when they are needed.
*   **Lazy Loading Data:** Fetch data only when it is required by the user or the application.
*   **Avoid Eager Loading:** Avoid loading large datasets or resources upfront.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

*   **SQL Injection:** Use parameterized queries or an ORM to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Use CSRF tokens to protect against CSRF attacks.
*   **Command Injection:** Avoid executing arbitrary commands based on user input.
*   **Path Traversal:** Validate file paths to prevent path traversal attacks.
*   **Regularly update your dependencies to patch security vulnerabilities.

### 4.2 Input Validation

*   **Validate All User Input:** Validate all user input to ensure that it is valid and safe.
*   **Use Regular Expressions:** Use regular expressions to validate input patterns.
*   **Sanitize Input:** Sanitize input to remove or escape potentially harmful characters.
*   **Limit Input Length:** Limit the length of input fields to prevent buffer overflows.

### 4.3 Authentication and Authorization

*   **Use Strong Passwords:** Enforce strong password policies and use password hashing algorithms (e.g., bcrypt, Argon2).
*   **Implement Role-Based Access Control (RBAC):** Define roles and permissions to control access to resources.
*   **Use Multi-Factor Authentication (MFA):** Add an extra layer of security with MFA.
*   **Secure Session Management:** Use secure session management techniques to protect user sessions.

### 4.4 Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data at rest and in transit.
*   **Use Secure Protocols:** Use secure protocols like HTTPS for communication.
*   **Data Masking:** Mask sensitive data when it is not needed.
*   **Data Anonymization:** Anonymize data to protect user privacy.

### 4.5 Secure API Communication

*   **Use Authentication Tokens:** Use authentication tokens to authenticate API requests (e.g., JWT).
*   **Rate Limiting:** Implement rate limiting to prevent abuse.
*   **Input Validation:** Validate all input to API endpoints.
*   **Output Sanitization:** Sanitize output from API endpoints to prevent injection attacks.

## 5. Testing Approaches

### 5.1 Unit Testing

*   **Test Individual Components:** Write unit tests for individual functions, classes, and modules.
*   **Use Assertions:** Use assertions to verify that the code behaves as expected.
*   **Test Edge Cases:** Test edge cases and boundary conditions.
*   **Test Exception Handling:** Test that exceptions are raised and handled correctly.
*   **Isolate Tests:** Ensure that unit tests are isolated and do not depend on external resources.

### 5.2 Integration Testing

*   **Test Interactions:** Write integration tests to verify that different components work together correctly.
*   **Test API Endpoints:** Test API endpoints to ensure that they are functioning as expected.
*   **Test Database Interactions:** Test database interactions to ensure that data is being stored and retrieved correctly.

### 5.3 End-to-End Testing

*   **Simulate User Flows:** Write end-to-end tests to simulate user flows and verify that the application behaves as expected from a user's perspective.
*   **Test UI Interactions:** Test UI interactions to ensure that the UI is functioning correctly.
*   **Test System Behavior:** Test overall system behavior and performance.

### 5.4 Test Organization

*   **Mirror Source Directory:** Organize tests in a directory structure that mirrors the source directory structure.
*   **Use Descriptive Names:** Use descriptive names for test files and test functions.
*   **Separate Test Files:** Separate unit tests, integration tests, and end-to-end tests into separate files.
*   **Use a Test Runner:** Use a test runner like `pytest` or `unittest` to run tests.

### 5.5 Mocking and Stubbing

*   **Use Mocks:** Use mocks to replace external dependencies with controlled test doubles.
*   **Use Stubs:** Use stubs to provide predefined responses for external dependencies.
*   **Isolate Units Under Test:** Use mocking and stubbing to isolate the unit under test and prevent dependencies from affecting test results.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   **Ignoring Pylint Warnings:** Ignoring Pylint warnings can lead to code quality issues and potential bugs. Always address Pylint warnings promptly.
*   **Over-Suppressing Messages:** Over-suppressing Pylint messages can hide important issues. Only suppress messages when you are certain that they are not relevant.
*   **Not Customizing Pylint Configuration:** Not customizing the Pylint configuration to match your project's needs can result in irrelevant warnings and missed issues. Customize the configuration to suit your project.
*   **Using Broad Exceptions:** Using broad exception clauses (e.g., `except Exception:`) can catch unintended exceptions and hide potential problems.
*   **Not Testing Edge Cases:** Not testing edge cases can lead to unexpected behavior and bugs.
*   **Not Using Virtual Environments:** Failing to use virtual environments can lead to dependency conflicts and deployment issues.

### 6.2 Edge Cases

*   **Dynamic Code Generation:** Pylint may have difficulty analyzing code that is dynamically generated at runtime.
*   **Complex Metaclasses:** Complex metaclasses can confuse Pylint and lead to false positives.
*   **Third-Party Libraries:** Pylint may not fully support all third-party libraries.
*   **Cython Extensions:** Cython extensions may not be analyzed correctly by Pylint.
*   **Monkey Patching**: Dynamic modification of code at runtime can confuse pylint and potentially lead to incorrect analysis.

### 6.3 Version-Specific Issues

*   **Compatibility Issues:** Different versions of Pylint may have compatibility issues with different versions of Python and third-party libraries.
*   **Message Changes:** Pylint messages may change between versions, requiring updates to your configuration.
*   **Checker Updates:** New checkers may be added in new versions of Pylint, requiring updates to your codebase.

### 6.4 Compatibility Concerns

*   **Black/Autopep8:** Ensure that Pylint's formatting rules are compatible with formatters like Black or Autopep8 to avoid conflicts.
*   **Mypy:** Use Mypy to complement Pylint's static analysis with type checking.
*   **Flake8:** Consider using Flake8 alongside Pylint for additional linting checks.

### 6.5 Debugging Strategies

*   **Read Error Messages Carefully:** Pylint error messages provide valuable information about the cause of the error.
*   **Use `--verbose`:** Use the `--verbose` flag to get more detailed output from Pylint.
*   **Isolate the Problem:** Try to isolate the problem to a specific function, class, or module.
*   **Use a Debugger:** Use a debugger to step through the code and identify the cause of the error.
*   **Consult the Pylint Documentation:** The Pylint documentation provides detailed information about all of Pylint's features and messages.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **VS Code with Python Extension:** VS Code is a popular code editor with excellent Python support.
*   **PyCharm:** PyCharm is a powerful IDE for Python development.
*   **Pylance (VS Code extension):** A fast, feature-rich language server for Python in VS Code.
*   **Docker:** Use Docker for consistent and reproducible development environments.

### 7.2 Build Configuration

*   **Use a Build System:** Use a build system like `Make`, `Poetry`, or `tox` to automate build tasks.
*   **Define Dependencies:** Define dependencies in a `requirements.txt` or `pyproject.toml` file.
*   **Use Virtual Environments:** Use virtual environments to isolate project dependencies.
*   **Automate Pylint Execution:** Automate Pylint execution as part of the build process.
*   **Include .pylintrc:** Ensure that the `.pylintrc` file is included in the project repository.

### 7.3 Linting and Formatting

*   **Use a Code Formatter:** Use a code formatter like `Black` or `Autopep8` to automatically format code.
*   **Integrate with Editor:** Integrate Pylint and the code formatter with your code editor for real-time feedback.
*   **Use Pre-Commit Hooks:** Use pre-commit hooks to run Pylint and the code formatter before committing changes.
*   **Address all linting and formatting issues consistently.

### 7.4 Deployment

*   **Use a Deployment Pipeline:** Use a deployment pipeline to automate the deployment process.
*   **Test Before Deployment:** Run unit tests, integration tests, and end-to-end tests before deploying the application.
*   **Monitor Application:** Monitor the application after deployment to detect and resolve issues.
*   **Use a Configuration Management Tool:** Use a configuration management tool like Ansible or Chef to manage application configuration in production.

### 7.5 CI/CD Integration

*   **Integrate with CI/CD System:** Integrate Pylint with your CI/CD system (e.g., Jenkins, GitLab CI, GitHub Actions).
*   **Run Pylint in CI/CD Pipeline:** Run Pylint as part of the CI/CD pipeline to automatically check code quality.
*   **Fail Build on Errors:** Configure the CI/CD pipeline to fail the build if Pylint reports any errors.
*   **Collect Metrics:** Collect metrics from Pylint runs to track code quality over time.
*   **Use code review processes and tools to enforce coding standards.

By adhering to these best practices, you can leverage Pylint to create high-quality, maintainable, and secure Python code.