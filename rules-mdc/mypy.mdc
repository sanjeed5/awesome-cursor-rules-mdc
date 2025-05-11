---
description: This rule file outlines best practices for using mypy in Python projects, emphasizing gradual adoption, consistent configuration, and leveraging advanced features for improved code quality and maintainability. It covers code organization, performance, security, testing, common pitfalls, and tooling.
globs: *.py
---
# Mypy Best Practices and Coding Standards

This document outlines the recommended best practices for using Mypy in Python projects. Following these guidelines can lead to more maintainable, robust, and understandable code.

## 1. Gradual Typing and Adoption

- **Start Small:** When introducing Mypy to an existing codebase, focus on a manageable subset of the code first. Choose modules or files that are relatively isolated or self-contained.
- **Iterative Annotation:** Gradually increase the coverage by adding type hints as you modify or add new code. Avoid large-scale refactoring solely for the purpose of adding type hints.
- **`# type: ignore` Strategically:** Use `# type: ignore` comments sparingly to temporarily suppress errors in code that is not yet fully typed. Always include a specific error code when using `# type: ignore` to avoid unintentionally ignoring other errors.  Review these regularly.
- **Prioritize Widely Imported Modules:** Focus on annotating modules that are imported by many other modules. This will provide the greatest benefit in terms of type checking and error detection.

## 2. Consistent Configuration and Integration

- **Standardized Configuration:** Ensure that all developers use the same Mypy configuration and version. Use a `mypy.ini` file or a `pyproject.toml` file to define the project's Mypy settings. This file should be version-controlled.
- **CI Integration:** Integrate Mypy checks into your Continuous Integration (CI) pipeline to catch type errors early in the development process. Use a pre-commit hook to run Mypy before committing code.
- **Editor Integration:** Encourage developers to use Mypy integration in their code editors to get real-time feedback on type errors. Most popular Python editors, such as VS Code, PyCharm, and Sublime Text, have Mypy plugins or extensions.
- **Version Pinning:** Pin the version of mypy in your project's dependencies to ensure consistent behavior across different environments.

## 3. Leveraging Advanced Features

- **Strict Mode:** Utilize Mypy's strict mode (enabled with the `--strict` flag) to catch more potential errors. Strict mode enables a collection of stricter type checking options.
- **`--warn-unused-ignores`:** Use this flag to identify `# type: ignore` comments that are no longer necessary because the corresponding errors have been fixed.
- **`--disallow-untyped-defs`:** Use this flag to require type annotations for all function definitions.
- **`--disallow-incomplete-defs`:** Use this flag to disallow function definitions with incomplete type annotations.
- **`--check-untyped-defs`:** Use this flag to type check function bodies even if the signature lacks type annotations.
- **Protocols and Structural Subtyping:** Take advantage of Mypy's support for protocols and structural subtyping to define flexible interfaces and improve code reusability.
- **Generics:** Use generics to write type-safe code that can work with different types of data.
- **TypedDict:** Use `TypedDict` to define the types of dictionaries with known keys and values. This can help to prevent errors when working with data structures.

## 4. Code Organization and Structure

- **Directory Structure:** Use a well-defined directory structure to organize your code. A common pattern is the `src` layout, where the project's source code is located in a `src` directory.
- **File Naming Conventions:** Follow consistent file naming conventions. Use lowercase letters and underscores for module names (e.g., `my_module.py`).
- **Module Organization:** Organize your code into logical modules. Each module should have a clear purpose and a well-defined interface.
- **Component Architecture:** Design your application using a component-based architecture. Each component should be responsible for a specific task and should have well-defined inputs and outputs.
- **Code Splitting:** Split large modules into smaller, more manageable files. This can improve code readability and maintainability.

## 5. Common Patterns and Anti-patterns

- **Dependency Injection:** Use dependency injection to decouple components and improve testability.
- **Abstract Factories:** Use abstract factories to create families of related objects without specifying their concrete classes.
- **Singletons:** Avoid using singletons excessively. They can make code harder to test and reason about.
- **Global State:** Minimize the use of global state. It can make code harder to understand and debug.
- **Exception Handling:** Use exception handling to gracefully handle errors and prevent the application from crashing. Avoid catching generic exceptions (e.g., `except Exception:`). Catch specific exceptions and handle them appropriately.

## 6. Performance Considerations

- **Profiling:** Use profiling tools to identify performance bottlenecks in your code.
- **Caching:** Use caching to store frequently accessed data and reduce the number of expensive operations.
- **Lazy Loading:** Use lazy loading to defer the loading of resources until they are actually needed.
- **Efficient Data Structures:** Choose appropriate data structures for your data. For example, use sets for membership testing and dictionaries for key-value lookups.
- **Avoid Unnecessary Copying:** Avoid making unnecessary copies of data. This can consume memory and slow down your code.

## 7. Security Best Practices

- **Input Validation:** Validate all user inputs to prevent injection attacks and other security vulnerabilities. Use type annotations to enforce type constraints on function arguments.
- **Authentication and Authorization:** Implement robust authentication and authorization mechanisms to protect your application from unauthorized access.
- **Data Protection:** Protect sensitive data by encrypting it and storing it securely.
- **Secure API Communication:** Use HTTPS to encrypt communication between your application and external APIs.
- **Dependency Management:** Regularly audit your project's dependencies for security vulnerabilities and update them to the latest versions.

## 8. Testing Approaches

- **Unit Testing:** Write unit tests for all components in your application. Unit tests should verify that each component behaves as expected in isolation.
- **Integration Testing:** Write integration tests to verify that different components in your application work together correctly.
- **End-to-End Testing:** Write end-to-end tests to verify that the entire application works as expected from the user's perspective.
- **Test Organization:** Organize your tests into a clear and logical structure. Use separate directories for unit tests, integration tests, and end-to-end tests.
- **Mocking and Stubbing:** Use mocking and stubbing to isolate components during testing and to simulate external dependencies.

## 9. Common Pitfalls and Gotchas

- **`Any` Type:** Avoid using the `Any` type excessively. It effectively disables type checking for the corresponding code.
- **Inconsistent Type Annotations:** Ensure that type annotations are consistent throughout your codebase. Inconsistent type annotations can lead to unexpected errors.
- **Ignoring Errors:** Avoid ignoring Mypy errors without a good reason. Mypy errors usually indicate a real problem in your code.
- **Version Compatibility:** Be aware of compatibility issues between different versions of Mypy and other libraries.
- **Circular Dependencies:** Avoid circular dependencies between modules. They can make code harder to understand and test.

## 10. Tooling and Environment

- **Development Tools:** Use a good code editor with Mypy integration, such as VS Code, PyCharm, or Sublime Text.
- **Build Configuration:** Use a build system, such as `setuptools` or `poetry`, to manage your project's dependencies and build process.
- **Linting and Formatting:** Use a linter, such as `flake8` or `ruff`, to enforce code style and detect potential errors. Use a code formatter, such as `black` or `ruff`, to automatically format your code.
- **Deployment:** Deploy your application to a production environment using a container system, such as Docker, or a cloud platform, such as AWS or Google Cloud.
- **CI/CD:** Integrate Mypy into your CI/CD pipeline to automatically check your code for type errors before deploying it to production.

## 11. Additional Best Practices

- **Use type hints for all function arguments and return values.** This makes it easier to understand what types of data the function expects and returns.
- **Use type aliases to simplify complex type annotations.** This can make your code more readable and maintainable.
- **Use the `typing` module to access advanced type features.** The `typing` module provides a number of useful type-related classes and functions, such as `List`, `Dict`, `Union`, and `Optional`.
- **Use the `reveal_type()` function to inspect the type of an expression.** This can be helpful for debugging type-related issues.
- **Keep your code clean and well-organized.** This makes it easier to understand and maintain.
- **Write clear and concise comments.** This helps others understand your code and how it works.
- **Follow the PEP 8 style guide.** This ensures that your code is consistent and readable.
- **Use a version control system.** This allows you to track changes to your code and collaborate with others.

By following these best practices, you can improve the quality, maintainability, and robustness of your Python code that uses mypy and other Python tools.