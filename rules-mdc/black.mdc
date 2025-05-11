---
description: Enforces consistent code formatting using Black, the uncompromising Python code formatter, promoting readability and reducing diffs. Covers best practices related to Black's configuration, usage, and integrations.
globs: *.py
---
- Adopt Black as the primary code formatter for all Python projects.
- Adhere to Black's default style guide for consistent formatting across projects.
- Integrate Black into the development workflow using pre-commit hooks and CI/CD pipelines.
- Configure Black using `pyproject.toml` file for project-specific settings, if needed.
- Use Black's CLI or editor integrations for seamless formatting.

## Black Code Formatting Best Practices

Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from `pycodestyle` nagging about formatting. You will save time and mental energy for more important matters.

### Key Principles

- **Consistency:** Black enforces a consistent code style, reducing ambiguity and promoting readability.
- **Automation:** Black automates the formatting process, eliminating manual effort and reducing formatting-related code review comments.
- **Integration:** Black integrates seamlessly with popular development tools and workflows.
- **Minimal Configuration:** Black provides sensible defaults and limits configuration options to ensure consistency.

### 1. Code Organization and Structure

Black primarily focuses on formatting within files, but it indirectly influences code organization by promoting readability and consistent style. Consider the following:

- **Directory Structure:** Black doesn't enforce a specific directory structure. However, a well-organized project with clear module boundaries will benefit more from consistent formatting.
- **File Naming Conventions:** Black uses the `.py` extension for python files. Adhering to PEP 8 naming conventions alongside Black is recommended for improved readability (`snake_case` for variables, functions, and modules; `PascalCase` for classes).
- **Module Organization:** Black formats code within modules. Break large modules into smaller, more manageable files to improve readability and maintainability. Organize related functions and classes within modules.
- **Component Architecture:** Black can be used with any component architecture. Focus on creating well-defined components with clear interfaces. Consistent formatting within each component enhances maintainability.
- **Code Splitting Strategies:** Black formats code regardless of the splitting strategy. Break code into smaller functions or classes to improve code readability and organization.

### 2. Common Patterns and Anti-patterns

- **Design Patterns:** Black doesn't impose any specific design patterns. However, it encourages clean, readable code that makes it easier to implement design patterns correctly. For example, consistent formatting makes it easier to identify and understand the structure of a Factory pattern.
- **Recommended Approaches:**
    - Use Black on all Python files in your project.
    - Integrate Black into your pre-commit hooks to automatically format code before committing.
    - Use Black's CLI to format entire directories or specific files.
    - Configure Black using `pyproject.toml` to customize settings like line length, if necessary.
- **Anti-patterns:**
    - Manually formatting code instead of using Black.
    - Ignoring Black's formatting suggestions.
    - Disabling Black for specific files or sections of code without a strong reason.
- **State Management:** Black doesn't directly address state management, but its consistent formatting promotes code clarity, making state management strategies easier to implement and understand. Consider using established state management patterns like Redux or Context API (if applicable).
- **Error Handling:** Black encourages readable code, making it easier to implement and understand error handling mechanisms. Use `try...except` blocks to handle exceptions gracefully. Consistent formatting of exception handling code improves maintainability.

### 3. Performance Considerations

Black's formatting process itself has minimal performance overhead. The main performance considerations are related to the code being formatted, not Black itself.

- **Optimization Techniques:** Black-formatted code is easier to read and understand, which can help identify performance bottlenecks. Use profiling tools to identify slow code and optimize accordingly.
- **Memory Management:** Black doesn't directly affect memory management. Follow Python's memory management best practices, such as using generators for large datasets and avoiding circular references.
- **Rendering Optimization:** Not applicable as Black is primarily a code formatter and not directly involved in rendering.
- **Bundle Size Optimization:** Not applicable as Black is a code formatter. Bundle size optimization techniques are relevant to web applications or other deployments where code size impacts load times.
- **Lazy Loading:** Not applicable as Black is a code formatter.

### 4. Security Best Practices

Black improves code readability, which can help identify potential security vulnerabilities. However, Black itself doesn't directly address security. Focus on the following security best practices:

- **Common Vulnerabilities:** Black doesn't prevent common vulnerabilities like SQL injection or cross-site scripting (XSS). Use secure coding practices to avoid these vulnerabilities.
- **Input Validation:** Implement robust input validation to prevent injection attacks. Black's consistent formatting makes it easier to review input validation code.
- **Authentication and Authorization:** Use established authentication and authorization patterns to protect your application. Black's consistent formatting can improve the readability of authentication and authorization code.
- **Data Protection:** Use encryption to protect sensitive data. Follow data protection best practices, such as storing passwords securely.
- **Secure API Communication:** Use HTTPS to encrypt API communication. Implement API authentication and authorization to prevent unauthorized access.

### 5. Testing Approaches

Black promotes consistent code formatting, making tests easier to write, read, and maintain.

- **Unit Testing:** Write unit tests for individual functions and classes. Black's consistent formatting makes it easier to write clear and concise unit tests.
- **Integration Testing:** Write integration tests to verify that different components of your application work together correctly. Consistent formatting makes it easier to understand the interactions between components.
- **End-to-end Testing:** Write end-to-end tests to verify that your application works as expected from a user's perspective. Consistent formatting makes it easier to debug end-to-end tests.
- **Test Organization:** Organize your tests into a logical directory structure. Follow a consistent naming convention for test files and functions.
- **Mocking and Stubbing:** Use mocking and stubbing to isolate units of code during testing. Black's consistent formatting makes it easier to understand the code being tested and create appropriate mocks.

### 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Not integrating Black into the development workflow.
    - Ignoring Black's formatting suggestions.
    - Over-configuring Black and deviating from its default style.
- **Edge Cases:**
    - Black may have difficulty formatting very complex or deeply nested expressions.
    - Black's formatting may not always be optimal for specific use cases.
- **Version-specific Issues:**
    - Ensure that all developers are using the same version of Black to avoid formatting inconsistencies.
- **Compatibility Concerns:**
    - Black may have compatibility issues with older versions of Python or other tools.
- **Debugging Strategies:**
    - Use Black's CLI to format code and identify formatting issues.
    - Use a debugger to step through code and understand its behavior.

### 7. Tooling and Environment

- **Recommended Tools:**
    - Black: The primary code formatter.
    - Pre-commit: A tool for managing pre-commit hooks.
    - VS Code, PyCharm, or other Python IDEs with Black integration.
- **Build Configuration:**
    - Configure Black in your project's `pyproject.toml` file.
    - Integrate Black into your build process to automatically format code.
- **Linting and Formatting:**
    - Use Black for formatting and other linters (e.g., Flake8, pylint) for code quality checks.
    - Configure your linter to ignore formatting errors reported by Black.
- **Deployment:**
    - Ensure that your deployment environment has Black installed.
    - Run Black as part of your deployment process to ensure consistent formatting.
- **CI/CD Integration:**
    - Integrate Black into your CI/CD pipeline to automatically format code and run tests.

### Example `.pre-commit-config.yaml`


repos:
-   repo: https://github.com/psf/black
    rev: 24.2.2  # Replace with the latest version
    hooks:
    -   id: black


### Example `pyproject.toml`

toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
exclude = '''
/(\.(git|hg|mypy_cache|tox|venv)|_build|build|dist)/  # Removed: |/foo/bar/
'''


By following these best practices, you can leverage Black to improve code quality, enhance collaboration, and reduce development time.