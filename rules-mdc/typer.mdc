---
description: This rule provides best practices and coding standards for developing command-line interfaces (CLIs) using the Typer library in Python. It includes guidelines for code organization, performance, security, and testing, aiming to enhance usability and maintainability.
globs: *.py
---
- Always use UV when installing depdendencies
- Always use python 3.12
- Always use classes instead of function

# Typer CLI Best Practices and Coding Standards

This document outlines best practices and coding standards for developing command-line interfaces (CLIs) using the Typer library in Python. Following these guidelines will help you create robust, user-friendly, and maintainable CLIs.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

Adopt a structured directory layout to improve code discoverability and maintainability. A typical project structure might look like this:


my_cli_project/
├── my_cli/
│   ├── __init__.py
│   ├── main.py          # Main application logic
│   ├── commands/        # Subcommands (if applicable)
│   │   ├── __init__.py
│   │   ├── command1.py
│   │   └── command2.py
│   ├── utils/           # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── models/          # Data models (if applicable)
│   │   ├── __init__.py
│   │   └── data_models.py
├── tests/             # Unit and integration tests
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_commands.py
│   └── conftest.py     # pytest configuration
├── README.md
├── pyproject.toml    # Project metadata and dependencies
└── .gitignore


*   **`my_cli/`**: Main package directory.
*   **`my_cli/main.py`**: Entry point of the CLI application, where Typer app is initialized.
*   **`my_cli/commands/`**: Directory for organizing subcommands into separate modules.
*   **`my_cli/utils/`**: Helper functions and utilities used throughout the application.
*   **`my_cli/models/`**: Data models and schemas used by the CLI.
*   **`tests/`**: Contains all tests for the application.
*   **`README.md`**: Project documentation.
*   **`pyproject.toml`**: Project metadata and dependencies (using Poetry or similar).
*   **.gitignore**: Specifies intentionally untracked files that Git should ignore.

### 1.2. File Naming Conventions

*   Use descriptive and consistent file names.
*   Main application file: `main.py`
*   Subcommand modules: `command_name.py`
*   Utility modules: `helpers.py`, `utils.py`
*   Model definitions: `models.py`, `data_models.py`
*   Test files: `test_module_name.py`

### 1.3. Module Organization Best Practices

*   **Single Responsibility Principle**: Each module should have a single, well-defined purpose.
*   **Clear Interfaces**: Define clear interfaces between modules to minimize dependencies.
*   **Avoid Circular Dependencies**: Be mindful of circular dependencies between modules, which can lead to import errors and code that is difficult to reason about.
*   **Use Packages**: Organize modules into packages using `__init__.py` files to create a hierarchical structure.

### 1.4. Component Architecture Recommendations

*   **Typer App Initialization**: Create a Typer app instance in `main.py`.
*   **Subcommand Grouping**: Group related commands using Typer's `@app.command()` decorator.
*   **Dependency Injection**: Use dependency injection to provide dependencies to command functions.
*   **Configuration Management**: Load configuration from environment variables or configuration files.

### 1.5. Code Splitting Strategies

*   **Subcommands as Modules**: Split subcommands into separate modules within the `commands/` directory.
*   **Utility Functions**: Extract reusable utility functions into the `utils/` directory.
*   **Data Models**: Define data models in the `models/` directory.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Command Pattern**: Typer inherently implements the command pattern, where each CLI command is a separate executable action.
*   **Dependency Injection**: Inject dependencies into command functions using Typer's type hinting system.
*   **Configuration Pattern**: Load configuration from environment variables or configuration files using a dedicated configuration module.

### 2.2. Recommended Approaches for Common Tasks

*   **Argument Parsing**: Utilize Typer's built-in argument parsing capabilities using type hints and decorators.
*   **Subcommands**: Organize commands into subcommands for complex CLIs using `@app.command()` decorator.
*   **Help Messages**: Customize help messages using docstrings for commands and parameters.
*   **Error Handling**: Implement robust error handling using `try...except` blocks and custom exception classes.

### 2.3. Anti-patterns and Code Smells

*   **God Object**: Avoid creating a single, monolithic command function that handles too many responsibilities.
*   **Magic Numbers**: Avoid hardcoding values directly in the code. Use constants or configuration variables instead.
*   **Duplicated Code**: Extract reusable code into utility functions or classes.
*   **Ignoring Errors**: Always handle errors gracefully and provide informative error messages to the user.

### 2.4. State Management Best Practices

*   **Stateless Commands**: Design commands to be stateless whenever possible. Pass all necessary data as arguments.
*   **Context Objects**: Use Typer's context object to share state between commands.
*   **External Storage**: Store persistent state in a database or configuration file.

### 2.5. Error Handling Patterns

*   **Try...Except Blocks**: Use `try...except` blocks to catch exceptions and handle errors gracefully.
*   **Custom Exceptions**: Define custom exception classes for specific error conditions.
*   **Logging**: Log errors and warnings to a file or console for debugging purposes.
*   **Informative Error Messages**: Provide clear and informative error messages to the user.
*   **Exit Codes**: Return appropriate exit codes to indicate success or failure.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Lazy Loading**: Use lazy loading to defer the loading of modules and resources until they are actually needed.
*   **Caching**: Cache frequently accessed data to reduce the number of database queries or API calls.
*   **Profiling**: Profile the code to identify performance bottlenecks and optimize them.
*   **Asynchronous Operations**: Use asynchronous operations for I/O-bound tasks to improve responsiveness.

### 3.2. Memory Management Considerations

*   **Resource Cleanup**: Ensure that resources are properly released after use to prevent memory leaks.
*   **Data Structures**: Use efficient data structures to minimize memory usage.
*   **Generators**: Use generators to process large datasets in a memory-efficient manner.

### 3.3. Bundle Size Optimization

*   **Dependency Management**: Minimize the number of dependencies to reduce the bundle size.
*   **Code Minification**: Minify the code to reduce its size.

### 3.4. Lazy Loading Strategies

*   **Conditional Imports**: Use conditional imports to load modules only when they are needed.
*   **Dynamic Loading**: Use dynamic loading to load modules at runtime.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

*   **Command Injection**: Prevent command injection by validating user input and avoiding the use of `os.system` or `subprocess.call` with untrusted input.
*   **Path Traversal**: Prevent path traversal by validating file paths and ensuring that users cannot access files outside of authorized directories.
*   **Denial of Service (DoS)**: Prevent DoS attacks by limiting resource consumption and implementing rate limiting.

### 4.2. Input Validation Best Practices

*   **Type Checking**: Use Typer's type hinting system to enforce type checking of user input.
*   **Regular Expressions**: Use regular expressions to validate user input against specific patterns.
*   **Range Validation**: Validate that numerical input falls within acceptable ranges.
*   **Whitelisting**: Whitelist allowed values for user input.

### 4.3. Authentication and Authorization Patterns

*   **API Keys**: Use API keys to authenticate users and authorize access to resources.
*   **OAuth 2.0**: Implement OAuth 2.0 for more complex authentication and authorization scenarios.
*   **Role-Based Access Control (RBAC)**: Implement RBAC to control access to resources based on user roles.

### 4.4. Data Protection Strategies

*   **Encryption**: Encrypt sensitive data at rest and in transit.
*   **Hashing**: Hash passwords and other sensitive data before storing them.
*   **Secure Storage**: Store sensitive data in a secure storage location, such as a password manager or a hardware security module (HSM).

### 4.5. Secure API Communication

*   **HTTPS**: Use HTTPS to encrypt communication between the CLI and the API.
*   **TLS/SSL**: Use TLS/SSL certificates to verify the identity of the API server.
*   **Input Sanitization**: Sanitize data received from APIs to prevent cross-site scripting (XSS) attacks.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   **Test-Driven Development (TDD)**: Write unit tests before writing the code.
*   **Mocking and Stubbing**: Use mocking and stubbing to isolate units of code and test them in isolation.
*   **Assertion Libraries**: Use assertion libraries like `pytest` or `unittest` to write clear and concise assertions.

### 5.2. Integration Testing Approaches

*   **Test Command Interactions**: Verify that commands interact correctly with each other.
*   **Test Database Interactions**: Verify that the CLI interacts correctly with the database.
*   **Test API Interactions**: Verify that the CLI interacts correctly with the API.

### 5.3. End-to-End Testing Recommendations

*   **Automated Testing**: Automate end-to-end tests using tools like `Selenium` or `Playwright`.
*   **Real-World Scenarios**: Test the CLI in real-world scenarios to ensure that it meets the needs of users.

### 5.4. Test Organization Best Practices

*   **Separate Test Files**: Create separate test files for each module or component.
*   **Descriptive Test Names**: Use descriptive test names to make it easy to understand what each test is testing.
*   **Test Fixtures**: Use test fixtures to set up and tear down test environments.

### 5.5. Mocking and Stubbing Techniques

*   **Mock External Dependencies**: Mock external dependencies like databases, APIs, and file systems.
*   **Stub Function Calls**: Stub function calls to control the behavior of dependencies.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Ignoring Exit Codes**: Not checking exit codes of subprocesses can lead to undetected errors.
*   **Incorrect Argument Parsing**: Mishandling of arguments leads to unexpected behavior.
*   **Lack of Input Validation**: Failing to validate user input can lead to security vulnerabilities.

### 6.2. Edge Cases

*   **Handling Large Files**: Properly handle large files to avoid memory issues.
*   **Internationalization (i18n)**: Be aware of character encoding issues.
*   **Timezone Handling**: Ensure correct timezone handling when displaying or processing time-related data.

### 6.3. Version-Specific Issues

*   **Check Typer Versions**: Be aware of compatibility issues between different Typer versions.
*   **Dependency Conflicts**:  Resolve conflicts between Typer and other dependencies.

### 6.4. Compatibility Concerns

*   **Operating System Compatibility**: Ensure that the CLI is compatible with different operating systems (Linux, macOS, Windows).
*   **Python Version Compatibility**: Ensure that the CLI is compatible with different Python versions (3.7, 3.8, 3.9, etc.).

### 6.5. Debugging Strategies

*   **Logging**: Use logging to track the flow of execution and identify errors.
*   **Debugging Tools**: Use debugging tools like `pdb` to step through the code and inspect variables.
*   **Error Messages**: Pay attention to error messages and stack traces to identify the source of errors.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **Text Editor**: Use a text editor like VS Code, Sublime Text, or Atom.
*   **IDE**: Use an IDE like PyCharm for advanced features like debugging and code completion.
*   **Virtual Environment**: Use a virtual environment to isolate project dependencies.
*   **Poetry or pip**: Use Poetry or pip for dependency management.

### 7.2. Build Configuration Best Practices

*   **pyproject.toml**: Use `pyproject.toml` to specify project metadata and dependencies.
*   **setup.py**: Use `setup.py` to package the CLI for distribution.

### 7.3. Linting and Formatting

*   **Flake8**: Use Flake8 to lint the code and enforce coding style guidelines.
*   **Black**: Use Black to automatically format the code.
*   **mypy**: Use mypy for static type checking.

### 7.4. Deployment Best Practices

*   **Packaging**: Package the CLI using `setuptools` or `Poetry`.
*   **Distribution**: Distribute the CLI using PyPI or other package repositories.

### 7.5. CI/CD Integration

*   **GitHub Actions**: Use GitHub Actions for CI/CD pipelines.
*   **Testing**: Run unit and integration tests in the CI/CD pipeline.
*   **Deployment**: Deploy the CLI automatically to PyPI or other package repositories.

## 8. Additional Tips

*   **Naming Conventions**:  Command names should be intuitive, starting with a lowercase letter and avoiding single-letter commands. Aim for meaningful names that are easy to remember while keeping them short and avoiding conflicts with existing commands.
*   **Exit Codes**: Adhere to standard exit codes where `0` indicates success and non-zero values indicate errors. This is crucial for interoperability with other command-line tools and CI/CD systems.
*   **Argument Parsing**: Utilize Typer for argument parsing instead of the built-in `argparse`, as it provides a more user-friendly experience and better support for features like subcommands and automatic help generation.
*   **Help and Documentation**: Implement comprehensive help messages and support for both `-h` and `--help` options. Clear documentation improves user experience and reduces confusion.
*   **Standard Input/Output**: Design your CLI to read from standard input (stdin) and write to standard output (stdout) and standard error (stderr) appropriately. This allows your tool to be easily integrated into pipelines.

By following these guidelines, you can create robust, user-friendly, and maintainable CLIs using the Typer library in Python.