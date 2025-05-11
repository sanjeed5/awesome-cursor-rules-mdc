---
description: Comprehensive best practices for using pdoc to generate and maintain Python project documentation. Covers code structure, performance, security, testing, and tooling to ensure high-quality documentation and maintainable projects.
globs: *.py
---
# pdoc Best Practices: A Comprehensive Guide

This document outlines best practices for using `pdoc` to generate API documentation for Python projects. It covers code organization, performance considerations, security, testing, and common pitfalls, all tailored for effective documentation generation and maintainability.

## Library Information:

- Name: pdoc
- Tags: development, documentation, python, auto-generation

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

- **Clear Module Grouping:** Organize your code into logical modules and sub-packages.  This makes it easier for `pdoc` to generate a navigable API.
- **`__init__.py` Files:** Ensure that each package directory contains an `__init__.py` file. This file can be empty, or it can contain initialization code and, importantly, the package-level docstring which `pdoc` will use.
- **Separate Concerns:**  Isolate different functionalities into separate modules. For example, separate data models, business logic, and utility functions into distinct files.

Example:


my_project/
├── my_package/
│   ├── __init__.py  # Package-level docstring here
│   ├── module_a.py  # Contains functions and classes
│   ├── module_b.py  # Contains more functions and classes
│   └── subpackage/
│       ├── __init__.py  # Subpackage-level docstring here
│       └── module_c.py # Contains functions and classes
└── setup.py       # Installation script


### 1.2 File Naming Conventions

- **Descriptive Names:** Use descriptive names for modules and packages. For example, `database_utils.py` is much better than `db.py`.
- **Lowercase with Underscores:** Follow the standard Python convention of using lowercase letters with underscores for module names (e.g., `my_module.py`).

### 1.3 Module Organization

- **Top-Level Docstrings:** Every module should have a top-level docstring that describes its purpose and lists the main classes, functions, and exceptions it defines.
- **`__all__` Variable:** Use the `__all__` variable to explicitly define the public API of a module. This tells `pdoc` (and other tools) which names should be included in the generated documentation.

Example:

python
# my_module.py
"""This module provides utility functions for data processing.

Classes:
    DataProcessor: Processes data.

Functions:
    clean_data: Cleans the data.
"""

__all__ = ['DataProcessor', 'clean_data']

class DataProcessor:
    """Processes data from various sources."""
    ...

def clean_data(data):
    """Cleans the input data."""
    ...


### 1.4 Component Architecture

- **Loose Coupling:** Design your components to be loosely coupled. This means that components should have minimal dependencies on each other.
- **Clear Interfaces:** Define clear interfaces for your components. This makes it easier to test and document your code.
- **Abstraction:** Use abstraction to hide the implementation details of your components. This makes your code more flexible and easier to maintain.

### 1.5 Code Splitting

- **Divide and Conquer:** Split large modules into smaller, more manageable files. This makes your code easier to understand and maintain.
- **By Functionality:** Split code based on functionality. For example, if you have a module that contains both database access code and business logic, split it into two separate modules.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

- **Factory Pattern:** Use the factory pattern to create objects. This makes it easier to switch between different implementations of a class.
- **Strategy Pattern:** Use the strategy pattern to encapsulate different algorithms. This makes it easier to switch between different algorithms at runtime.
- **Decorator Pattern:** Use the decorator pattern to add functionality to objects dynamically.

### 2.2 Recommended Approaches

- **Use Docstrings:** All modules, classes, functions, and methods must include docstrings. Docstrings are used by `pdoc` to create the API documentation.
- **Follow PEP 257:** Adhere to PEP 257 for docstring conventions. This ensures consistency and readability.
- **Choose a Style:** Consistently use a docstring style (Google, NumPy, reStructuredText) throughout your project.
- **Include Examples:** Provide simple, executable examples in your docstrings. This helps users understand how to use your code.  These examples can be embedded directly within docstrings using the `doctest` module.

### 2.3 Anti-patterns

- **Missing Docstrings:** Failing to document modules, classes, functions, or methods.
- **Incomplete Docstrings:** Providing docstrings that are too brief or lack essential information.
- **Inconsistent Style:** Mixing different docstring styles within the same project.
- **Redundant Documentation:** Repeating information that is already clear from the code itself. Focus on the *why* rather than the *how*.
- **Ignoring `__all__`:**  Not defining `__all__` or defining it incorrectly can lead to undocumented or wrongly documented API surfaces.

### 2.4 State Management

- **Minimize Global State:** Avoid using global variables as much as possible. If you must use global variables, document them clearly.
- **Use Classes to Encapsulate State:** Use classes to encapsulate state. This makes it easier to manage and reason about your code.

### 2.5 Error Handling

- **Raise Exceptions:** Raise exceptions when errors occur. This makes it easier to handle errors gracefully.
- **Use Try-Except Blocks:** Use try-except blocks to catch exceptions. This prevents your program from crashing when errors occur.
- **Document Exceptions:** Document the exceptions that your functions and methods can raise.

## 3. Performance Considerations

### 3.1 Optimization Techniques

- **Profiling:** Use a profiler to identify performance bottlenecks in your code.
- **Caching:** Use caching to store frequently accessed data.  Consider using `functools.lru_cache` for memoization.
- **Vectorization:** Use vectorized operations when working with numerical data. This is often faster than using loops.

### 3.2 Memory Management

- **Avoid Memory Leaks:** Be careful to avoid memory leaks. Free up memory when it is no longer needed.
- **Use Generators:** Use generators to process large amounts of data. This can reduce memory usage.
- **Use Data Structures Efficiently:** Use the right data structures for the job. For example, use sets for membership testing and dictionaries for lookups.

### 3.3 Rendering Optimization

- Not generally applicable to `pdoc` itself, as it mainly generates static HTML.  However, the design of your code *can* affect how quickly `pdoc` can process it. Well-structured, easily-introspected code will be processed faster.

### 3.4 Bundle Size

- Not applicable to `pdoc` itself, as it is a documentation generator and not a web application framework.

### 3.5 Lazy Loading

- Not directly applicable to `pdoc`, but consider lazy loading modules within your own code to improve startup time if you have a large project that `pdoc` needs to process. This is done using `importlib.import_module`.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

- **Not Directly Applicable:** `pdoc` generates static documentation and does not execute arbitrary user-supplied code.  Therefore, it is less vulnerable than many other types of applications. However, be mindful of the dependencies you use and keep them up to date.

### 4.2 Input Validation

- **Not Directly Applicable:** `pdoc` does not take direct user input.

### 4.3 Authentication and Authorization

- **Not Applicable:** `pdoc` is a documentation generation tool and does not require authentication or authorization.

### 4.4 Data Protection

- **Not Applicable:** `pdoc` does not handle sensitive data.

### 4.5 Secure API Communication

- **Not Applicable:** `pdoc` does not communicate with external APIs.

## 5. Testing Approaches

### 5.1 Unit Testing

- **Test Docstrings:** Use `doctest` to test examples embedded in your docstrings. This ensures that your examples are up-to-date and correct. `pdoc` renders these examples, making them executable directly from the documentation.
- **Focus on Public API:** Write unit tests for the public API of your modules, classes, and functions.

### 5.2 Integration Testing

- **Test Interactions:** Test the interactions between different components of your system.

### 5.3 End-to-End Testing

- **Test Complete Workflows:** Test complete workflows to ensure that your system is working correctly.

### 5.4 Test Organization

- **Separate Test Files:** Keep your tests in separate files from your code. This makes it easier to find and run your tests.
- **Organize by Module:** Organize your tests by module. This makes it easier to find the tests for a specific module.

### 5.5 Mocking and Stubbing

- **Use Mock Objects:** Use mock objects to isolate the code that you are testing from its dependencies. This makes it easier to test your code in isolation.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

- **Incorrect Docstring Formatting:** Using the wrong docstring format or inconsistent formatting.
- **Omitting Parameters:** Failing to document all parameters of a function or method.
- **Not Updating Documentation:** Failing to update the documentation when the code changes.
- **Circular Imports:**  Circular imports can confuse `pdoc`. Refactor your code to avoid them.

### 6.2 Edge Cases

- **Dynamic Code Generation:** If your code heavily uses dynamic code generation (`eval`, `exec`), `pdoc` may have difficulty introspecting it.
- **Metaclasses:** Complex metaclasses can also make it harder for `pdoc` to generate accurate documentation.

### 6.3 Version-Specific Issues

- **Check Compatibility:** Always check the compatibility of `pdoc` with your Python version.  Refer to the official `pdoc` documentation for version-specific information.

### 6.4 Compatibility Concerns

- **Third-Party Libraries:** Ensure that `pdoc` is compatible with any third-party libraries that you are using.

### 6.5 Debugging Strategies

- **Verbose Mode:** Run `pdoc` in verbose mode to see more detailed output. This can help you identify errors.
- **Check for Errors:** Check the output of `pdoc` for any errors or warnings.
- **Inspect Generated HTML:** Inspect the generated HTML to see if the documentation looks correct.

## 7. Tooling and Environment

### 7.1 Recommended Tools

- **IDE:** Use a good IDE (e.g., VS Code, PyCharm) with Python support.
- **Linting:** Use a linter (e.g., pylint, flake8) to catch errors and enforce coding standards.
- **Formatting:** Use a code formatter (e.g., black, autopep8) to format your code.

### 7.2 Build Configuration

- **`setup.py` or `pyproject.toml`:** Use a `setup.py` file or `pyproject.toml` to define your project's dependencies and metadata.
- **Include Documentation Files:** Make sure to include documentation files in your build configuration.

### 7.3 Linting and Formatting

- **Consistent Style:** Enforce a consistent coding style using a linter and code formatter.
- **Follow PEP 8:** Follow the PEP 8 style guide for Python code.
- **Use Docstring Checks:** Configure your linter to check for missing or incomplete docstrings.

### 7.4 Deployment Best Practices

- **Generate Documentation Automatically:** Automate the generation of documentation as part of your build process.
- **Host Documentation:** Host the generated documentation on a web server or documentation hosting service (e.g., Read the Docs).

### 7.5 CI/CD Integration

- **Integrate with CI/CD:** Integrate `pdoc` with your CI/CD pipeline to automatically generate and deploy documentation on every commit.

By following these best practices, you can ensure that your Python projects have high-quality, well-maintained documentation that is easy for users to understand and use.