---
description: This rule provides guidance on best practices for using setuptools in Python projects, covering code organization, performance, security, testing, and common pitfalls.
globs: **/setup.py
---
# setuptools Best Practices

This document outlines best practices for using `setuptools` in Python projects. Following these guidelines ensures maintainable, performant, and secure code.

## Library Information:
- Name: setuptools
- Tags: development, build-tool, python, packaging

## 1. Code Organization and Structure:

- **Directory Structure:**
    - At the root:
        - `setup.py`:  The main setup script.
        - `setup.cfg`: Configuration file for `setup.py` (optional but recommended).
        - `pyproject.toml`: Build system configuration (modern approach).
        - `README.md` or `README.rst`: Project description.
        - `LICENSE.txt`: License information.
        - `.gitignore`: Specifies intentionally untracked files that Git should ignore.
    - A top-level package directory (e.g., `my_package`):
        - `my_package/`:
            - `__init__.py`: Makes the directory a Python package.
            - `module1.py`: Module containing code.
            - `module2.py`: Another module.
            - `data/`: Directory for package data (optional).
    - `tests/`:
        - `test_module1.py`: Unit tests for `module1.py`.
        - `test_module2.py`: Unit tests for `module2.py`.
        - `conftest.py`:  Configuration file for pytest.
        - `__init__.py` (optional).
    - `docs/` (optional):
        - Project documentation (e.g., using Sphinx).

- **File Naming Conventions:**
    - Python modules: `module_name.py` (snake_case).
    - Test files: `test_module_name.py` or `module_name_test.py`.
    - Package directories: `package_name/` (lowercase).
    - Configuration files: `setup.cfg`, `pyproject.toml`, `MANIFEST.in`.

- **Module Organization:**
    - Group related functions and classes within a module.
    - Keep modules focused on a specific responsibility.
    - Use clear and descriptive module names.
    - Utilize subpackages for logical grouping of modules within larger projects.

- **Component Architecture:**
    - Favor modular design, breaking down the project into reusable components.
    - Define clear interfaces between components.
    - Follow the Single Responsibility Principle (SRP) for each component.
    - Consider using a layered architecture if appropriate for your project's complexity.

- **Code Splitting Strategies:**
    - Split large modules into smaller, more manageable files.
    - Decompose complex functions into smaller, well-defined functions.
    - Extract reusable code into separate modules or packages.
    - Employ lazy loading for modules that are not immediately needed.

## 2. Common Patterns and Anti-patterns:

- **Design Patterns:**
    - **Factory Pattern:**  Use factory functions or classes to create instances of objects, especially when complex initialization is required.
    - **Dependency Injection:**  Inject dependencies into classes or functions to improve testability and reduce coupling.
    - **Facade Pattern:**  Provide a simplified interface to a complex subsystem.
    - **Singleton Pattern:**  Use sparingly; ensure thread safety if needed.
    - **Observer Pattern:**  Useful for event handling and asynchronous operations.

- **Recommended Approaches:**
    - **Declaring Dependencies:**  Use `install_requires` in `setup.py` or `pyproject.toml` (preferred) to specify project dependencies.
    - **Managing Package Data:**  Use `package_data` in `setup.py` to include non-code files (e.g., data files, templates) within your package.
    - **Creating Entry Points:** Define console scripts using `entry_points` in `setup.py` or `pyproject.toml` to create command-line tools.
    - **Versioning:** Follow Semantic Versioning (SemVer) to clearly communicate the nature of changes in each release.
    - **Using `pyproject.toml`:** Utilize `pyproject.toml` for build system configuration, build dependencies and other metadata. This aligns with modern packaging practices.

- **Anti-patterns and Code Smells:**
    - **Large `setup.py` Files:**  Keep `setup.py` concise; move complex logic to separate modules.
    - **Hardcoded Paths:**  Avoid hardcoding file paths; use relative paths or the `pkg_resources` module to access package data.
    - **Global State:**  Minimize the use of global variables; prefer passing state as arguments to functions or methods.
    - **Ignoring Errors:**  Always handle exceptions appropriately; never ignore errors without logging or taking corrective action.
    - **Over-engineering:** Avoid unnecessary complexity; keep the design simple and focused.

- **State Management:**
    - For CLI tools, use configuration files or environment variables to store persistent state.
    - For more complex applications, consider using a database or other persistent storage mechanism.
    - Avoid storing sensitive information in plain text configuration files; use encryption or secure storage.

- **Error Handling:**
    - Use `try...except` blocks to handle exceptions gracefully.
    - Log errors with sufficient context for debugging.
    - Raise custom exceptions to provide more specific error information.
    - Avoid catching generic exceptions (`except Exception:`) unless you re-raise them or log the error.

## 3. Performance Considerations:

- **Optimization Techniques:**
    - **Code Profiling:** Use profiling tools (e.g., `cProfile`) to identify performance bottlenecks.
    - **Algorithm Optimization:** Choose efficient algorithms and data structures.
    - **Caching:** Implement caching mechanisms to reduce redundant computations.
    - **Code Optimization:**  Use efficient code constructs and avoid unnecessary operations.
    - **Concurrency/Parallelism:**  Consider using threading or multiprocessing for CPU-bound tasks.

- **Memory Management:**
    - Use generators or iterators to process large datasets without loading everything into memory.
    - Release resources (e.g., file handles, network connections) promptly.
    - Avoid creating unnecessary copies of data.
    - Be mindful of memory leaks; use memory profiling tools to detect them.

- **Bundle Size Optimization:**
    - Minimize the size of your package by excluding unnecessary files (e.g., test files, documentation) from the distribution.
    - Use compression to reduce the size of package data.
    - Consider using a smaller version of a dependency or only requiring features of the dependency that you need.

- **Lazy Loading:**
    - Defer loading of modules or data until they are actually needed.
    - Use the `importlib` module to dynamically import modules at runtime.
    - Implement lazy properties to defer the computation of attribute values.

## 4. Security Best Practices:

- **Common Vulnerabilities:**
    - **Dependency Confusion:**  Prevent dependency confusion attacks by using a unique package name and verifying dependencies.
    - **Arbitrary Code Execution:**  Avoid using `eval()` or `exec()` to execute untrusted code.
    - **Injection Attacks:**  Sanitize user inputs to prevent injection attacks (e.g., SQL injection, command injection).
    - **Cross-Site Scripting (XSS):**  Encode output to prevent XSS attacks, particularly when dealing with web interfaces.

- **Input Validation:**
    - Validate all user inputs to ensure they conform to expected formats and ranges.
    - Use regular expressions or validation libraries to enforce input constraints.
    - Sanitize inputs to remove or escape potentially malicious characters.

- **Authentication and Authorization:**
    - Use secure authentication mechanisms (e.g., OAuth 2.0, JWT) to verify user identities.
    - Implement authorization checks to ensure that users only have access to the resources they are authorized to access.
    - Store passwords securely using hashing algorithms (e.g., bcrypt, scrypt).

- **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to secure communication between clients and servers.
    - Implement data masking or anonymization techniques to protect personally identifiable information (PII).

- **Secure API Communication:**
    - Use API keys or tokens to authenticate API requests.
    - Enforce rate limiting to prevent denial-of-service attacks.
    - Validate API requests and responses to ensure data integrity.

## 5. Testing Approaches:

- **Unit Testing:**
    - Write unit tests for each module and function to verify their correctness.
    - Use mocking or stubbing to isolate units of code from their dependencies.
    - Aim for high test coverage to ensure that all code paths are tested.
    - Use `pytest` and `unittest` modules.

- **Integration Testing:**
    - Write integration tests to verify the interactions between different components of the system.
    - Test the integration with external dependencies (e.g., databases, APIs).
    - Use test doubles or test environments to simulate external dependencies.

- **End-to-End Testing:**
    - Write end-to-end tests to verify the functionality of the entire system from the user's perspective.
    - Use browser automation tools (e.g., Selenium, Playwright) to simulate user interactions.
    - Test the system in a realistic environment.

- **Test Organization:**
    - Organize tests into separate directories (e.g., `tests/`).
    - Use descriptive test names to clearly indicate what each test is verifying.
    - Group related tests into test classes or modules.
    - Use test fixtures to set up and tear down test environments.

- **Mocking and Stubbing:**
    - Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`) to create mock objects that simulate the behavior of dependencies.
    - Use stubbing to replace dependencies with simplified versions that return predefined values.
    - Avoid over-mocking; only mock dependencies that are difficult to test directly.

## 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes:**
    - **Incorrect Dependency Specifications:**  Ensure that dependencies are correctly specified in `install_requires` or `pyproject.toml` with accurate version constraints.
    - **Missing Package Data:**  Remember to include non-code files in `package_data` if they are required at runtime.
    - **Inconsistent Versioning:**  Follow Semantic Versioning (SemVer) consistently and update version numbers appropriately.
    - **Ignoring Encoding Issues:**  Handle text encoding correctly to avoid errors when dealing with non-ASCII characters.
    - **Failing to Test:**  Write comprehensive tests to catch errors early and prevent regressions.

- **Edge Cases:**
    - **Platform-Specific Issues:**  Test your package on different operating systems and Python versions to identify platform-specific issues.
    - **Dependency Conflicts:**  Be aware of potential dependency conflicts and use virtual environments to isolate your project's dependencies.
    - **Unicode Handling:**  Handle Unicode characters correctly, especially when dealing with user inputs or file names.
    - **Resource Limits:**  Be mindful of resource limits (e.g., memory, file handles) when processing large datasets.

- **Version-Specific Issues:**
    - Be aware of compatibility issues between different versions of setuptools.
    - Consult the setuptools documentation for version-specific recommendations.

- **Compatibility Concerns:**
    - Test your package with different versions of Python to ensure compatibility.
    - Be aware of potential conflicts with other libraries or frameworks.
    - Use conditional imports or feature detection to handle compatibility issues gracefully.

- **Debugging Strategies:**
    - Use a debugger to step through your code and inspect variables.
    - Add logging statements to track the flow of execution and identify errors.
    - Use unit tests to isolate and reproduce bugs.
    - Consult online resources (e.g., Stack Overflow) and the setuptools documentation for troubleshooting tips.

## 7. Tooling and Environment:

- **Recommended Development Tools:**
    - **Virtual Environment Managers:** `venv`, `virtualenv`, `conda` or Poetry to create isolated Python environments.
    - **Package Managers:** `pip` or Poetry to install and manage dependencies.
    - **Code Editors:** VS Code, PyCharm, Sublime Text, or other IDEs with Python support.
    - **Debuggers:** `pdb` or IDE-integrated debuggers.
    - **Profilers:** `cProfile` to identify performance bottlenecks.
    - **Testing Frameworks:** `pytest` or `unittest` for writing and running tests.
    - **Linters and Formatters:** `flake8`, `pylint`, `black`, and `isort` to enforce code style and quality.
    - **Build System:** `build` to build distributions.

- **Build Configuration:**
    - Use `setup.cfg` or `pyproject.toml` to configure the build process.
    - Specify dependencies, package data, and entry points in the configuration file.
    - Use build scripts to automate build tasks.

- **Linting and Formatting:**
    - Use `flake8` or `pylint` to enforce code style guidelines.
    - Use `black` to automatically format your code.
    - Use `isort` to sort imports alphabetically.

- **Deployment:**
    - Use `twine` to securely upload your package to PyPI.
    - Use virtual environments to isolate your application's dependencies in production.
    - Consider using containerization (e.g., Docker) to create reproducible deployment environments.

- **CI/CD:**
    - Use CI/CD systems (e.g., GitHub Actions, Jenkins, Travis CI, CircleCI) to automate building, testing, and deploying your package.
    - Configure CI/CD pipelines to run tests, linters, and formatters on every commit.
    - Automate the release process using CI/CD.

By adhering to these best practices, you can effectively leverage `setuptools` to create well-structured, maintainable, and high-quality Python packages.