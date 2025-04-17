---
description: Comprehensive best practices and coding standards for Python projects using the nose2 testing framework. Covers code organization, common patterns, performance, security, testing, and tooling.
globs: *test*.py
---
# nose2 Best Practices and Coding Standards

This document provides a comprehensive guide to best practices and coding standards for Python projects using the nose2 testing framework. Following these guidelines will help you write clean, maintainable, and efficient testing code.

## 1. Code Organization and Structure

### 1.1. Directory Structure

-   **Dedicated Test Directory:** Maintain a dedicated directory (e.g., `tests/`) at the project root to house all test files. This keeps test code separate from the application code.
-   **Mirror Application Structure:**  Mirror the structure of your application code within the test directory. For example, if your application has modules in `myapp/module1/` and `myapp/module2/`, create corresponding directories `tests/module1/` and `tests/module2/`.
-   **Test Modules:** Each application module should have a corresponding test module.  For `myapp/module.py`, the test module should be named `tests/test_module.py` or `tests/module_test.py`

Example:


myproject/
├── myapp/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   ├── test_module2.py
│   └── ...
├── ...


### 1.2. File Naming Conventions

-   **Test File Prefix:**  Name test files with a prefix like `test_` or a suffix `_test` (e.g., `test_module.py` or `module_test.py`).  nose2 automatically discovers files that follow this convention.
-   **Descriptive Names:** Use descriptive file names that clearly indicate the module or component being tested.  For example, `test_user_authentication.py` for tests related to user authentication.

### 1.3. Module Organization

-   **Single Responsibility:** Each test module should focus on testing a single application module or a closely related set of functionalities.
-   **Test Classes:** Organize tests within a module into test classes. Each class should test a specific aspect or component of the module. Inherit from `unittest.TestCase`.
-   **Test Functions/Methods:** Each test case should be a separate function (starting with `test_`) within a test class.  These methods should be small, focused, and test a single condition or scenario.

Example:

python
# tests/test_user.py
import unittest
from myapp import user

class TestUser(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        self.user = user.User("testuser", "password")

    def tearDown(self):
        # Teardown code to run after each test
        pass

    def test_user_creation(self):
        self.assertEqual(self.user.username, "testuser")
        self.assertTrue(self.user.check_password("password"))

    def test_invalid_password(self):
        self.assertFalse(self.user.check_password("wrong_password"))



### 1.4. Component Architecture

-   **Isolate Components:** Design your application to isolate components, making them easier to test independently. Use dependency injection or other techniques to decouple modules.
-   **Testable Interfaces:** Define clear and testable interfaces between components. This allows you to mock or stub dependencies during testing.

### 1.5. Code Splitting Strategies

-   **Functional Decomposition:** Split large test functions into smaller, more manageable functions that focus on specific aspects of the test.
-   **Parameterization:** Use parameterized tests (e.g., with `nose2.tools.params`) to test the same code with different inputs and expected outputs. This avoids code duplication and improves test coverage.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Arrange-Act-Assert (AAA):** Follow the AAA pattern in each test: _Arrange_ the test environment (set up data, mock dependencies), _Act_ (execute the code under test), and _Assert_ (verify the expected outcome).
-   **Test Fixtures:** Use `setUp` and `tearDown` methods in `unittest.TestCase` to create and clean up test fixtures (e.g., database connections, temporary files) before and after each test.
-   **Mocking:** Use mocking libraries like `unittest.mock` or `mock` to replace real dependencies with controlled substitutes during testing. This allows you to isolate the code under test and simulate various scenarios.

### 2.2. Recommended Approaches

-   **Test-Driven Development (TDD):** Write tests before writing the application code. This helps you define clear requirements and ensure that the code is testable.
-   **Behavior-Driven Development (BDD):** Use BDD frameworks like `behave` to write tests in a human-readable format that describes the expected behavior of the system.
-   **Focus on Edge Cases:**  Don't only test the happy path. Dedicate tests to verify edge cases, boundary conditions, and error handling.

### 2.3. Anti-patterns and Code Smells

-   **Over-testing:** Avoid testing implementation details that are likely to change. Focus on testing the behavior and expected outputs of the code.
-   **Fragile Tests:** Tests that break frequently due to minor code changes are fragile.  Strive for tests that are resilient to refactoring and implementation changes.
-   **Slow Tests:**  Slow tests can significantly slow down the development process. Optimize your tests to run quickly.  Use techniques like mocking and in-memory databases to speed up testing.
-   **Ignoring Test Failures:** Always investigate and fix failing tests. Don't ignore them or disable them without a proper reason.
-   **Redundant Assertions:** Avoid having more than one assertion per test case unless absolutely necessary.  Multiple assertions make it hard to pinpoint what failed.

### 2.4. State Management

-   **Isolate Test State:** Each test should be independent and not rely on the state left by previous tests. Use `setUp` and `tearDown` to ensure a clean state for each test.
-   **Avoid Global State:** Minimize the use of global variables or shared state in your application, as they can make tests harder to write and maintain.

### 2.5. Error Handling

-   **Test Exception Handling:** Write tests to verify that your code raises the correct exceptions in error scenarios.
-   **Use `assertRaises`:** Use `self.assertRaises(ExceptionType, callable, *args, **kwargs)` to assert that a specific exception is raised when calling a function or method.
-   **Context Managers for Asserting Exceptions:** Use context managers when checking for exceptions in more complex scenarios.

Example:

python
import unittest

class MyClass:
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestMyClass(unittest.TestCase):

    def setUp(self):
        self.my_class = MyClass()

    def test_divide_valid(self):
        self.assertEqual(self.my_class.divide(10, 2), 5)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError) as context:
            self.my_class.divide(10, 0)
        self.assertEqual(str(context.exception), "Cannot divide by zero")


## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Profiling:** Use profiling tools to identify performance bottlenecks in your tests. Optimize the slowest tests first.
-   **Mocking Expensive Operations:** Mock out slow operations (e.g., database queries, network requests) during testing to reduce test execution time.
-   **Caching Test Data:** Cache frequently used test data to avoid redundant calculations or data retrieval.

### 3.2. Memory Management

-   **Clean Up Resources:**  Ensure that tests properly clean up any resources they allocate (e.g., files, database connections) to avoid memory leaks.
-   **Limit Test Data Size:**  Use realistic but minimal test data to reduce memory consumption during testing.

### 3.3. Rendering Optimization (If Applicable)

-   This is generally not applicable to `nose2` testing itself, but if testing code involves UI rendering, strategies like reducing render frequency and complexity should be considered.

### 3.4. Bundle Size Optimization (If Applicable)

- This is generally not applicable to `nose2` testing itself.

### 3.5. Lazy Loading

-   **Lazy Initialization of Test Data:**  Defer the initialization of test data until it is actually needed by a test case. This can reduce test setup time and memory consumption.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

-   **Insecure Dependencies:** Use dependency scanning tools to identify and address vulnerabilities in third-party libraries used by your application or tests.
-   **Hardcoded Secrets:** Avoid hardcoding secrets (e.g., passwords, API keys) in your tests. Use environment variables or configuration files to store sensitive information.
-   **Insufficient Input Validation:**  Test that your application properly validates user inputs to prevent injection attacks and other security vulnerabilities.

### 4.2. Input Validation

-   **Test Boundary Conditions:**  Test that your application handles invalid or unexpected inputs gracefully, including boundary conditions, edge cases, and malicious inputs.

### 4.3. Authentication and Authorization

-   **Test Authentication Flows:** Write tests to verify that your application correctly authenticates users and grants access to authorized resources.
-   **Role-Based Access Control (RBAC):**  If your application uses RBAC, test that users with different roles have the appropriate permissions.

### 4.4. Data Protection

-   **Test Data Encryption:** If your application encrypts sensitive data, test that the encryption and decryption processes are working correctly.
-   **Secure Storage:**  Test that sensitive data is stored securely (e.g., using hashed passwords, encrypted databases).

### 4.5. Secure API Communication

-   **Test HTTPS:** If your application communicates with external APIs, test that it uses HTTPS to encrypt the communication.
-   **API Key Security:**  Protect API keys and other credentials used for API communication.  Don't store them directly in the code, and use appropriate access controls.

## 5. Testing Approaches

### 5.1. Unit Testing

-   **Focus on Individual Units:** Unit tests should focus on testing individual functions, methods, or classes in isolation.
-   **Mock Dependencies:** Use mocking to isolate the unit under test from its dependencies.
-   **Test All Code Paths:** Aim for high code coverage by testing all possible code paths and branches.

### 5.2. Integration Testing

-   **Test Interactions Between Components:** Integration tests verify that different components of the application work together correctly.
-   **Use Real Dependencies (Where Possible):**  Unlike unit tests, integration tests should use real dependencies (e.g., databases, external APIs) where feasible, or integration-specific lightweight replacements.

### 5.3. End-to-End (E2E) Testing

-   **Test the Entire System:** E2E tests simulate real user scenarios and test the entire application from end to end.
-   **Use Automated Testing Tools:** Use automated testing tools like Selenium or Cypress to automate E2E tests.

### 5.4. Test Organization

-   **Test Suites:**  Use test suites to group related tests together. This makes it easier to run specific sets of tests.
-   **Test Runners:** Use test runners (e.g., `nose2`, `pytest`) to discover and execute tests automatically.

### 5.5. Mocking and Stubbing

-   **`unittest.mock`:** Use the `unittest.mock` library (or the older `mock` library for Python 2) to create mock objects and stub out dependencies.
-   **`patch` Decorator:** Use the `patch` decorator to replace objects or attributes with mock objects during testing.
-   **`side_effect`:**  Use the `side_effect` attribute of mock objects to simulate different behaviors or raise exceptions when the mock object is called.

Example:

python
import unittest
from unittest.mock import patch

class MyClass:
    def get_data(self):
        # This method might make an external API call
        return "Real data"

    def process_data(self):
        data = self.get_data()
        return data.upper()

class TestMyClass(unittest.TestCase):

    @patch('__main__.MyClass.get_data')  # Patch the get_data method
    def test_process_data(self, mock_get_data):
        # Configure the mock to return a specific value
        mock_get_data.return_value = "Mock data"

        my_class = MyClass()
        result = my_class.process_data()

        self.assertEqual(result, "MOCK DATA")
        mock_get_data.assert_called_once() # Verify that the mock was called


## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Not Using `setUp` and `tearDown` Correctly:** Failing to properly set up test conditions or clean up resources after each test can lead to inconsistent test results.
-   **Ignoring Dependencies:** Forgetting to mock out dependencies can make tests slow, unreliable, and prone to external factors.
-   **Asserting Too Much in One Test:** Trying to test too many things in a single test case makes it difficult to identify the source of failures and maintain the tests.
-   **Writing Tests That Are Too Specific:** Writing tests that are tightly coupled to the implementation details can make them brittle and difficult to maintain.

### 6.2. Edge Cases

-   **Null Values:** Test how your application handles null or missing values.
-   **Empty Collections:** Test how your application handles empty lists, dictionaries, or sets.
-   **Large Data Sets:** Test how your application performs with large amounts of data.
-   **Special Characters:** Test how your application handles special characters in user inputs.
-   **Date and Time Zones:** Test how your application handles different date and time zones.

### 6.3. Version-Specific Issues

-   **Python Version Compatibility:** nose2 requires Python 3. Ensure compatibility across different minor versions of Python 3.
-   **Dependency Versions:** Be aware of compatibility issues between nose2 and its dependencies. Specify version constraints in your project's requirements file.

### 6.4. Compatibility Concerns

-   **`unittest` Compatibility:** nose2 extends `unittest`, so understanding `unittest` is fundamental. Be aware of any differences in behavior between standard `unittest` and nose2.

### 6.5. Debugging Strategies

-   **Use a Debugger:** Use a debugger (e.g., `pdb`, `ipdb`) to step through your tests and examine the state of the application.
-   **Print Statements:** Use `print` statements to output debugging information to the console.
-   **Logging:** Use the `logging` module to record debugging information to a file. This is useful for debugging tests that run in a CI/CD environment.
-   **Isolate the Problem:**  When a test fails, try to isolate the problem by running the test in isolation or by commenting out parts of the test.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **IDE:** PyCharm, VS Code with Python extension, or other Python-friendly IDE.
-   **Text Editor:**  Sublime Text, Atom, or other text editor.
-   **Debugger:**  `pdb` (Python Debugger) or `ipdb` (IPython Debugger).
-   **Mocking Library:** `unittest.mock` (built-in) or `mock` (for older Python versions).

### 7.2. Build Configuration

-   **`setup.py` or `pyproject.toml`:** Use `setup.py` or `pyproject.toml` to define your project's dependencies and build configuration.
-   **`requirements.txt`:** Use `requirements.txt` to list the project's dependencies. Generate this file using `pip freeze > requirements.txt`.
-   **Virtual Environments:** Use virtual environments (e.g., `venv`, `virtualenv`) to isolate the project's dependencies from the system's Python environment.

### 7.3. Linting and Formatting

-   **Pylint:** Use Pylint to enforce coding standards and detect potential errors in your code.  Configure Pylint with a `.pylintrc` file.
-   **pycodestyle (formerly pep8):** Use pycodestyle to check your code for compliance with PEP 8.
-   **Flake8:**  Use Flake8 as a wrapper around pycodestyle, pyflakes, and McCabe complexity checker.
-   **Black:**  Use Black to automatically format your code according to a consistent style.
-   **isort:** Use isort to automatically sort your imports.

### 7.4. Deployment Best Practices

-   This is generally out of scope of `nose2` itself, but automated testing as part of the deployment pipeline ensures high quality code.

### 7.5. CI/CD Integration

-   **GitHub Actions, GitLab CI, Jenkins:** Integrate nose2 into your CI/CD pipeline to automatically run tests whenever code is pushed or merged.
-   **Test Reporting:**  Configure your CI/CD system to generate test reports and display test results.
-   **Code Coverage Reporting:**  Integrate code coverage tools (e.g., `coverage.py`) into your CI/CD pipeline to track code coverage and identify areas that need more testing.
-   **Fail Fast:**  Configure your CI/CD pipeline to fail immediately if any tests fail. This prevents broken code from being deployed to production.

By following these best practices and coding standards, you can create robust and reliable Python testing code using nose2.