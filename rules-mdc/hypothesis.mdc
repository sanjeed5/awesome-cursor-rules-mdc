---
description: Comprehensive guide covering best practices for the Hypothesis Python library, including coding standards, testing, performance, and security.  Provides actionable guidance for developers to write maintainable, robust, and efficient property-based tests.
globs: *.py
---
# Hypothesis Library Best Practices and Coding Standards

This document provides comprehensive guidelines for using the Hypothesis Python library effectively.  Following these best practices will lead to more maintainable, robust, and efficient property-based tests.

## I. General Python Coding Standards (Applicable to Hypothesis Projects)

These are general Python coding standards that should be adhered to in any Python project, including those using Hypothesis.

*   **PEP 8 Compliance:**  Adhere strictly to PEP 8 style guidelines for Python code. This includes:
    *   Indentation: Use 4 spaces per indentation level.
    *   Line Length: Limit lines to a maximum of 79 characters (or 72 for comments and docstrings).
    *   Blank Lines:  Use blank lines to separate top-level function and class definitions (two lines) and method definitions inside a class (one line).
    *   Imports: Group imports into standard library, third-party, and local application/library-specific imports, separated by blank lines.
    *   Whitespace: Avoid extraneous whitespace.
    *   Naming Conventions: Follow appropriate naming conventions for variables, functions, classes, and constants.
*   **PEP 257 Compliance:** Follow PEP 257 for docstring conventions.
    *   Write docstrings for all public modules, functions, classes, and methods.
    *   Use triple double quotes (`"""`) for docstrings.
    *   Include a summary line, use case (if appropriate), arguments, return type and semantics (unless None is returned), and examples in multiline docstrings.
*   **Use Python 3.12+:** Ensure that your project uses a modern version of Python (3.12 or higher) for access to the latest features and security updates.
*   **Virtual Environments:** Use virtual environments (e.g., `venv` or `virtualenv`) to isolate project dependencies.
*   **Dependency Management:** Use a dependency management tool like `pip` with a `requirements.txt` or `pyproject.toml` file to specify project dependencies. It's recommended to use a modern package installer such as [UV](https://astral.sh/blog/uv) for faster and more reproducible builds.
*   **Code Reviews:**  Conduct regular code reviews to ensure code quality and adherence to coding standards.
*   **Type Hinting:** Use type hints to improve code readability and enable static analysis.
*   **Linters and Formatters:** Use linters (e.g., `flake8`, `pylint`) and formatters (e.g., `black`, `autopep8`) to automatically enforce coding standards.
*   **Error Handling:** Implement robust error handling using `try...except` blocks.
*   **Logging:** Use the `logging` module for structured logging.
*   **Clear and Concise Code:** Write code that is easy to understand and maintain.
*   **Modularity:** Design your code with a modular architecture.
*   **Documentation:** Write clear and comprehensive documentation.
*   **Follow SOLID principles** When designing code and modules.

## II. Hypothesis-Specific Best Practices

These guidelines are specifically tailored to the Hypothesis library for property-based testing.

### A. Writing Properties

*   **Focus on Properties, Not Examples:** Hypothesis excels when you define *properties* that should always hold true for a wide range of inputs, rather than just providing a fixed set of examples.  A property describes a general relationship that should be invariant.
*   **Define Strategies Clearly:** Strategies are the core of Hypothesis. Define strategies that generate diverse and representative data for your properties. Hypothesis provides many built-in strategies, and you can combine and customize them.
*   **Shrinking is Crucial:** Hypothesis shrinks failing examples to the smallest and simplest form that still causes the failure.  This simplifies debugging.  Ensure your strategies are designed to allow effective shrinking.
*   **Avoid Direct Assertion:** Generally, avoid directly asserting specific values within your properties.  Instead, focus on asserting relationships between inputs and outputs. This strengthens the test and covers more potential cases.
*   **Use Assume Sparingly:** The `assume()` function tells Hypothesis to discard certain examples. Overuse of `assume()` can make your tests less effective and potentially mask bugs. Try to refine your strategies instead of filtering excessively.
*   **Make Properties Declarative:** Properties should describe *what* should happen, not *how* it should happen.  This makes the tests more robust and easier to understand.
*   **Test Invariants, Not Implementations:** Properties should test the high-level invariants of your code, rather than implementation details. This makes the tests more resilient to code changes.
*   **Minimize Side Effects:** Properties should be as pure as possible, with minimal side effects. This makes the tests more predictable and easier to debug.
*   **Consider Edge Cases:** Think about edge cases and boundary conditions that might violate your properties.  Design your strategies to include these cases.
*   **Use Composite Strategies:** Composite strategies allow you to combine multiple strategies to generate complex data structures. Use them to create realistic and diverse test cases.
*   **Label Strategies:** Label strategies with descriptive names to improve the readability of your test output.
*   **Use Dataclasses for Structured Data:** Define dataclasses to represent structured data in your properties. This improves code readability and maintainability.

### B. Strategy Design

*   **Start with Built-in Strategies:** Hypothesis provides a rich set of built-in strategies for generating common data types.  Start with these strategies and customize them as needed.
*   **Constrain Strategies Appropriately:** Use constraints (e.g., `min_value`, `max_value`, `length`) to narrow the range of values generated by your strategies and improve test performance.
*   **Use `sampled_from` for Discrete Values:** The `sampled_from` strategy is useful for generating discrete values from a list or set.
*   **Combine Strategies with `one_of`:** The `one_of` strategy allows you to combine multiple strategies to generate a variety of data types.
*   **Use `lists`, `sets`, and `dictionaries` to Generate Collections:** These strategies generate collections of data.  Specify the element type and size constraints as needed.
*   **Consider `recursive` Strategies for Nested Structures:** The `recursive` strategy allows you to generate nested data structures, such as trees or graphs.
*   **Write Custom Strategies When Necessary:** If the built-in strategies are not sufficient, you can write custom strategies to generate data specific to your application.
*   **Use `register_type_strategy` for Custom Types:** Register custom strategies for your custom types to make them available to other strategies.
*   **Think About Shrinking When Designing Strategies:** Design your strategies with shrinking in mind.  The goal is to shrink failing examples to the smallest and simplest form possible.

### C. Code Organization

*   **Separate Tests from Implementation:** Keep your hypothesis tests in a separate directory from your implementation code (e.g., `tests/`).
*   **Organize Tests by Module:**  Mirror the structure of your implementation code in your test directory.  Create a test module for each implementation module.
*   **Use Descriptive Test Names:**  Give your tests descriptive names that indicate the property being tested.
*   **Use Fixtures for Setup and Teardown:** Use pytest fixtures to handle setup and teardown tasks for your tests.
*   **Create Helper Functions for Common Tasks:**  Create helper functions to encapsulate common tasks in your tests.
*   **Use a `conftest.py` file:** Use a `conftest.py` file to define fixtures and other configuration options that are shared across multiple test modules.
*   **Keep Test Modules Small:**  Break large test modules into smaller, more manageable modules.

### D. Error Handling

*   **Test for Expected Exceptions:** Use `pytest.raises` to assert that your code raises the expected exceptions under certain conditions.
*   **Use Context Managers for Exception Handling:**  Use context managers to handle exceptions and ensure that resources are properly released.
*   **Log Errors and Warnings:** Use the `logging` module to log errors and warnings in your tests.
*   **Use `report` to Add Context to Failures:** Use `hypothesis.report` to add information about the current state of the system under test to the failure report.
*   **Fail Fast:**  Write tests that fail quickly when a property is violated.

### E. Performance Considerations

*   **Minimize Computation in Properties:**  Properties should be as lightweight as possible to avoid slowing down the tests.
*   **Use Caching to Avoid Redundant Computations:** Use caching to avoid redundant computations in your properties.
*   **Optimize Strategies for Performance:**  Optimize your strategies to generate data efficiently.
*   **Use `max_examples` to Limit Test Run Time:**  Use the `max_examples` setting to limit the number of examples generated by Hypothesis and prevent tests from running indefinitely.
*   **Profile Slow Tests:**  Use profiling tools to identify slow tests and optimize their performance.
*   **Consider Data Generation Overhead:** Be aware that complex strategy definitions can have a significant performance overhead.
*    **Avoid Unnecessary `assume` Calls:**  Excessive use of `assume` can lead to wasted test cycles.

### F. Security Best Practices

*   **Test for Input Validation:** Use Hypothesis to test the input validation logic of your code. Generate a wide range of inputs, including invalid inputs, to ensure that your code handles them correctly.
*   **Test for Common Vulnerabilities:**  Use Hypothesis to test for common vulnerabilities, such as SQL injection, cross-site scripting (XSS), and buffer overflows.
*   **Use Strategies to Generate Malicious Inputs:**  Create strategies that generate malicious inputs to test the security of your code.
*   **Sanitize Inputs:** Sanitize all inputs to prevent vulnerabilities.
*   **Escape Outputs:** Escape all outputs to prevent vulnerabilities.
*   **Use Secure API Communication:** Use HTTPS for secure API communication.
*   **Implement Authentication and Authorization:** Implement authentication and authorization to protect your data.
*   **Limit User Privileges:** Limit user privileges to the minimum necessary to perform their tasks.
*   **Monitor Your Application for Security Threats:** Monitor your application for security threats and respond promptly to any incidents.

### G. Common Pitfalls and Gotchas

*   **Overuse of `assume()`:** As mentioned before, overuse of `assume()` can hide bugs.
*   **Complex Strategy Definitions:**  Complex strategy definitions can be difficult to understand and maintain.
*   **Unrealistic Test Data:**  Generating unrealistic test data can lead to false positives or false negatives.
*   **Slow Test Execution:**  Slow test execution can make it difficult to iterate quickly.
*   **Lack of Understanding of Shrinking:**  A lack of understanding of shrinking can make it difficult to debug failing tests.
*   **Not Testing Edge Cases:**  Failing to test edge cases can lead to bugs in production.
*   **Ignoring Hypothesis Output:** Ignoring the Hypothesis output can cause you to miss important information about failing tests.
*   **Incorrect Type Annotations:** Incorrect type annotations can cause Hypothesis to generate incorrect data.
*   **Misunderstanding of Strategy Composition:** Misunderstanding how strategies are composed can lead to unexpected results.

### H. Tooling and Environment

*   **pytest:** Use pytest as your test runner. It integrates seamlessly with Hypothesis and provides a rich set of features for writing and running tests.
*   **hypothesis-auto:** Use the `hypothesis-auto` extension to automatically generate strategies for your dataclasses and other data types.
*   **black:** Use black to automatically format your code and ensure PEP 8 compliance.
*   **flake8:** Use flake8 to lint your code and identify potential problems.
*   **mypy:** Use mypy to statically type check your code and identify type errors.
*   **tox:** Use tox to automate testing in multiple environments.
*   **CI/CD:** Integrate your tests into your CI/CD pipeline to ensure that your code is always tested before it is deployed.
*   **VS Code or PyCharm:** Use a good IDE (like VS Code with the Python extension, or PyCharm) for code editing, debugging, and testing.

### I. Testing Approaches

*   **Unit Testing:**  Focus on testing individual units of code in isolation.
*   **Integration Testing:** Test how different units of code interact with each other.
*   **End-to-End Testing:** Test the entire application from end to end.
*   **Property-Based Testing:** Use Hypothesis to generate a wide range of inputs and test the properties of your code.
*   **Mutation Testing:**  Use mutation testing to assess the quality of your tests.

### J. Recommended Workflow

1.  **Start with a Property:** Define the property that you want to test.
2.  **Define a Strategy:** Create a strategy to generate data for your property.
3.  **Write the Test:** Write the test using Hypothesis and pytest.
4.  **Run the Test:** Run the test and see if it passes.
5.  **Fix Any Bugs:** If the test fails, fix the bug in your code.
6.  **Repeat:** Repeat steps 1-5 until you are confident that your code is correct.

## III. Example Hypothesis Test

python
import pytest
from hypothesis import given, strategies as st

def add(x, y):
    return x + y

@given(st.integers(), st.integers())
def test_addition_is_commutative(x, y):
    assert add(x, y) == add(y, x)


## IV. Conclusion

By following these best practices, you can use Hypothesis to write more effective and robust property-based tests that improve the quality and reliability of your Python code.