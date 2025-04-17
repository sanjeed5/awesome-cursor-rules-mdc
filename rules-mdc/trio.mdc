---
description: This rule provides comprehensive best practices for developing with the Trio asynchronous I/O library in Python, covering code organization, performance, testing, and common pitfalls.
globs: *.py
---
- Use `flake8-async` to check for issues in Trio code. `@file:flake8_async_rules.mdc`
- Utilize Trio's nursery feature for managing concurrent tasks to prevent orphaned tasks and improve error handling. Use `async with trio.open_nursery() as nursery: nursery.start_soon(my_task)`. See Trio documentation for advanced usage.
- Clearly distinguish between async and synchronous functions. Async functions should primarily be used for I/O-bound operations. CPU-bound tasks should run in separate processes or threads to avoid blocking the event loop.
- Leverage Trio's testing utilities (`trio.testing`) to ensure that your async code behaves as expected. Use tools like `trio.testing.MockClock` to control time in tests.
- Adhere to PEP 8 for consistent code formatting. Use a linter and formatter like `black` to ensure compliance.
- Always use absolute imports for better readability and maintainability: `import mypkg.sibling`. Use explicit relative imports (`from . import sibling`) only when necessary in complex package layouts.
- Avoid wildcard imports (`from <module> import *`).
- Place module-level dunder names (e.g., `__all__`, `__author__`, `__version__`) after the module docstring but before any import statements (except `from __future__` imports).
- Use 4 spaces for indentation.
- Limit lines to a maximum of 79 characters (or 99 if agreed upon within a team). For docstrings and comments, limit lines to 72 characters.
- Surround top-level function and class definitions with two blank lines. Method definitions inside a class are surrounded by a single blank line.
- Use blank lines in functions to indicate logical sections.
- Code in the core Python distribution should always use UTF-8.
- Follow the naming conventions as described in PEP 8. Class names should use CapWords. Function and variable names should be lowercase with underscores.
- Always use `self` for the first argument to instance methods and `cls` for the first argument to class methods.
- Use one leading underscore for non-public methods and instance variables. Use two leading underscores to invoke Python’s name mangling for avoiding name clashes with subclasses.
- Constants should be defined on a module level and written in all capital letters with underscores separating words.
- Always decide whether a class’s methods and instance variables should be public or non-public. If in doubt, choose non-public.
- Comparisons to singletons like `None` should always be done with `is` or `is not`, never the equality operators.
- Use `is not` operator rather than `not ... is`.
- When implementing ordering operations with rich comparisons, it is best to implement all six operations (`__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`).
- Always use a `def` statement instead of an assignment statement that binds a lambda expression directly to an identifier.
- Derive exceptions from `Exception` rather than `BaseException`. Design exception hierarchies based on the distinctions that code catching the exceptions is likely to need.
- Use exception chaining appropriately (`raise X from Y`).
- When catching exceptions, mention specific exceptions whenever possible instead of using a bare `except:` clause.
- Limit the `try` clause to the absolute minimum amount of code necessary.
- When a resource is local to a particular section of code, use a `with` statement to ensure it is cleaned up promptly and reliably after use.
- Be consistent in return statements. Either all `return` statements in a function should return an expression, or none of them should.
- Use `''.startswith()` and `''.endswith()` instead of string slicing to check for prefixes or suffixes.
- Object type comparisons should always use `isinstance()` instead of comparing types directly.
- For sequences (strings, lists, tuples), use the fact that empty sequences are false.
- Don’t write string literals that rely on significant trailing whitespace.
- Don’t compare boolean values to `True` or `False` using `==`.
- Function annotations should use PEP 484 syntax.
- Annotations for module-level variables, class and instance variables, and local variables should have a single space after the colon.

## Code Organization and Structure

- **Directory Structure:**
    - A typical project structure might look like this:
        
        my_project/
        ├── src/
        │   ├── __init__.py
        │   ├── main.py  # Entry point
        │   ├── utils.py # Utility functions
        │   ├── services/
        │   │   ├── __init__.py
        │   │   ├── http_service.py
        │   │   └── db_service.py
        │   └── models/
        │       ├── __init__.py
        │       └── user.py
        ├── tests/
        │   ├── __init__.py
        │   ├── test_main.py
        │   ├── test_utils.py
        │   ├── services/
        │   │   ├── test_http_service.py
        │   │   └── test_db_service.py
        │   └── models/
        │       └── test_user.py
        ├── README.md
        ├── pyproject.toml  # Project configuration (poetry, pipenv)
        └── .gitignore
        
    - Use a `src` directory to hold the main application code.  This helps separate application code from configuration and documentation.
    - Keep tests in a separate `tests` directory, mirroring the structure of `src`.

- **File Naming Conventions:**
    - Use lowercase names for files and modules (e.g., `http_service.py`, `utils.py`).
    - Test files should be prefixed with `test_` (e.g., `test_http_service.py`).

- **Module Organization:**
    - Organize code into logical modules based on functionality (e.g., `services`, `models`, `utils`).
    - Use `__init__.py` files to make directories importable as packages.
    - Avoid circular dependencies between modules.

- **Component Architecture:**
    - Consider using a layered architecture (e.g., presentation, business logic, data access) to separate concerns.
    - Dependency Injection: Use dependency injection to make components more testable and reusable.

- **Code Splitting:**
    - Break down large modules into smaller, more manageable files.
    - Consider splitting modules based on functionality or responsibilities.

## Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Asynchronous Factory:**  Use factories to create asynchronous resources (e.g., connections) to manage initialization efficiently.
        python
        async def create_db_connection():
            conn = await connect_to_db()
            return conn
        
        async def main():
            conn = await create_db_connection()
            # Use the connection
        
    - **Resource Pooling:** Implement resource pooling for database connections or network connections to reduce overhead.

- **Recommended Approaches:**
    - Use nurseries for structured concurrency.
    - Use streams (`trio.Stream`) for communication between tasks.
    - Use channels (`trio.QueueChannel`) for passing data between tasks.

- **Anti-patterns:**
    - **Sleeping without Cancellation:** Avoid using `time.sleep()` directly, as it blocks the event loop and ignores cancellation. Use `await trio.sleep()` instead.
    - **Long-Running Synchronous Operations:**  Don't perform CPU-bound operations directly in async functions. Offload them to separate threads or processes.
    - **Ignoring Cancellation:** Ensure that your async functions handle cancellation requests (`trio.Cancelled`).
    - **Unstructured Concurrency:**  Avoid spawning tasks without proper management (e.g., without using nurseries). This can lead to orphaned tasks and difficult debugging.

- **State Management:**
    - Immutable Data: Prefer immutable data structures to avoid race conditions and simplify reasoning about state.
    - Task-Local Storage: Use `trio.TaskLocal` to store task-specific data.
    - Thread-Safe Data Structures: If shared mutable state is necessary, use thread-safe data structures (e.g., `trio.Lock`, `trio.Semaphore`).

- **Error Handling:**
    - Use `try...except` blocks to handle exceptions within async functions.
    - Propagate exceptions appropriately to the nursery for proper error handling.
    - Consider using exception groups to handle multiple exceptions that occur concurrently.

## Performance Considerations

- **Optimization Techniques:**
    - Minimize context switching:  Reduce unnecessary `await` calls.
    - Use efficient data structures: Choose appropriate data structures for your specific use case.
    - Avoid excessive copying:  Use views or iterators when possible to avoid copying large data structures.

- **Memory Management:**
    - Release resources promptly: Use `with` statements or `try...finally` blocks to ensure that resources are released even if exceptions occur.
    - Avoid circular references: Be mindful of potential circular references, which can prevent garbage collection.

## Security Best Practices

- **Common Vulnerabilities:**
    - **Race Conditions:**  Be aware of potential race conditions when accessing shared mutable state.
    - **Cancellation Errors:** Improper cancellation handling can lead to resource leaks or incorrect program behavior.

- **Input Validation:**
    - Validate all external inputs to prevent injection attacks and other security vulnerabilities.
    - Sanitize user inputs before using them in database queries or other sensitive operations.

- **Authentication and Authorization:**
    - Use established authentication and authorization libraries.
    - Implement proper access controls to protect sensitive data.

- **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure communication protocols (e.g., HTTPS).

- **Secure API Communication:**
    - Validate all API requests and responses.
    - Implement rate limiting to prevent abuse.

## Testing Approaches

- **Unit Testing:**
    - Use `trio.testing` to write unit tests for async functions.
    - Use `trio.testing.MockClock` to control time in tests.
    - Mock external dependencies to isolate the code being tested.

- **Integration Testing:**
    - Test the interaction between different components of your application.
    - Use real or simulated external services to test the integration with external systems.

- **End-to-End Testing:**
    - Test the entire application flow from the user interface to the database.

- **Test Organization:**
    - Organize tests in a directory structure that mirrors the structure of your application code.
    - Write clear and concise test names that describe the behavior being tested.

- **Mocking and Stubbing:**
    - Use mocking libraries like `unittest.mock` to replace external dependencies with mock objects.
    - Use stubbing to provide predefined responses for external dependencies.

## Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Blocking the event loop with synchronous operations.
    - Ignoring cancellation requests.
    - Using `time.sleep()` instead of `trio.sleep()`.
    - Not handling exceptions properly.

- **Edge Cases:**
    - Handling timeouts and deadlines.
    - Dealing with cancellation in complex workflows.
    - Managing resources in the presence of exceptions and cancellations.

- **Version-Specific Issues:**
    - Be aware of any known bugs or limitations in specific versions of Trio.

- **Compatibility Concerns:**
    - Ensure compatibility between Trio and other libraries you are using.

- **Debugging Strategies:**
    - Use debuggers like `pdb` or `ipdb` to step through your code and inspect variables.
    - Use logging to track the execution flow of your application.
    - Use Trio's built-in debugging tools to identify performance bottlenecks and other issues.

## Tooling and Environment

- **Recommended Development Tools:**
    - VS Code with the Python extension.
    - PyCharm.
    - IPython or Jupyter Notebook for interactive development.

- **Build Configuration:**
    - Use a build system like `poetry` or `pipenv` to manage dependencies.

- **Linting and Formatting:**
    - Use `flake8` and `black` to ensure consistent code style.
    - Configure your editor to automatically format code on save.

- **Deployment:**
    - Use a process manager like `systemd` or `supervisor` to manage your application.

- **CI/CD Integration:**
    - Use CI/CD tools like GitHub Actions or GitLab CI to automate testing and deployment.