---
description: This rule provides best practices and coding standards for developing with the AnyIO library, focusing on structured concurrency, task management, and proper use of synchronization primitives to create robust, cross-backend asynchronous applications.
globs: *.py
---
# AnyIO Best Practices and Coding Standards

This document provides comprehensive guidelines for developing with the AnyIO library, covering code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and recommended tooling.

## Library Information:
- Name: anyio
- Tags: python, async, compatibility-layer

## 1. Code Organization and Structure

- **Directory Structure:** Organize your project to separate concerns clearly.
    - `src/`: Contains the main application code.
    - `tests/`: Contains unit, integration, and end-to-end tests.
    - `examples/`: Contains example code showcasing AnyIO features.
    - `docs/`: Contains project documentation (using Sphinx, for example).
    - `tasks.py`: Invoke tasks for building, testing, and deploying (using Invoke).
- **File Naming Conventions:**
    - Use descriptive and consistent file names.
    - `my_component.py` for a component's source code.
    - `my_component_test.py` for its tests.
    - `conftest.py` in the `tests/` directory for test fixtures.
- **Module Organization:**
    - Group related functionality into modules.
    - Use packages to further structure your code.
    - Employ clear and concise module-level docstrings.
- **Component Architecture:**
    - Design components with clear responsibilities.
    - Use interfaces or abstract base classes for decoupling.
    - Consider using dependency injection to improve testability.
- **Code Splitting:**
    - Break down large functions into smaller, more manageable ones.
    - Decompose complex modules into simpler sub-modules.
    - Consider lazy-loading modules or components to improve startup time.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Task Groups (Nurseries):**  Use task groups (`anyio.create_task_group()`) for structured concurrency. This is the most crucial pattern in AnyIO.
    - **Resource Acquisition Is Initialization (RAII):**  Use `async with` statements for managing resources (e.g., sockets, files) to ensure proper cleanup, even in case of exceptions or cancellations.
    - **Observer Pattern:**  Use AnyIO's `Event` or `Condition` primitives to implement observer patterns for asynchronous communication.
    - **Producer-Consumer Pattern:**  Use `anyio.Queue` for implementing producer-consumer patterns with asynchronous tasks.
- **Recommended Approaches:**
    - Use `anyio.sleep()` for non-blocking delays.
    - Employ `anyio.to_thread.run_sync()` to run blocking code in a separate thread to avoid blocking the event loop.
    - Utilize `anyio.Path` for asynchronous file I/O operations.
- **Anti-patterns:**
    - **Blocking the Event Loop:** Avoid performing long-running synchronous operations in the main event loop. Use `anyio.to_thread.run_sync()` to offload such tasks to worker threads.
    - **Ignoring Exceptions:** Always handle exceptions gracefully within tasks and task groups. Unhandled exceptions can crash your application.
    - **Unstructured Concurrency:** Avoid creating tasks without a task group. This can lead to resource leaks and difficult-to-debug issues.
    - **Spin Locks:** Do not use busy-waiting or spin locks in asynchronous code. Use synchronization primitives like `Lock` or `Condition` instead.
- **State Management:**
    - Use immutable data structures whenever possible to avoid race conditions.
    - Employ `Lock` or `Condition` objects to protect shared mutable state.
    - Consider using atomic operations for simple state updates.
- **Error Handling:**
    - Wrap code in `try...except` blocks to catch exceptions.
    - Use `try...finally` blocks to ensure cleanup, even in case of exceptions.
    - Leverage task groups to handle exceptions across multiple tasks.
    - Implement retry mechanisms for transient errors.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Asynchronous I/O:** Utilize AnyIO's asynchronous I/O functions (e.g., `anyio.Path`, `anyio.open_file`) for optimal performance when dealing with I/O-bound operations.
    - **Connection Pooling:** Implement connection pooling for database connections or other network resources to reduce overhead.
    - **Caching:** Cache frequently accessed data to minimize I/O operations.
    - **Minimize Context Switching:** Reduce unnecessary context switching by batching operations and avoiding excessive `await` calls.
- **Memory Management:**
    - Avoid creating large, unnecessary data structures.
    - Use generators or iterators to process large datasets in a memory-efficient manner.
    - Properly close resources (e.g., sockets, files) to release memory.
    - Use the `del` statement or `weakref` to break circular references and allow garbage collection.
- **Rendering Optimization:** N/A (AnyIO is not directly involved in rendering).
- **Bundle Size Optimization:** N/A (AnyIO itself is a dependency, not a frontend bundle).
- **Lazy Loading:**
    - Use lazy loading to defer the initialization of components until they are needed.
    - Implement code splitting to load modules on demand.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Denial of Service (DoS):** Protect against DoS attacks by limiting the number of concurrent connections or requests.
    - **Injection Attacks:** Sanitize user inputs to prevent injection attacks (e.g., SQL injection, command injection).
    - **Cross-Site Scripting (XSS):** N/A (AnyIO is not directly involved in front-end development).
- **Input Validation:**
    - Validate all user inputs to ensure they conform to expected formats and ranges.
    - Use input validation libraries to simplify the process.
    - Escape or sanitize user inputs before using them in database queries or system commands.
- **Authentication and Authorization:**
    - Implement authentication to verify the identity of users.
    - Implement authorization to control access to resources based on user roles.
    - Use secure password hashing algorithms (e.g., bcrypt, Argon2).
- **Data Protection:**
    - Use encryption to protect sensitive data at rest and in transit.
    - Store cryptographic keys securely.
    - Comply with relevant data privacy regulations (e.g., GDPR, CCPA).
- **Secure API Communication:**
    - Use HTTPS for all API communication to encrypt data in transit.
    - Implement proper authentication and authorization mechanisms for API endpoints.
    - Protect against Cross-Site Request Forgery (CSRF) attacks if your API is used by web applications.

## 5. Testing Approaches

- **Unit Testing:**
    - Write unit tests for individual components to verify their functionality.
    - Use mocking or stubbing to isolate components from their dependencies.
    - Aim for high test coverage to ensure that all code paths are tested.
- **Integration Testing:**
    - Write integration tests to verify the interaction between different components.
    - Test the integration with external services (e.g., databases, APIs).
- **End-to-End Testing:**
    - Write end-to-end tests to verify the entire application flow.
    - Simulate real user interactions to test the application from a user's perspective.
- **Test Organization:**
    - Organize tests into separate modules based on the components or features they test.
    - Use a consistent naming convention for test files and functions.
    - Employ test fixtures to set up and tear down test environments.
- **Mocking and Stubbing:**
    - Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`) to replace dependencies with mock objects.
    - Stub external services or APIs to simulate different scenarios.
    - Verify that mock objects are called with the expected arguments.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - **Not using `async with` for resources:** Forgetting to use `async with` for resources that require asynchronous cleanup can lead to resource leaks.
    - **Mixing asyncio and Trio idioms:** Avoid mixing idioms from different asynchronous frameworks. Stick to AnyIO's API.
    - **Incorrectly using `to_thread.run_sync`:**  Ensure that the code passed to `to_thread.run_sync` is actually blocking.
- **Edge Cases:**
    - **Cancellation:** Be aware of how cancellation works in AnyIO and handle it gracefully.
    - **Timeouts:** Implement timeouts to prevent tasks from running indefinitely.
    - **Signal Handling:** Properly handle signals (e.g., `SIGINT`, `SIGTERM`) to ensure graceful shutdown.
- **Version-Specific Issues:**
    - Consult the AnyIO changelog for known issues and breaking changes in different versions.
- **Compatibility Concerns:**
    - Ensure that AnyIO is compatible with the versions of asyncio or Trio you are using.
    - Be aware of potential compatibility issues with other libraries that use asynchronous programming.
- **Debugging Strategies:**
    - Use logging to trace the execution flow of your code.
    - Employ debuggers (e.g., `pdb`, `ipdb`) to step through your code and inspect variables.
    - Use asynchronous debugging tools to debug asynchronous code.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **Python:** Use the latest stable version of Python (3.10+ recommended).
    - **Poetry/Pip:** Use Poetry or Pip for dependency management.
    - **Pytest:** Use Pytest for testing.
    - **Black/Ruff:** Use Black and Ruff for code formatting and linting.
    - **VS Code/PyCharm:** Use VS Code or PyCharm as your IDE.
- **Build Configuration:**
    - Use a `pyproject.toml` file to configure Poetry or Pip.
    - Specify dependencies and build settings in the `pyproject.toml` file.
    - Use a `setup.py` file for projects that need to be installed with `pip install -e .`
- **Linting and Formatting:**
    - Configure Black and Ruff to enforce consistent code formatting and style.
    - Use a pre-commit hook to automatically format and lint code before committing.
- **Deployment:**
    - Package your application into a Docker container for easy deployment.
    - Use a process manager (e.g., systemd, Supervisor) to manage your application.
    - Deploy your application to a cloud platform (e.g., AWS, Azure, GCP).
- **CI/CD Integration:**
    - Use a CI/CD pipeline to automate the build, test, and deployment process.
    - Integrate your CI/CD pipeline with your version control system (e.g., GitHub, GitLab, Bitbucket).
    - Use automated testing to ensure that code changes do not introduce regressions.