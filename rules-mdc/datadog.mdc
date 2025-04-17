---
description: This rule outlines best practices for coding standards, observability, and effective use of the Datadog library in Python projects. It covers coding style, metric/tag naming, dashboard design, security, and performance optimization.
globs: *.py
---
- **Coding Standards**
  - Use `black` for code formatting to ensure consistent style.
  - Use `isort` to sort imports lexicographically.
  - Employ `flake8` for style checks and to maintain code quality.
  - Utilize `bugbear` plugin for `flake8` to identify potential bugs and design issues. Enable the following checks:
    - `B001`: Avoid bare `except:` clauses; prefer `except Exception:`. Catches unexpected events.
    - `B003`: Avoid direct assignment to `os.environ`. Use `os.environ.clear()` or `env=` argument to `Popen`.
    - `B006`: Do not use mutable data structures for argument defaults.
    - `B007`: Loop control variable should be used within the loop body.
    - `B301`: Use Python 3 `.iter*` methods on dictionaries.
    - `B305`: Use `next()` builtin instead of `.next()`.
    - `B306`: Use `str(e)` or `e.args` to access exception messages.
    - `B902`: Use `self` for instance methods, and `cls` for class methods.
  - Enforce consistent logging format with `logging-format` `flake8` plugin. Enable the following checks:
    - `G001`: Logging statements should not use `string.format()`.
    - `G002`: Logging statements should not use `%` formatting.
    - `G003`: Logging statements should not use `+` concatenation.
    - `G004`: Logging statements should not use f-strings (Python 3.6+).
    - `G010`: Logging statements should not use `warn`; use `warning` instead.
    - `G100`: Logging statements should not use extra arguments unless whitelisted.
    - `G201`: Logging statements should not use `error(..., exc_info=True)`; use `exception(...)` instead.
    - `G202`: Logging statements should not use redundant `exc_info=True` in `exception`.
  - Use type checking with `mypy` for improved code quality. Configure `mypy` in `hatch.toml`:
    toml
    [env.collectors.datadog-checks]
    check-types: true
    mypy-args = [
    "--py2",
    "--install-types",
    "--non-interactive",
    "datadog_checks/",
    "tests/",
    ]
mypy-deps = [
    "types-mock==0.1.5",
    ]
    
  - Consider using flags like `--check-untyped-defs` and `--disallow-untyped-defs` for stricter type checking. Configure the files to be checked with mypy.ini file.

- **Metric and Tag Naming Conventions**
  - Use descriptive and meaningful names for metrics and tags.
  - Avoid abbreviations that might have multiple meanings.
  - Maintain consistency in naming across all teams, apps, and services.
  - Avoid reserved keywords.
  - Prefix metrics with a namespace depicting the application or service generating the data (e.g., `http.nginx.response_time`).
  - Metric names must start with a letter.
  - Metric names can only contain ASCII alphanumerics, underscores, and periods. Other characters are converted to underscores.
  - Metric names should not exceed 200 characters (preferably less than 100).
  - Tag names must start with a letter.
  - Tag names may contain alphanumerics, underscores, minuses, colons, periods, and slashes. Other characters are converted to underscores.
  - Tags can be up to 200 characters long (including both key and value) and support Unicode, but are converted to lowercase.
  - Use the `key:value` syntax for tags for optimal functionality. Commonly used metric tag keys are `instance`, `name`, and `role`.
  - Implement unified service tagging using `env`, `service`, and `version` tags.

- **Effective Dashboard Design**
  - Design dashboards carefully for effective visualization and correlation.
  - Utilize out-of-the-box dashboards provided by Datadog.
  - Contribute to community-curated dashboards to enhance visibility and correlation of observability data.
  - Follow integration dashboard best practices when contributing.
  - Consult the "Effective Dashboards" repository for dashboard design guidelines.

- **Code Organization and Structure**
  - **Directory Structure:**
    - Organize code into modules based on functionality (e.g., `metrics`, `logs`, `tracing`).
    - Use a `common` directory for shared utilities and helper functions.
  - **File Naming Conventions:**
    - Use descriptive file names that reflect the module's purpose (e.g., `metrics_collector.py`, `log_processor.py`).
  - **Module Organization:**
    - Keep modules small and focused on a single responsibility.
    - Use `__init__.py` files to define package structure and expose necessary functionality.
  - **Component Architecture:**
    - Design components with clear interfaces and separation of concerns.
    - Use dependency injection to manage dependencies between components.
  - **Code Splitting:**
    - Consider splitting large modules into smaller files to improve maintainability.
    - Use lazy loading for infrequently used modules.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - Use the Observer pattern for subscribing to Datadog events.
    - Implement the Decorator pattern for adding custom functionality to Datadog integrations.
  - **Recommended Approaches:**
    - Use Datadog's API client libraries for interacting with the Datadog API.
    - Implement custom checks using Datadog's check framework.
  - **Anti-patterns:**
    - Avoid hardcoding API keys or other sensitive information in the code.
    - Avoid excessive logging, which can impact performance.
  - **State Management:**
    - Use appropriate data structures for storing state (e.g., dictionaries, lists).
    - Consider using a dedicated state management library for complex state management needs.
  - **Error Handling:**
    - Implement robust error handling to prevent application crashes.
    - Use try-except blocks to catch and handle exceptions.
    - Log errors to Datadog for monitoring and analysis.

- **Performance Considerations**
  - **Optimization Techniques:**
    - Use efficient data structures and algorithms.
    - Minimize network requests to the Datadog API.
    - Use caching to reduce database load.
  - **Memory Management:**
    - Avoid memory leaks by properly releasing resources.
    - Use garbage collection to clean up unused objects.
  - **Bundle Size Optimization:**
    - Remove unused dependencies from the project.
    - Use code minification to reduce bundle size.
  - **Lazy Loading:**
    - Load infrequently used modules or components on demand.

- **Security Best Practices**
  - **Vulnerabilities:**
    - Prevent injection attacks by validating user inputs.
    - Protect against cross-site scripting (XSS) attacks by encoding outputs.
  - **Input Validation:**
    - Validate all user inputs to prevent malicious data from entering the system.
    - Use regular expressions or other validation techniques to enforce input constraints.
  - **Authentication and Authorization:**
    - Use strong authentication mechanisms to verify user identities.
    - Implement authorization policies to control access to resources.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure communication protocols (e.g., HTTPS) for API communication.
  - **Secure API Communication:**
    - Use TLS/SSL for secure communication with the Datadog API.
    - Store API keys securely and do not expose them in the code.

- **Testing Approaches**
  - **Unit Testing:**
    - Write unit tests for individual components to verify their functionality.
    - Use mocking to isolate components from external dependencies.
  - **Integration Testing:**
    - Write integration tests to verify interactions between components.
    - Use a testing framework like pytest for writing and running tests.
  - **End-to-end Testing:**
    - Write end-to-end tests to verify the entire application workflow.
    - Use a tool like Selenium or Cypress for automating browser tests.
  - **Test Organization:**
    - Organize tests into separate directories based on component or functionality.
    - Use descriptive test names to clearly indicate the purpose of each test.
  - **Mocking and Stubbing:**
    - Use mocking to simulate external dependencies during testing.
    - Use stubbing to replace complex components with simplified versions.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - Incorrectly configuring Datadog integrations.
    - Not properly handling errors or exceptions.
    - Overusing logging, which can impact performance.
  - **Edge Cases:**
    - Handling large volumes of data.
    - Dealing with intermittent network connectivity issues.
  - **Version-specific Issues:**
    - Be aware of compatibility issues between different versions of Datadog and its dependencies.
  - **Compatibility Concerns:**
    - Ensure compatibility between Datadog and other technologies used in the project.
  - **Debugging Strategies:**
    - Use Datadog's monitoring tools to track down performance bottlenecks and errors.
    - Use logging to trace the execution flow of the application.

- **Tooling and Environment**
  - **Recommended Tools:**
    - Use an IDE like VS Code or PyCharm for development.
    - Use a version control system like Git for managing code.
    - Use a build tool like Make or Poetry for building and deploying the application.
  - **Build Configuration:**
    - Use a build tool to automate the build process.
    - Configure the build tool to generate optimized bundles.
  - **Linting and Formatting:**
    - Use a linter like Flake8 to enforce coding standards.
    - Use a formatter like Black to automatically format code.
  - **Deployment:**
    - Use a deployment tool like Docker or Kubernetes for deploying the application.
    - Use a CI/CD pipeline to automate the deployment process.
  - **CI/CD Integration:**
    - Integrate Datadog into the CI/CD pipeline to automatically monitor and analyze application performance.
    - Use Datadog's API to create custom dashboards and alerts.