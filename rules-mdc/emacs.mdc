---
description: Comprehensive guide to Emacs Lisp coding standards, best practices, and common pitfalls. Focuses on maintainability, readability, and performance for building robust Emacs extensions.
globs: *.el
---
- ## Code Organization and Structure
  - ### Directory Structure
    - Organize your Emacs Lisp projects into well-defined directories to improve maintainability and discoverability.
    - `/`: Project root. Contains the main `.el` file (e.g., `my-package.el`) and a README.
    - `/src`: Source code. Contains the core Emacs Lisp files.
    - `/lib`: Libraries.  External libraries or dependencies used by your project.
    - `/test`: Tests.  Unit and integration tests.
    - `/doc`: Documentation.  User documentation and API references. Consider using `org-mode` for easy export to various formats.
    - `/.cursor`: Project rules directory for Cursor AI tool.
  - ### File Naming Conventions
    - Use lowercase letters and hyphens for file names (e.g., `my-package.el`).
    - Each file should generally correspond to a single module or component.
    - Test files should be named following a consistent pattern, such as `my-module-test.el` or `my-module.test.el`.
    - Always end Emacs Lisp files with the `.el` extension.
  - ### Module Organization
    - Encapsulate related functions and variables within modules.
    - Use `provide` to declare a module and `require` to load dependencies.
    - Avoid creating overly large files; break them down into smaller, more manageable modules.
    - Example:
      lisp
      (provide 'my-module)

      (require 'another-module)
      
      (defun my-module-function (arg)
        ;; ...
      )
      
  - ### Component Architecture
    - Design your applications using a component-based architecture to promote code reuse and modularity.
    - Each component should have a well-defined interface and a clear responsibility.
    - Favor composition over inheritance.
    - Example: Separate UI elements from business logic.
  - ### Code Splitting
    - Use `autoload` to defer loading of less frequently used functions, improving startup time.
    - Break up large modules into smaller files and load them on demand.
    - Ensure that your package provides a mechanism for disabling or uninstalling components to avoid conflicts.

- ## Common Patterns and Anti-patterns
  - ### Design Patterns
    - **Command Pattern:** Encapsulate actions as objects, enabling parameterization, queuing, and logging.
    - **Observer Pattern:** Define a one-to-many dependency between objects, so that when one object changes state, all its dependents are notified and updated automatically (e.g., using hooks).
    - **Strategy Pattern:** Define a family of algorithms, encapsulate each one, and make them interchangeable (e.g., customizable behavior through user options).
  - ### Recommended Approaches for Common Tasks
    - **Configuration Management:** Use `defcustom` for user-configurable options.  Provide default values and documentation.
    - **UI Creation:** Use `widget` for creating interactive UI elements.
    - **Asynchronous Operations:** Leverage `timer` and `process` for non-blocking tasks.
    - **Data Storage:** Consider using SQLite via `sqlite.el` for persistent data storage.
  - ### Anti-patterns and Code Smells
    - **Global State Mutation:** Minimize use of global variables.  If necessary, use `defvar` with caution.
    - **Deeply Nested Conditionals:** Refactor complex logic into smaller, more readable functions.
    - **Magic Numbers/Strings:** Define constants for frequently used values.
    - **Duplicated Code:** Extract common logic into reusable functions or macros.
    - **Ignoring Errors:** Always handle errors and exceptions gracefully.
  - ### State Management
    - Prefer lexical scoping with `lexical-binding: t` to create closures and manage state within functions.
    - Use `cl-letf` to temporarily redefine functions for testing or customization.
    - Utilize property lists or hash tables to associate state with buffers or other objects.
  - ### Error Handling
    - Use `condition-case` to handle exceptions and prevent crashes.
    - Provide informative error messages to users.
    - Log errors to a file or the `*Messages*` buffer for debugging.
    - Always clean up resources (e.g., timers, processes) in the `finally` clause of `condition-case`.
    - Example:
      lisp
      (condition-case err
          (progn
            ;; Code that might raise an error
            (do-something))
        (error
         (message "An error occurred: %s" err))
        (finally
         ;; Clean up resources
         (cleanup)))
      

- ## Performance Considerations
  - ### Optimization Techniques
    - **Profiling:** Use `profiler.el` or `elp.el` to identify performance bottlenecks.
    - **Byte Compilation:** Always byte-compile your Emacs Lisp code using `byte-compile-file`.
    - **Caching:** Cache frequently accessed data to reduce computation time.
    - **Lazy Loading:** Use `autoload` for functions that are not immediately needed.
    - **Tail Recursion:** Convert recursive functions to tail-recursive form to avoid stack overflow errors.
    - **StringBuilder:** Utilize `insert` for building large strings instead of repeated concatenation.
  - ### Memory Management
    - Minimize the creation of temporary objects.
    - Reuse existing data structures when possible.
    - Be mindful of garbage collection overhead; avoid creating excessive garbage.
    -  Use weak references where possible, with libraries such as `weak-ref` to avoid memory leaks.
  - ### Rendering Optimization
    - Minimize the number of buffer updates.
    - Use overlays efficiently to avoid performance degradation.
    - Batch updates when possible.
  - ### Bundle Size Optimization
    - Remove unnecessary dependencies.
    - Use code minification tools to reduce the size of your Emacs Lisp files.
    - Compress images and other assets.
  - ### Lazy Loading
    - Use `autoload` for functions and libraries that are not immediately required.
    - Load dependencies on demand rather than at startup.
    - Provide hooks for users to customize which features are loaded.

- ## Security Best Practices
  - ### Common Vulnerabilities
    - **Code Injection:** Prevent execution of untrusted code.
    - **Cross-Site Scripting (XSS):** Sanitize user input to prevent malicious scripts from being injected into UI elements.
    - **Denial-of-Service (DoS):** Limit resource consumption to prevent attacks that overwhelm the system.
    - **Path Traversal:** Validate file paths to prevent access to unauthorized files.
  - ### Input Validation
    - Sanitize all user input to prevent code injection and XSS attacks.
    - Validate file paths to prevent path traversal vulnerabilities.
    - Use regular expressions to enforce input formats.
  - ### Authentication and Authorization
    - Use secure authentication mechanisms such as OAuth or API keys.
    - Implement authorization checks to ensure that users only have access to resources they are authorized to access.
    - Store credentials securely using the Emacs password manager or a dedicated secrets store.
  - ### Data Protection
    - Encrypt sensitive data at rest and in transit.
    - Use secure communication protocols such as HTTPS.
    - Protect against data breaches by implementing appropriate access controls.
  - ### Secure API Communication
    - Use HTTPS for all API communication.
    - Validate API responses to prevent data injection attacks.
    - Implement rate limiting to prevent abuse.

- ## Testing Approaches
  - ### Unit Testing
    - Use ERT or Buttercup to write unit tests for your Emacs Lisp code.
    - Test individual functions and components in isolation.
    - Mock external dependencies to create a predictable testing environment.
  - ### Integration Testing
    - Test the interaction between different components of your Emacs Lisp application.
    - Verify that data flows correctly between components.
    - Ensure that your application integrates properly with Emacs itself.
  - ### End-to-end Testing
    - Use Director or similar tools to automate end-to-end tests.
    - Simulate user interactions and verify that the application behaves as expected.
    - Test the entire application from the user's perspective.
  - ### Test Organization
    - Organize your tests into separate files or directories.
    - Use descriptive names for your test cases.
    - Group related tests together.
    - Provide a clear and concise test report.
  - ### Mocking and Stubbing
    - Use `cl-letf` or similar techniques to mock external dependencies.
    - Create stubs for functions that are difficult or impossible to test directly.
    - Verify that mocked functions are called with the correct arguments.

- ## Common Pitfalls and Gotchas
  - ### Frequent Mistakes
    - Forgetting to byte-compile Emacs Lisp code.
    - Using global variables excessively.
    - Ignoring error conditions.
    - Writing overly complex functions.
    - Failing to document code properly.
    - Not using lexical binding.
  - ### Edge Cases
    - Handling international characters correctly.
    - Dealing with different Emacs versions and platforms.
    - Managing asynchronous operations and timers.
    - Ensuring compatibility with other packages.
    - Testing in various Emacs configurations (e.g., `-Q` for no init file).
  - ### Version-Specific Issues
    - Be aware of API changes between Emacs versions.
    - Use conditional compilation to support multiple versions of Emacs.
    - Test your code on different Emacs versions to ensure compatibility.
  - ### Compatibility Concerns
    - Avoid using deprecated functions and features.
    - Test your code with different Emacs configurations to identify compatibility issues.
    - Provide a mechanism for users to disable or uninstall your package if necessary.
  - ### Debugging Strategies
    - Use `edebug` to step through your code and inspect variables.
    - Print debugging messages to the `*Messages*` buffer.
    - Use `trace` to monitor function calls.
    - Enable `debug-on-error` to catch exceptions.
    - Check the Emacs error log for clues.

- ## Tooling and Environment
  - ### Recommended Development Tools
    - **Emacs:** The best tool for developing Emacs Lisp code.
    - **ERT/Buttercup:** Testing frameworks.
    - **Flycheck:** Real-time syntax checking and linting.
    - **Edebug:** Interactive debugger.
    - **Profiler.el/ELP:** Profiling tools.
    - **Counsel/Ivy/Helm:** Completion frameworks for improved navigation.
    - **Magit:** Git integration.
  - ### Build Configuration
    - Use a `Makefile` or similar build tool to automate common tasks such as byte-compilation, testing, and documentation generation.
    - Define dependencies explicitly in your build configuration.
    - Use version control to track changes to your build configuration.
  - ### Linting and Formatting
    - Use `flycheck` with `package-lint` and `elsa` for static code analysis.
    - Enforce consistent code formatting using `aggressive-indent-mode` or similar tools.
    - Configure your editor to automatically format code on save.
  - ### Deployment
    - Package your Emacs Lisp code into a `.el` file.
    - Distribute your package via MELPA or a similar repository.
    - Provide clear instructions on how to install and configure your package.
  - ### CI/CD Integration
    - Use GitHub Actions, Travis CI, or similar tools to automate testing and deployment.
    - Run tests on every commit to ensure code quality.
    - Automatically deploy your package to MELPA on release.

- ## Best Practices Summary
  - Always use lexical binding: `;; -*- lexical-binding: t; -*-`.
  - Prefix all global symbols with a unique project prefix to avoid namespace collisions.
  - Write clear and concise documentation for all functions and variables using docstrings.
  - Use `defcustom` for user-configurable options.
  - Handle errors gracefully using `condition-case`.
  - Byte-compile your Emacs Lisp code before deploying.
  - Write unit tests for your code using ERT or Buttercup.
  - Follow the Emacs Lisp Style Guide.
  - Leverage the power of Emacs' interactive development environment for real-time feedback.

- @file https://raw.githubusercontent.com/bbatsov/emacs-lisp-style-guide/master/emacs-lisp-style-guide.md
- @file https://raw.githubusercontent.com/emacs-tw/awesome-elisp/master/README.md