---
description: This rule provides best practices for developing efficient, maintainable, and scalable GUI applications with CustomTkinter. It covers code organization, performance, security, testing, and common pitfalls.
globs: *.py
---
- **Always use UV when installing dependencies.**
- **Always use Python 3.10 or higher.** CustomTkinter requires a modern Python version for compatibility and feature support.
- **Prefer an object-oriented approach.** Encapsulate UI elements and logic within classes for better organization, maintainability, and scalability. Avoid procedural coding for complex applications.
- **Use virtual environments.**  Create isolated environments for each project to manage dependencies and avoid conflicts.
- **Use descriptive variable and function names.**  Clear and concise names improve code readability.

## 1. Code Organization and Structure

- **Directory Structure:**
    - Project Root
        - `src/` (Source code)
            - `gui/` (CustomTkinter-related modules)
                - `__init__.py`
                - `main_window.py` (Main application window)
                - `components/` (Reusable UI components)
                    - `__init__.py`
                    - `button.py`
                    - `label.py`
                - `models/` (Data models if applicable)
                    - `__init__.py`
                    - `user.py`
            - `utils/` (Utility functions and helpers)
                - `__init__.py`
                - `config.py`
                - `helpers.py`
            - `services/` (API interaction services)
                - `__init__.py`
                - `api_service.py`
        - `tests/` (Tests)
            - `__init__.py`
            - `gui/`
                - `test_main_window.py`
            - `utils/`
                - `test_helpers.py`
        - `data/` (Data files, configuration files)
        - `requirements.txt` (Dependencies)
        - `README.md` (Project documentation)
        - `.gitignore`
        - `LICENSE`
        - `pyproject.toml` or `setup.py`

- **File Naming Conventions:**
    - Python files: `lowercase_with_underscores.py`
    - Class names: `PascalCase` (e.g., `MainWindow`, `CustomButton`)
    - Variable names: `lowercase_with_underscores` (e.g., `main_window`, `button_text`)
    - Constants: `UPPERCASE_WITH_UNDERSCORES` (e.g., `DEFAULT_FONT`, `API_KEY`)

- **Module Organization:**
    - Group related functionalities into separate modules.
    - Use `__init__.py` files to define packages and control namespace imports.
    - Keep modules focused on a single responsibility.

- **Component Architecture:**
    - Divide the UI into reusable components (e.g., buttons, labels, forms).
    - Create custom widget classes by inheriting from `customtkinter.CTkBaseClass` or other appropriate customtkinter base classes.
    - Utilize the Model-View-Controller (MVC) or Model-View-ViewModel (MVVM) design pattern to separate UI logic from data and presentation.

- **Code Splitting:**
    - Break down large UI components into smaller, manageable parts.
    - Use functions and methods to encapsulate specific actions.
    - Consider creating separate modules for complex features.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **MVC (Model-View-Controller):** Separate data (Model), UI (View), and user input handling (Controller).
    - **MVVM (Model-View-ViewModel):** Similar to MVC, but uses a ViewModel to mediate between the Model and View.
    - **Observer:** Implement event handling and notifications between components.
    - **Factory:** Create UI elements dynamically based on configuration or user input.
    - **Singleton:** Ensures a class only has one instance.

- **Recommended Approaches:**
    - **Dynamic UI updates:** Use `configure()` to modify widget properties instead of recreating widgets.
    - **Data binding:**  Link UI elements to data models for automatic updates.
    - **Theming:** Create reusable themes with consistent styles for your application.
    - **Asynchronous Operations:** Use `after()` or `threading` module  to handle long-running tasks without blocking the main event loop.
    - **Use `CTkFont`**: Use this to ensure that the correct font family is used, that is appropriate across all operating systems. 

- **Anti-patterns and Code Smells:**
    - **God object:** Avoid creating a single class that handles too many responsibilities.
    - **Global variables:** Minimize the use of global variables; use instance variables or dependency injection instead.
    - **Deeply nested code:** Refactor complex conditional statements and loops into smaller, more readable functions.
    - **Mixing grid() and pack() layout managers**: Stick to one layout manager per container to avoid unexpected behavior.

- **State Management:**
    - **Centralized state:** Use a dedicated state management class or library to store and manage application state (e.g., using the observer pattern). This can be a simple class for smaller apps, or something like RxPY for more complex state handling.
    - **Immutable data:**  Treat state data as immutable and create new copies when modifying it to prevent unexpected side effects.
    - **Observable state:** Use observable patterns to notify components when the state changes.

- **Error Handling:**
    - **Try-except blocks:** Wrap potentially failing code with `try-except` blocks to handle exceptions gracefully.
    - **Logging:** Use the `logging` module to record errors and debug information.
    - **User feedback:**  Provide informative error messages to the user.
    - **Specific exception handling:** Handle exceptions in specific branches of your application, rather than catching all errors in a single block.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Minimize UI redraws:** Use `update_idletasks()` to refresh the UI without a full redraw, especially when making multiple changes to widgets.  Only update relevant portions of the UI.
    - **Efficient layout management:** Choose the appropriate layout manager (grid or pack) and use it consistently.
    - **Widget caching:** Reuse existing widgets instead of creating new ones whenever possible.
    - **Batch updates:** Group multiple UI updates into a single operation to reduce the number of redraws.
    - **Efficient image handling**: Use optimized image formats (e.g., WebP) and resize images appropriately for the UI.

- **Memory Management:**
    - **Avoid memory leaks:** Be careful with circular references and ensure that objects are properly released when they are no longer needed.
    - **Resource cleanup:** Close files, release database connections, and other resources when they are no longer needed.

- **Rendering Optimization:**
    - **Reduce widget complexity:** Avoid using overly complex widgets or layouts.
    - **Use hardware acceleration:**  Enable hardware acceleration if available to improve rendering performance. This is typically handled by Tkinter itself, but ensure your system supports it.

- **Lazy Loading:**
    - **Load UI elements on demand:** Load UI elements only when they are needed to reduce the initial load time.
    - **Virtual scrolling:**  Implement virtual scrolling for large lists or grids to render only the visible items.
    - **Use `after()` wisely:**  Delegate intensive tasks to run in the background using `after()` to prevent freezing.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Code injection:** Prevent code injection by validating and sanitizing user inputs.
    - **Cross-site scripting (XSS):**  Escape user-generated content to prevent XSS attacks.
    - **Data exposure:** Protect sensitive data by encrypting it and storing it securely.
    - **Denial of Service (DoS):** Prevent DoS attacks by limiting the rate of requests and validating input sizes.

- **Input Validation:**
    - **Validate all user inputs:**  Check data types, formats, and ranges to prevent invalid data from entering the system.
    - **Sanitize inputs:** Remove or escape potentially harmful characters from user inputs.
    - **Use parameterized queries:**  Avoid SQL injection by using parameterized queries when interacting with databases.

- **Authentication and Authorization:**
    - **Secure password storage:** Use strong hashing algorithms (e.g., bcrypt, Argon2) to store passwords.
    - **Two-factor authentication (2FA):**  Implement 2FA to enhance account security.
    - **Role-based access control (RBAC):**  Control access to resources based on user roles.
    - **Use HTTPS:** Always use HTTPS for secure communication between the client and server.

- **Data Protection:**
    - **Encryption:** Encrypt sensitive data both in transit and at rest.
    - **Data masking:** Mask or redact sensitive data when displaying it to users.
    - **Regular backups:**  Create regular backups of your data to prevent data loss.

- **Secure API Communication:**
    - **API keys:**  Store API keys securely and protect them from unauthorized access.
    - **Rate limiting:**  Limit the rate of API requests to prevent abuse.
    - **Input validation:**  Validate all data received from APIs.
    - **HTTPS:**  Always use HTTPS when communicating with APIs.

## 5. Testing Approaches

- **Unit Testing:**
    - **Test individual components:**  Write unit tests for each component to ensure that it functions correctly.
    - **Mock dependencies:**  Use mocking to isolate the component being tested from its dependencies.
    - **Test edge cases:**  Test edge cases and boundary conditions to identify potential issues.
    - **Use a testing framework:** Use pytest or unittest.

- **Integration Testing:**
    - **Test interactions between components:**  Write integration tests to ensure that components work together correctly.
    - **Test API interactions:**  Test the interaction between the UI and the backend API.
    - **Use a testing framework:** Use pytest or unittest.

- **End-to-End Testing:**
    - **Test the entire application workflow:**  Write end-to-end tests to simulate user interactions and ensure that the application functions correctly from start to finish.
    - **Use a testing framework:** Use playwright or selenium. 

- **Test Organization:**
    - **Separate test files:** Create separate test files for each module or component.
    - **Follow a consistent naming convention:** Use a consistent naming convention for test files and functions (e.g., `test_<module_name>.py`, `test_<function_name>`).

- **Mocking and Stubbing:**
    - **Use mocking libraries:**  Use libraries like `unittest.mock` or `pytest-mock` to create mock objects for dependencies.
    - **Stub external services:**  Use stubs to replace external services with predictable responses during testing.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - **Forgetting `self`:**  Forgetting to use `self` when accessing instance variables.
    - **Incorrect layout management:**  Mixing `grid` and `pack` in the same container or not specifying proper row/column configurations.
    - **Blocking the main loop:** Performing long-running operations in the main thread, causing the UI to freeze. Use `after()` or `threading` for background tasks.
    - **Incorrect event handling:**  Not properly binding events to UI elements.
    - **Using the same variable name for Tkinter elements**: Tkinter elements should be named descriptively so that they are not confused with other local variables.

- **Edge Cases:**
    - **Handling different screen resolutions:**  Test the UI on different screen resolutions to ensure that it looks correct.
    - **Handling different operating systems:**  Test the UI on different operating systems to ensure that it functions correctly.
    - **Handling different font sizes:**  Test the UI with different font sizes to ensure that the layout remains consistent.

- **Version-Specific Issues:**
    - **Check compatibility:**  Consult the CustomTkinter documentation for version-specific issues and compatibility information.

- **Compatibility Concerns:**
    - **Tkinter versions:** Ensure compatibility between CustomTkinter and the underlying Tkinter version.
    - **Operating system:**  Test the application on different operating systems to identify potential compatibility issues.
    - **Python versions:** Ensure that the application is compatible with the target Python versions.

- **Debugging Strategies:**
    - **Use print statements:**  Insert `print` statements to debug code and inspect variable values.
    - **Use a debugger:** Use a debugger to step through code and inspect the execution flow.
    - **Read error messages:**  Carefully read error messages to understand the root cause of the problem.

## 7. Tooling and Environment

- **Recommended Tools:**
    - **Text editor:** VS Code with Python extension.
    - **Virtual environment manager:** `venv` or `conda`.
    - **Package manager:** `pip` or `poetry`.
    - **Testing framework:** `pytest`.
    - **Linting and formatting:** `pylint` or `flake8` and `black` or `autopep8`.

- **Build Configuration:**
    - **Use `pyproject.toml` or `setup.py`:**  Define project metadata, dependencies, and build instructions in `pyproject.toml` or `setup.py`.
    - **Specify dependencies:**  List all dependencies in `requirements.txt` or `pyproject.toml`.

- **Linting and Formatting:**
    - **Configure linting rules:**  Configure linting tools to enforce code style and identify potential issues.
    - **Use a formatter:** Use a code formatter to automatically format code according to a consistent style guide.
    - **Integrate with CI/CD:** Integrate linting and formatting checks into the CI/CD pipeline.

- **Deployment:**
    - **Package the application:** Package the application into an executable or installer using tools like `pyinstaller` or `cx_Freeze`.
    - **Create a virtual environment:** Create a virtual environment for the application and include all dependencies.
    - **Follow platform-specific guidelines:** Follow platform-specific guidelines for deploying applications.

- **CI/CD Integration:**
    - **Automate testing:**  Automate unit, integration, and end-to-end tests in the CI/CD pipeline.
    - **Automate linting and formatting:**  Automate linting and formatting checks in the CI/CD pipeline.
    - **Automate deployment:**  Automate the deployment process in the CI/CD pipeline.