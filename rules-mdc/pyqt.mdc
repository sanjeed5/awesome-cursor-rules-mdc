---
description: This rule file provides comprehensive guidelines for PyQt development, covering code organization, common patterns, performance, security, testing, and tooling. It aims to help developers create maintainable, efficient, and secure PyQt applications.
globs: *.py
---
- **Follow Consistent Importing:**
  - Avoid wildcard imports like `from PyQt5.QtGui import *`. They clutter the namespace and can lead to naming conflicts.
  - Instead, import only the necessary modules or classes, e.g., `from PyQt5 import QtGui` and access members with the module prefix: `QtGui.QPushButton`.
  - *Rationale:* Improves code readability and avoids potential namespace collisions.

- **Adhere to Naming Conventions:**
  - Follow Qt's camelCase style for method names (e.g., `loadItems()`, `refreshResults()`). This maintains consistency with the Qt API and reduces confusion.
  - For signals and slots, consistently use camelCase.
  - *Rationale:* Ensures consistency with Qt's API and reduces cognitive load for developers.

- **Separate GUI and Business Logic (MVC or similar):**
  - Keep GUI code separate from business logic using patterns like Model-View-Controller (MVC), Model-View-Presenter (MVP), or Model-View-ViewModel (MVVM).
  - *Model:* Represents the data and business logic.
  - *View:* Displays the data to the user.
  - *Controller/Presenter/ViewModel:* Handles user input and updates the Model and View.
  - *Rationale:* Enhances maintainability, testability, and scalability by decoupling concerns.

- **Leverage Qt Designer (or Alternatives):**
  - Use Qt Designer (or alternative tools like QML) for layout management and rapid prototyping.
  - Load the generated `.ui` files into your Python code using `PyQt6.uic.loadUi`.
  - *Rationale:* Speeds up GUI development and simplifies layout design.

- **Employ Signals and Slots:**
  - Use Qt's signals and slots mechanism for communication between objects.
  - Connect signals to slots using `QObject.connect()` or the newer, more Pythonic syntax in PyQt6.
  - *Rationale:* Provides a type-safe and flexible way to handle events and user interactions.

- **Protect UI Members:**
  - Treat UI components defined in Qt Designer as protected or private members.
  - Provide access to child widgets of your class through methods instead of direct access.
  - *Rationale:* Facilitates future modifications to the UI without breaking external code.

- **Naming Conventions for Designer Components:**
  - Adopt a consistent naming convention for UI components created in Qt Designer.
  - Use prefixes like `ui_` to indicate that a component was created in Designer (e.g., `ui_project_combo`, `uiProjectCombo`). Append the component type.
  - *Rationale:* Improves code readability and makes it easier to identify the origin and purpose of UI components.

- **Getter and Setter Guidelines:**
  - Use getter and setter methods to access and modify internal data instead of directly exposing public members. Use camelCase for both the getter and setter methods.
  - Create protected members using a single leading underscore (e.g., `_value`).
  - *Rationale:* Provides encapsulation, allows for future changes to the internal implementation, and maintains consistency with Qt's style.

- **Never Expose Public Members in Qt Classes:**
  - In Python, there isn't a strict public/protected/private syntax, but use underscores to flag different types of data.
  - Python uses underscores to flag different types of data.
     - `No underscores`: public variable/member
     - `1 opening underscore`: protected variable/member
     - `1 trailing underscore`: used when defining a variable that would otherwise clash with a Python internal
     - `2 opening underscores`: private variable/member (does some funkiness under the hood so it becomes pretty hard to access from outside your class or module)
     - `2 opening underscores and 2 trailing underscores`: built-in, usually used by internal python variables and methods
  -  When developing new Qt objects, widgets, or classes you should create protected members, and allow access to them via getter and setter methods.

- **Code Organization and Structure:**
  - *Directory Structure:* Organize your project into logical directories (e.g., `ui`, `models`, `controllers`, `widgets`).
  - *File Naming:* Use descriptive file names (e.g., `mainwindow.py`, `user_model.py`).
  - *Module Organization:* Break your code into smaller, reusable modules.
  - *Component Architecture:* Design your application using a component-based architecture for better modularity.

- **Common Patterns and Anti-Patterns:**
  - *Design Patterns:* Apply design patterns like Observer, Singleton, Factory, and Strategy where appropriate.
  - *Anti-Patterns:* Avoid God classes, tight coupling, and code duplication.
  - *State Management:* Use signals and slots or dedicated state management libraries for managing application state.
  - *Error Handling:* Implement robust error handling using try-except blocks and logging.

- **Performance Considerations:**
  - *Optimization:* Optimize your code by minimizing expensive operations, using caching, and avoiding unnecessary GUI updates.
  - *Memory Management:* Be mindful of memory usage, especially when dealing with large images or data sets. Use `del` to remove unwanted objects from memory.
  - *Rendering Optimization:* Optimize rendering by using double buffering, avoiding unnecessary repaints, and using optimized drawing methods.
  - Deleting objects that are no longer in use.
  - Limit the number of widgets.

- **Security Best Practices:**
  - *Vulnerabilities:* Be aware of common vulnerabilities like code injection, cross-site scripting (XSS), and SQL injection if using databases.
  - *Input Validation:* Validate all user inputs to prevent malicious data from entering your application.
  - *Authentication:* Implement proper authentication and authorization mechanisms to protect sensitive data.
  - *Data Protection:* Encrypt sensitive data and use secure communication protocols.
  - When fetching external data be mindful about malicious data.

- **Testing Approaches:**
  - *Unit Tests:* Write unit tests for individual components and functions.
  - *Integration Tests:* Test the interaction between different components.
  - *End-to-End Tests:* Test the entire application flow.
  - *Test Organization:* Organize your tests into logical directories and use meaningful names.
  - *Mocking:* Use mocking and stubbing to isolate components during testing.
  - Using `pytest` to generate tests.

- **Common Pitfalls and Gotchas:**
  - *Mistakes:* Avoid circular dependencies, memory leaks, and neglecting event handling.
  - *Edge Cases:* Be aware of edge cases like unexpected user inputs, network errors, and file system issues.
  - *Version Issues:* Pay attention to version-specific issues and compatibility concerns when upgrading PyQt or other dependencies.
  - Ensure the GUI thread is kept responsive.

- **Tooling and Environment:**
  - *Development Tools:* Use IDEs like VS Code with Python extensions, PyCharm, or Qt Creator.
  - *Build Configuration:* Use build tools like CMake or setuptools for managing your project build process.
  - *Linting:* Lint your code using tools like Pylint or Flake8.
  - *Formatting:* Format your code using tools like Black or autopep8.
  - *Deployment:* Use tools like PyInstaller or cx_Freeze for creating standalone executables.
  - *CI/CD:* Integrate your project with CI/CD pipelines for automated testing and deployment.

- **Specific Recommendations:**
  - Always make sure that long running process happen in the background.
  - Utilize `QThread` to move computationally heavy process off the main thread.

- **Code Splitting:**
  - Break large classes and functions into smaller, more manageable units.
  - Extract reusable code into separate modules or libraries.

- **State Management Best Practices:**
   - Use signals and slots effectively to propagate state changes throughout the application.
   - Consider using a dedicated state management library or pattern if your application has complex state requirements.

- **Additional Considerations:**
    - If you need to handle asynchronous operations, leverage `QFuture` and `QThreadPool` for efficient execution.
    - Use `QSettings` to store application settings and preferences.
    - Prioritize accessibility by ensuring your application is usable by people with disabilities (e.g., provide keyboard navigation, use appropriate color contrast).