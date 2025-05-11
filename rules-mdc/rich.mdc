---
description: Comprehensive best practices and coding standards for the Rich library, focusing on code quality, performance, and maintainability within Python terminal applications.
globs: *.py
---
- **General Guidelines**
  - Adhere to PEP 8 coding style guidelines for Python code, emphasizing readability and consistency.
  - Limit lines to a maximum of 79 characters to enhance readability across different environments.
  - Prioritize code clarity and simplicity, making it easy to understand and maintain.
  - Use UTF-8 encoding for source files to ensure compatibility with a wide range of characters.

- **Installation and Environment**
  - Use `uv` for installing dependencies to improve speed and reliability, if appropriate.
  - Specify Python 3.12 or later to leverage the latest language features and performance improvements.  (If applicable for the specific features of rich you are using)

- **Code Organization and Structure**

  - **Directory Structure:** Follow a logical directory structure.
    
    project_root/
    ├── src/
    │   ├── main.py        # Entry point of the application
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── helper_functions.py
    │   ├── modules/
    │   │   ├── __init__.py
    │   │   ├── module_a.py  # Rich-related components
    │   │   ├── module_b.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_main.py
    │   ├── test_module_a.py
    ├── README.md
    ├── pyproject.toml  # Or requirements.txt
    
  - **File Naming:** Use descriptive lowercase names with underscores (e.g., `console_output.py`).
  - **Module Organization:** Group related functionalities into separate modules (e.g., `rich_display.py`, `data_formatting.py`).
  - **Component Architecture:** Design modular components with clear interfaces for reusability and maintainability.
  - **Code Splitting:**  Break down large files into smaller, more manageable pieces based on functionality. Consider lazy loading of less-frequently used modules.

- **Coding Style and Best Practices**

  - **Indentation:** Use 4 spaces for indentation.
  - **Blank Lines:** Separate top-level functions and classes with two blank lines, and methods within a class with one blank line.
  - **Imports:**
    - Group imports in the following order:
      1. Standard library imports
      2. Third-party library imports (including Rich)
      3. Local application imports
    - Use absolute imports for clarity, unless relative imports significantly improve readability within a package.
    - Avoid wildcard imports (`from module import *`).
  - **Naming Conventions:**
    - Use lowercase with underscores for function and variable names (`my_variable`, `my_function`).
    - Use CapWords for class names (`MyClass`).
    - Use UPPER_CASE_WITH_UNDERSCORES for constants (`MAX_VALUE`).
  - **String Quotes:** Use double quotes consistently for strings, especially for docstrings.

- **Rich Library Specific Best Practices**

  - **Console Instantiation:** Create a single `Console` instance for your application and reuse it throughout.
    python
    from rich.console import Console

    console = Console()

    def my_function():
        console.print("Hello, [bold red]World![/bold red]")
    
  - **Styling:** Use Rich's markup system for styling text.  Refer to the Rich documentation for available styles and their usage.
    python
    console.print("[link=https://example.com]Click here[/link] to visit example.com")
    
  - **Tables:** Utilize the `Table` class for structured data display.  Configure columns and rows appropriately.
    python
    from rich.table import Table

    table = Table(title="My Data")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Age", style="magenta")
    table.add_row("Alice", "30")
    table.add_row("Bob", "25")
    console.print(table)
    
  - **Progress Bars:** Employ the `Progress` class for tracking long-running tasks.  Configure the progress bar to accurately reflect the task's progress.
    python
    from rich.progress import Progress
    import time

    with Progress() as progress:
        task1 = progress.add_task("[red]Downloading...", total=1000)
        task2 = progress.add_task("[green]Processing...", total=100)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.1)
            time.sleep(0.01) #Simulate some work
    
  - **Inspect:** Use `console.inspect()` for debugging and understanding objects.  This is invaluable for exploring Rich's own classes and objects, or any object in your application.
    python
    from rich.panel import Panel

    panel = Panel("Hello, World!")
    console.inspect(panel, methods=True)
    

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**  Employ appropriate design patterns such as Factory Pattern for creating Rich objects or Strategy Pattern for different output styles.
  - **Recommended Approaches:**  Use Rich's features for displaying data structures (lists, dictionaries) in a human-readable format.
  - **Anti-patterns:**  Avoid directly printing to the console using `print()` when you should be using `console.print()` to take advantage of Rich's features. Avoid excessive nesting of Rich markup, which can reduce readability.
  - **State Management:** Manage the state of Rich components appropriately, especially when creating dynamic displays.  Avoid mutating objects directly without updating the console output.
  - **Error Handling:** Handle exceptions gracefully and use Rich's features to display error messages to the user in a clear and informative way.
    python
    try:
        result = 1 / 0
    except Exception as e:
        console.print_exception(show_locals=True)
    

- **Performance Considerations**
  - **Optimization Techniques:** Use Rich's built-in caching mechanisms to avoid re-rendering the same content repeatedly.  Minimize the use of computationally expensive Rich features when performance is critical.
  - **Memory Management:** Be mindful of memory usage when displaying large amounts of data with Rich. Consider using generators or iterators to process data in chunks.
  - **Rendering Optimization:** If applicable, profile your Rich-based application to identify rendering bottlenecks. Optimize the rendering of complex Rich elements by simplifying the markup or reducing the number of elements.
  - **Bundle Size Optimization:** Not directly applicable since rich is primarily server-side for terminal apps but for web integrated terminals, ensure only necessary Rich dependencies are bundled.
  - **Lazy Loading:** Not directly applicable, but relevant for other parts of your application that might interact with Rich.

- **Security Best Practices**
  - **Vulnerabilities:** Be aware of potential vulnerabilities related to untrusted input. Sanitize any user-provided text before displaying it with Rich to prevent markup injection attacks.
  - **Input Validation:** Validate any input before using it in Rich's markup to prevent unexpected behavior or security issues.
  - **Authentication and Authorization:** Not directly applicable to Rich, but ensure proper authentication and authorization mechanisms are in place for any data displayed by Rich.
  - **Data Protection:** Protect sensitive data by masking or redacting it before displaying it with Rich.  Utilize Rich's styling options to emphasize sensitive data that requires special attention.

- **Testing Approaches**
  - **Unit Testing:** Write unit tests for individual Rich components to ensure they function correctly.  Mock the `Console` object to isolate the component under test.
  - **Integration Testing:** Test the integration of Rich components with other parts of your application to ensure they work together seamlessly.
  - **End-to-end Testing:** Verify the overall behavior of your Rich-based application by simulating user interactions and validating the output displayed on the console.
  - **Test Organization:** Organize your tests into logical modules that correspond to the structure of your application.
  - **Mocking and Stubbing:** Use mocking and stubbing techniques to isolate components and simulate dependencies during testing.  The `unittest.mock` module provides tools for creating mock objects.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:** Forgetting to import the `Console` class or using `print()` instead of `console.print()`.  Overcomplicating Rich markup, leading to unreadable code.
  - **Edge Cases:** Handling Unicode characters correctly.  Dealing with terminals that have limited color support. Handling very large data sets, which can cause memory issues or performance problems.
  - **Version-Specific Issues:** Being aware of breaking changes or new features in different versions of Rich. Consult the Rich changelog for details.
  - **Compatibility Concerns:** Ensuring compatibility between Rich and other terminal libraries or frameworks you are using.
  - **Debugging:** Using `console.inspect()` to explore objects and identify issues.  Setting breakpoints and stepping through the code to understand the flow of execution.

- **Tooling and Environment**
  - **Recommended Tools:** VS Code with the Python extension, PyCharm, or any other IDE that supports Python development. Consider using a Rich-specific plugin if available.
  - **Build Configuration:** Use `pyproject.toml` (preferred) or `requirements.txt` to manage project dependencies, including Rich.
  - **Linting and Formatting:** Use Pylint, Flake8, and Black to enforce code style guidelines and catch potential errors.
  - **Deployment:** Ensure the target environment has Python and Rich installed. Consider using a virtual environment to isolate dependencies.
  - **CI/CD Integration:** Integrate Rich-based applications into your CI/CD pipeline to automate testing and deployment.  Use tools like Jenkins, GitLab CI, or GitHub Actions.

- **Additional Notes**
  - Consult the official Rich documentation (https://rich.readthedocs.io) for the most up-to-date information and examples.
  - Explore Rich's examples and demonstrations to learn more about its capabilities.
  - Contribute to the Rich community by reporting bugs, suggesting features, or submitting pull requests.

- **References**
  - PEP 8: Style Guide for Python Code (https://peps.python.org/pep-0008/)
  - Rich Documentation: (https://rich.readthedocs.io)