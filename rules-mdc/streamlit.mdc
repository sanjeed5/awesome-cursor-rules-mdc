---
description: This rule provides guidelines and best practices for developing maintainable, performant, and secure Streamlit applications. It covers code organization, performance optimization, security considerations, testing strategies, and common pitfalls to avoid.
globs: *.py
---
- **Code Organization and Structure**:
  - **Directory Structure**: Structure your Streamlit application using a modular directory structure to improve maintainability and collaboration. A suggested layout includes folders for pages, services, models, components, and utilities.
    
    my_streamlit_app/
    ├── app.py                # Main entry point for the Streamlit app
    ├── pages/              # Directory for individual app pages
    │   ├── dashboard.py    # Dashboard page
    │   ├── reports.py      # Reports page
    │   └── settings.py     # Settings page
    ├── services/           # Directory for business logic and data operations
    │   ├── data_loader.py  # Handles data loading
    │   ├── analysis.py     # Contains data analysis functions
    │   └── utils.py        # Utility functions
    ├── models/             # Directory for data models and schemas
    │   └── data_model.py   # Defines data structures
    ├── components/         # Directory for reusable UI components
    │   ├── button.py       # Custom button component
    │   └── chart.py        # Custom chart component
    ├── utils/              # Miscellaneous utility functions
    │   └── helpers.py      # Helper functions
    ├── requirements.txt      # List of Python dependencies
    ├── .gitignore            # Specifies intentionally untracked files that Git should ignore
    └── .venv/                # Python virtual environment (optional)
    
  - **File Naming Conventions**: Use descriptive and consistent file names for your Streamlit components, services, and pages.  For example, `data_loader.py`, `dashboard.py`, and `button.py`.
  - **Module Organization**: Organize your Streamlit application into logical modules to promote code reuse and reduce complexity.  Create separate modules for data loading, data processing, UI components, and utility functions.  Use relative imports to maintain a clear module structure.
  - **Component Architecture**: Build reusable UI components using Streamlit's `st.components.v1` API or custom functions.  Encapsulate component logic and styling to promote consistency and maintainability.  Use props or parameters to customize component behavior and appearance.
  - **Code Splitting**: Split large Streamlit applications into multiple pages using the `st.page_config` and `st.switch_page` functions. This enhances navigation and reduces initial loading times. Consider lazy loading strategies using the `secrets` feature to control which parts of the application are loaded when. 

- **Common Patterns and Anti-patterns**:
  - **Design Patterns**: Employ the Observer pattern for reactive UIs, Facade pattern for simplifying complex interactions, and Strategy pattern to manage different data sources or algorithms.
  - **Recommended Approaches**: Utilize Streamlit's `st.form` for handling user input and submission. Use `st.session_state` to manage application state across reruns. Leverage Streamlit's caching mechanisms (`@st.cache_data`, `@st.cache_resource`) to optimize performance.
  - **Anti-patterns**: Avoid performing expensive computations or data loading operations directly within the main Streamlit application loop.  Do not store large datasets in `st.session_state`.  Avoid using global variables, as they can lead to unexpected behavior.
  - **State Management**:  Use `st.session_state` for storing and managing application state across reruns.  Initialize session state variables using a conditional statement (`if 'key' not in st.session_state:`) to prevent overwriting values on subsequent reruns. Consider implementing a state management class or module to encapsulate and manage complex state logic.
  - **Error Handling**:  Implement comprehensive error handling throughout your Streamlit application.  Use `try...except` blocks to catch exceptions and display informative error messages to the user using `st.error` or `st.exception`.  Log errors to a file or monitoring system for debugging and analysis.

- **Performance Considerations**:
  - **Optimization Techniques**: Use caching (`@st.cache_data`, `@st.cache_resource`) to store the results of expensive computations or data loading operations. Optimize data loading by using appropriate data formats (e.g., Parquet, Feather) and filtering data at the source. Debounce user interactions to reduce the frequency of reruns.
  - **Memory Management**:  Avoid loading large datasets into memory unnecessarily.  Use data streaming techniques or chunking to process large datasets in smaller portions.  Delete unused variables and data structures to release memory.
  - **Rendering Optimization**:  Use Streamlit's built-in rendering optimizations, such as delta generation, to minimize the amount of data that needs to be sent to the browser.  Avoid creating complex layouts with deeply nested components, as this can impact rendering performance.
  - **Bundle Size**: Minimize external dependencies to reduce the bundle size and improve loading times. Use a `.streamlitignore` file to exclude unnecessary files and directories from the Streamlit deployment bundle.  Consider using a CDN to serve static assets.
  - **Lazy Loading**: Implement lazy loading for expensive components or sections of your Streamlit application.  Load components only when they are needed, using conditional rendering or callbacks. Utilize the `secrets` functionality to control the conditional loading of certain modules if access to a given secret is available. 

- **Security Best Practices**:
  - **Common Vulnerabilities**: Be aware of common vulnerabilities, such as cross-site scripting (XSS), SQL injection, and command injection.  Prevent these vulnerabilities by implementing proper input validation and sanitization, and by avoiding the execution of untrusted code.
  - **Input Validation**: Validate all user inputs to prevent malicious code from being injected into your Streamlit application.  Use regular expressions or custom validation functions to ensure that inputs conform to expected formats and ranges. Sanitize user inputs by encoding or escaping special characters.
  - **Authentication and Authorization**: Implement authentication and authorization mechanisms to control access to sensitive data and functionality.  Use Streamlit's `secrets` API to store API keys, passwords, and other sensitive information securely.  Consider using a third-party authentication provider, such as Auth0 or Google Identity Platform.
  - **Data Protection**: Protect sensitive data by encrypting it at rest and in transit.  Use HTTPS to secure communication between the browser and the Streamlit application.  Implement data masking or redaction to prevent sensitive data from being displayed to unauthorized users.
  - **Secure API Communication**: Always use HTTPS for API communications. Validate the origin and authenticity of data from external APIs. Protect API keys using Streamlit secrets and environment variables.

- **Testing Approaches**:
  - **Unit Testing**: Unit test individual Streamlit components, services, and utility functions using the `pytest` framework.  Mock external dependencies to isolate the code being tested.  Write test cases that cover a range of inputs and edge cases.
  - **Integration Testing**: Integrate test Streamlit applications to verify that components and modules work together correctly.  Use Streamlit's testing utilities to simulate user interactions and verify the output.
  - **End-to-end Testing**: Perform end-to-end tests to verify that the entire Streamlit application works as expected.  Use Selenium or Cypress to automate browser interactions and verify the UI and functionality.
  - **Test Organization**: Organize your tests into a separate `tests` directory within your Streamlit project.  Use descriptive test names and docstrings to explain the purpose of each test.  Follow a consistent naming convention for test files and functions.
  - **Mocking and Stubbing**: Use mocking and stubbing techniques to isolate the code being tested and to simulate external dependencies.  Use the `unittest.mock` module or a third-party mocking library to create mock objects and stubs.

- **Common Pitfalls and Gotchas**:
  - **Frequent Mistakes**: Forgetting to use `@st.cache_data` or `@st.cache_resource` for expensive operations, not managing `st.session_state` properly, and neglecting error handling are common mistakes.
  - **Edge Cases**: Be aware of edge cases, such as handling missing data, dealing with invalid inputs, and managing concurrent user sessions.
  - **Version-Specific Issues**: Check the Streamlit documentation and release notes for version-specific issues and compatibility concerns. Be aware of breaking changes when upgrading to a newer version of Streamlit.
  - **Compatibility Concerns**: Ensure that your Streamlit application is compatible with the browsers and operating systems that your users will be using.  Test your application on different devices and browsers to identify and resolve compatibility issues.
  - **Debugging Strategies**: Use Streamlit's debugging tools, such as the debugger and the console, to identify and resolve issues.  Print debug messages to the console to track the execution flow and to inspect variables. Use `st.experimental_rerun` to programmatically trigger reruns during debugging.

- **Tooling and Environment**:
  - **Recommended Tools**: Use VS Code with the Python extension for development. Use `pip` or `uv` for dependency management, and Docker for containerization.
  - **Build Configuration**: Create a `requirements.txt` file to list your Streamlit application's dependencies.  Use a virtual environment to isolate your project's dependencies from the system-level Python installation.  Consider using a build tool, such as Make or Poetry, to automate the build process.
  - **Linting and Formatting**: Use linters and formatters, such as `flake8` and `black`, to enforce code style guidelines and to identify potential errors. Configure your editor to automatically run linters and formatters on save.
  - **Deployment**: Deploy your Streamlit application to Streamlit Cloud, Heroku, AWS, or another cloud platform.  Use Docker to containerize your application and to ensure consistent deployment across different environments.
  - **CI/CD Integration**: Integrate your Streamlit project with a CI/CD pipeline to automate the testing, building, and deployment processes.  Use GitHub Actions, GitLab CI, or another CI/CD platform to configure your pipeline.