---
description: This rule provides comprehensive guidelines for developing robust, secure, and performant Tauri applications. It covers code organization, security best practices, performance optimization, testing strategies, and common pitfalls to avoid.
globs: *.(rs|js|ts|html|css|json|toml|md)
---
# Tauri Best Practices and Coding Standards

This document outlines best practices and coding standards for developing Tauri applications. Following these guidelines will help you create maintainable, secure, and performant applications.

## 1. Code Organization and Structure

### 1.1 Directory Structure

Adopt a clear and consistent directory structure to improve maintainability and collaboration.


my-tauri-app/
├── .cargo/              # Cargo configuration
├── .git/                # Git repository
├── .gitignore           # Git ignore file
├── .rustfmt.toml       # Rust code formatting configuration
├── src-tauri/           # Rust backend source code
│   ├── src/              # Rust source files
│   │   ├── main.rs        # Main application entry point
│   │   ├── commands.rs    # Custom commands
│   │   ├── models.rs      # Data models
│   │   ├── utils.rs       # Utility functions
│   │   └── state.rs       # Application State management
│   ├── Cargo.toml         # Rust project configuration
│   ├── tauri.conf.json    # Tauri application configuration
│   └── icons/             # Application icons
├── src/                 # Web frontend source code
│   ├── components/        # Reusable UI components
│   │   ├── MyComponent.jsx # Example React component
│   │   └── ...
│   ├── pages/             # Application pages/routes
│   │   ├── Home.jsx       # Home page component
│   │   └── ...
│   ├── App.jsx            # Main application component
│   ├── index.css          # Global styles
│   ├── index.js           # Entry point for web app
│   ├── assets/            # Static assets (images, fonts, etc.)
│   │   ├── logo.png       # Application logo
│   │   └── ...
│   └── utils/             # Frontend utility functions
├── static/              # Static resources (images, fonts, etc.) served directly
├── README.md            # Project documentation
├── LICENSE              # License file
└── package.json         # Node.js project configuration


### 1.2 File Naming Conventions

-   **Rust files:** Use snake_case for file names (e.g., `my_module.rs`).
-   **JavaScript/TypeScript files:** Use PascalCase for component files (e.g., `MyComponent.jsx`) and camelCase for other files (e.g., `utils.js`).
-   **CSS files:** Use kebab-case (e.g., `main-styles.css`).
-   **HTML files:** Use kebab-case (e.g., `index.html`).

### 1.3 Module Organization

-   Group related functionality into modules.
-   Use the `mod` keyword in Rust to define modules.
-   Create separate files for each module.
-   Use `pub` keyword to export items from modules.

rust
// src-tauri/src/commands.rs

pub mod my_command {
  use tauri::{{command, State}};

  #[command]
  pub fn greet(name: String) -> String {
    format!("Hello, {{}}! You've been greeted from Rust!", name)
  }
}



### 1.4 Component Architecture

-   Adopt a component-based architecture for the frontend (e.g., using React, Vue, or Svelte).
-   Break down the UI into reusable components.
-   Follow a consistent component structure (e.g., separate files for component logic, styles, and tests).

### 1.5 Code Splitting Strategies

-   Implement code splitting to reduce the initial bundle size.
-   Use dynamic imports for lazy-loading modules and components.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to Tauri

-   **Command Pattern:** Use Tauri commands to expose Rust functions to the frontend. This allows you to perform complex operations in Rust and communicate the results to the frontend.
-   **State Management:** Use a state management solution (e.g., Redux, Zustand, or React Context) to manage application state and ensure data consistency between the frontend and backend.  Also use Tauri's `State` to manage state on the Rust side.
-   **Event System:**  Use Tauri's event system for inter-process communication (IPC). The backend can emit events that the frontend listens for, enabling real-time updates and notifications.

### 2.2 Recommended Approaches for Common Tasks

-   **File System Access:** Use the Tauri API for file system access instead of directly using Node.js APIs. This ensures that your application respects the Tauri security model.
-   **External API Calls:** Perform external API calls in the Rust backend to protect API keys and sensitive data.
-   **Data Persistence:** Use a database or local storage for persistent data storage. Consider using an ORM like Diesel or SQLx for database access in Rust.

### 2.3 Anti-patterns and Code Smells to Avoid

-   **Over-reliance on JavaScript:** Offload complex logic to the Rust backend whenever possible to leverage Rust's performance and security features.
-   **Ignoring Security Policies:**  Always configure and enforce strict Content Security Policies (CSPs) to prevent XSS attacks.
-   **Exposing Sensitive Data:** Avoid exposing sensitive data (e.g., API keys, database credentials) in the frontend code.
-   **Blocking the Main Thread:**  Avoid long-running tasks on the main thread, both in Rust and JavaScript, to prevent the UI from freezing.

### 2.4 State Management Best Practices

-   Choose a state management solution that fits your application's complexity.
-   Use immutable data structures to prevent accidental state mutations.
-   Centralize state management logic to improve maintainability.
-   Use Tauri's State management features to manage complex application states on the Rust side.

### 2.5 Error Handling Patterns

-   Use Rust's `Result` type for error handling in the backend.
-   Provide informative error messages to the frontend.
-   Implement global error handling to catch unhandled exceptions.
-   Use appropriate logging to track errors and debug issues.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Profiling:** Use profiling tools (e.g., Chrome DevTools, Rust's `perf` tool) to identify performance bottlenecks.
-   **Code Optimization:** Optimize Rust code for performance by using efficient algorithms, data structures, and memory management techniques.
-   **Webview Optimization:** Optimize frontend code for webview performance by reducing DOM manipulations, minimizing JavaScript execution time, and optimizing CSS styles.

### 3.2 Memory Management

-   Use Rust's ownership and borrowing system to prevent memory leaks and data races.
-   Avoid unnecessary allocations and deallocations.
-   Use smart pointers (e.g., `Box`, `Rc`, `Arc`) to manage memory ownership.

### 3.3 Rendering Optimization

-   Use virtual DOM techniques to minimize DOM updates.
-   Optimize CSS selectors to improve rendering performance.
-   Use hardware acceleration for animations and transitions.

### 3.4 Bundle Size Optimization

-   Use code splitting to reduce the initial bundle size.
-   Remove unused code and dependencies.
-   Optimize images and other assets.
-   Use a bundler (e.g., Webpack, Parcel, or Rollup) to optimize the bundle.

### 3.5 Lazy Loading Strategies

-   Use dynamic imports to lazy-load modules and components.
-   Load images and other assets only when they are visible.
-   Use a virtualized list to render large lists of data.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

-   **Cross-Site Scripting (XSS):** Prevent XSS attacks by enforcing strict Content Security Policies (CSPs) and sanitizing user inputs.
-   **Remote Code Execution (RCE):** Prevent RCE attacks by avoiding the use of `eval()` and other unsafe JavaScript functions. Isolate the webview context.
-   **SQL Injection:** Prevent SQL injection attacks by using parameterized queries or an ORM.
-   **Path Traversal:** Prevent path traversal attacks by validating user-provided file paths and using the Tauri API for file system access.
-   **Dependency Vulnerabilities:** Regularly update dependencies and use tools like `cargo audit` and `npm audit` to identify and fix vulnerabilities.

### 4.2 Input Validation

-   Validate all user inputs on both the frontend and backend.
-   Use appropriate data types and formats.
-   Sanitize inputs to remove potentially malicious characters.
-   Implement rate limiting to prevent brute-force attacks.

### 4.3 Authentication and Authorization Patterns

-   Use a secure authentication protocol (e.g., OAuth 2.0, JWT).
-   Store passwords securely using a strong hashing algorithm (e.g., bcrypt).
-   Implement authorization checks to ensure that users only have access to the resources they are authorized to access.

### 4.4 Data Protection Strategies

-   Encrypt sensitive data at rest and in transit.
-   Use HTTPS for all network communication.
-   Store API keys and other secrets securely using environment variables or a dedicated secrets management solution.
-   Avoid storing sensitive data in the frontend code or in local storage.

### 4.5 Secure API Communication

-   Use the Tauri command pattern for secure communication between the frontend and backend.
-   Validate all data exchanged between the frontend and backend.
-   Implement rate limiting to prevent abuse.
-   Use a secure communication protocol (e.g., TLS) for external API calls.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

-   Write unit tests for all critical functions and modules.
-   Use a testing framework (e.g., `cargo test` for Rust, Jest or Mocha for JavaScript) to automate unit testing.
-   Aim for high code coverage.
-   Test edge cases and error conditions.

### 5.2 Integration Testing

-   Write integration tests to verify the interaction between different modules and components.
-   Use a testing framework to automate integration testing.
-   Test the integration between the frontend and backend.

### 5.3 End-to-End Testing

-   Write end-to-end tests to verify the entire application flow.
-   Use a testing framework (e.g., Cypress, Playwright) to automate end-to-end testing.
-   Test the application in a realistic environment.

### 5.4 Test Organization

-   Organize tests into separate directories for unit tests, integration tests, and end-to-end tests.
-   Follow a consistent naming convention for test files and functions.
-   Use test suites to group related tests.

### 5.5 Mocking and Stubbing

-   Use mocking and stubbing to isolate units of code during testing.
-   Use a mocking framework (e.g., Mockall for Rust, Jest or Sinon.js for JavaScript) to create mocks and stubs.
-   Mock external dependencies and APIs.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

-   **Incorrectly Configuring CSP:** Not setting a restrictive enough content security policy.
-   **Misunderstanding Tauri's Security Model:**  Assuming Node.js APIs are safe to use directly in the frontend.
-   **Ignoring Error Handling:**  Not handling errors properly, leading to unexpected application behavior.
-   **Overcomplicating the Frontend:**  Trying to do too much in the frontend, instead of leveraging the Rust backend.

### 6.2 Edge Cases to Be Aware Of

-   **File System Permissions:**  Handling file system permissions correctly on different operating systems.
-   **Webview Compatibility:**  Ensuring compatibility with different webview versions.
-   **Application Update Process:**  Handling application updates gracefully and securely.

### 6.3 Version-Specific Issues

-   Be aware of breaking changes in Tauri releases.
-   Consult the Tauri changelog for information about version-specific issues.

### 6.4 Compatibility Concerns

-   Test your application on different operating systems (Windows, macOS, Linux).
-   Test your application on different architectures (x86, x64, ARM).

### 6.5 Debugging Strategies

-   Use the Tauri developer console for debugging frontend code.
-   Use Rust's debugging tools for debugging backend code.
-   Use logging to track application behavior.
-   Use a debugger to step through code and inspect variables.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

-   **Rust:** Use the latest stable version of Rust.
-   **Node.js:** Use a recent version of Node.js.
-   **IDE:** Use a powerful IDE (e.g., Visual Studio Code, IntelliJ IDEA) with Rust and JavaScript/TypeScript support.
-   **Debugger:** Use a debugger (e.g., GDB, LLDB) for debugging Rust code.
-   **Web Inspector:** Use the web inspector (e.g., Chrome DevTools, Firefox Developer Tools) for debugging frontend code.

### 7.2 Build Configuration

-   Use the `tauri.conf.json` file to configure the build process.
-   Configure the application name, version, and other metadata.
-   Configure the build target and output directory.
-   Configure the security policies and permissions.

### 7.3 Linting and Formatting

-   Use a linter (e.g., Clippy for Rust, ESLint for JavaScript) to enforce code style and identify potential issues.
-   Use a code formatter (e.g., Rustfmt for Rust, Prettier for JavaScript) to automatically format your code.
-   Configure your IDE to automatically run the linter and formatter on save.

### 7.4 Deployment Best Practices

-   Sign your application for the target platform.
-   Use a code signing certificate to verify the authenticity of your application.
-   Package your application for distribution using the appropriate tools (e.g., NSIS for Windows, DMG for macOS, DEB/RPM for Linux).

### 7.5 CI/CD Integration

-   Use a CI/CD system (e.g., GitHub Actions, GitLab CI, Jenkins) to automate the build, test, and deployment process.
-   Configure your CI/CD system to run the linter, formatter, and tests.
-   Configure your CI/CD system to automatically build and package your application for different platforms.
-   Automate the deployment process to reduce manual effort and errors.

By adhering to these best practices, you can develop Tauri applications that are secure, performant, and maintainable. Remember to stay updated with the latest Tauri documentation and community recommendations for continuous improvement.