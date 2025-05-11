---
description: Enforces best practices, coding standards, and performance considerations for Electron development. Covers code structure, security, testing, and common pitfalls to ensure robust and maintainable applications.
globs: *.js,*.jsx,*.ts,*.tsx,*.html,*.css,*.scss,*.mjs,*.cjs
---
- **General Practices**
  - Adhere to the official Electron coding style guide. Run `npm run lint` to detect style issues using `cpplint` and `eslint`.
  - Aim for line lengths between 80 and 100 characters for readability.
  - Use `sh` instead of `cmd` in code blocks for cross-platform compatibility in documentation.
  - Select the appropriate Electron version based on project needs and compatibility with dependencies.
  - Always update electron to the latest stable version to receive security fixes and performance improvements.

- **Code Organization and Structure**
  - **Directory Structure:**
    - Structure your project logically. Consider using a structure like:
      
      electron-app/
      ├── main.js         # Main process entry point
      ├── preload.js      # Preload script (for specific renderers if needed)
      ├── renderer/
      │   ├── index.html    # Main HTML file
      │   ├── script.js     # Renderer process scripts
      │   ├── style.css     # Stylesheets
      │   └── components/ # Reusable components (if applicable)
      ├── assets/         # Static assets (images, fonts, etc.)
      ├── package.json
      └── electron-builder.yml # Electron Builder configuration
      
    - Organize your code into modules based on functionality (e.g., authentication, data handling, UI components).
  - **File Naming Conventions:**
    - Use descriptive and consistent naming conventions.
    - JavaScript/TypeScript files: `camelCase` for variables and functions, `PascalCase` for classes and components.
    - CSS/SCSS files: `kebab-case`.
  - **Module Organization:**
    - Split your application logic into separate modules for better maintainability and reusability.
    - Use ES modules (`import`/`export`) or CommonJS (`require`/`module.exports`) for modularization.  Be consistent throughout the project.
  - **Component Architecture:**
    - Design reusable UI components for the renderer process.
    - Use a component-based framework (e.g., React, Vue, Angular) to manage UI complexity.
  - **Code Splitting:**
    - Implement code splitting to reduce the initial load time of your application.
    - Use dynamic imports (`import()`) to load modules on demand.
    - Consider using webpack or Parcel for bundling and code splitting.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - **Model-View-Controller (MVC):**  Organize application logic into models, views, and controllers to separate concerns.
    - **Observer Pattern:** Implement a publish-subscribe mechanism for communication between different parts of the application.
    - **Singleton Pattern:** Use singletons sparingly. If used, implement carefully to avoid unexpected side effects.
  - **Recommended Approaches:**
    - **Inter-Process Communication (IPC):** Use `ipcRenderer` and `ipcMain` for communication between the main and renderer processes.  Sanitize all data passed via IPC.
    - **Native Modules:** Consider using native modules for performance-critical tasks or access to platform-specific APIs.
    - **Remote Module:** Avoid using the remote module, as it can introduce security vulnerabilities and performance issues. Use IPC instead.
  - **Anti-patterns:**
    - **Tight Coupling:** Avoid tightly coupling components or modules to make the code more flexible and testable.
    - **Global State:** Minimize the use of global state.  Use state management libraries or techniques to manage application state effectively.
    - **Long-Running Tasks in the Renderer Process:** Move long-running tasks to the main process or use worker threads to prevent blocking the UI.
  - **State Management:**
    - For simple applications, use basic state management techniques (e.g., component state, context API).
    - For complex applications, consider using state management libraries like Redux, Zustand, or Vuex.
  - **Error Handling:**
    - Implement robust error handling mechanisms throughout your application.
    - Use try-catch blocks to handle exceptions and prevent crashes.
    - Log errors to a file or remote logging service for debugging.

- **Performance Considerations**
  - **Optimization Techniques:**
    - **Reduce Load on the Renderer Process:** Minimize the amount of work performed in the renderer process. Offload heavy tasks to the main process or worker threads.
    - **Hardware Acceleration:** Enable hardware acceleration to improve rendering performance.
    - **Minimize DOM Manipulations:** Reduce the number of DOM manipulations to improve UI responsiveness.
  - **Memory Management:**
    - **Garbage Collection:** Be mindful of memory leaks and optimize code to reduce memory consumption.
    - **Object Pools:** Use object pools to reuse objects and avoid unnecessary memory allocation.
  - **Rendering Optimization:**
    - **Virtual DOM:** Use a virtual DOM library (e.g., React, Vue) to optimize rendering performance.
    - **CSS Sprites:** Use CSS sprites to reduce the number of HTTP requests for images.
  - **Bundle Size Optimization:**
    - **Code Minification:** Minify JavaScript and CSS code to reduce bundle size.
    - **Tree Shaking:** Use tree shaking to remove unused code from the bundle.
  - **Lazy Loading:**
    - **Lazy Load Images:** Lazy load images to improve initial page load time.
    - **Lazy Load Modules:** Lazy load modules on demand to reduce initial bundle size.

- **Security Best Practices**
  - **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user input and escaping output.
    - **Remote Code Execution (RCE):** Avoid RCE vulnerabilities by validating input and preventing the execution of arbitrary code.
    - **Insecure Deserialization:** Prevent insecure deserialization vulnerabilities by validating serialized data.
  - **Input Validation:**
    - Validate all user input to prevent malicious data from being processed.
    - Use appropriate data types and formats to prevent type confusion attacks.
  - **Authentication and Authorization:**
    - Implement secure authentication and authorization mechanisms to protect sensitive data and resources.
    - Use strong passwords and multi-factor authentication.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure storage mechanisms to protect sensitive data.
  - **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Validate API responses to prevent data injection attacks.
  - **Enable Context Isolation:** Enable context isolation for the renderer process to prevent access to the main process's JavaScript context.
  - **Disable Node.js Integration:** Disable Node.js integration in the renderer process unless strictly necessary. If needed, only expose a limited API using contextBridge.
  - **Handle `new-window` events:** Validate and sanitize all URLs before allowing a new window to be opened from a renderer process.

- **Testing Approaches**
  - **Unit Testing:**
    - Write unit tests to verify the functionality of individual components or modules.
    - Use a testing framework like Jest or Mocha.
  - **Integration Testing:**
    - Write integration tests to verify the interaction between different components or modules.
    - Use a testing framework like Cypress or Puppeteer.
  - **End-to-End Testing:**
    - Write end-to-end tests to verify the overall functionality of the application.
    - Use a testing framework like Cypress or Puppeteer.
  - **Test Organization:**
    - Organize tests into separate files or directories based on functionality.
    - Use descriptive test names to clearly identify the purpose of each test.
  - **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components or modules during testing.
    - Use a mocking library like Jest or Sinon.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - **Not Sanitizing User Input:** Failing to sanitize user input can lead to XSS vulnerabilities.
    - **Using the Remote Module:** The `remote` module provides direct access to main process functionality from the renderer, but bypassing IPC exposes security risks. Use `contextBridge` and IPC.
    - **Blocking the Main Process:** Performing long-running tasks in the main process can block the UI and make the application unresponsive.
  - **Edge Cases:**
    - **Handling Different Screen Resolutions:** Test your application on different screen resolutions to ensure proper UI rendering.
    - **Handling Different Operating Systems:** Test your application on different operating systems (Windows, macOS, Linux) to ensure compatibility.
  - **Version-Specific Issues:**
    - Be aware of version-specific issues and compatibility concerns when upgrading Electron or its dependencies.
  - **Debugging Strategies:**
    - Use the Chrome DevTools to debug the renderer process.
    - Use the Electron DevTools to debug the main process.

- **Tooling and Environment**
  - **Recommended Development Tools:**
    - **Visual Studio Code:** A popular code editor with excellent support for Electron development.
    - **Electron Forge or Electron Builder:** Tools for packaging and distributing Electron applications.
    - **Chrome DevTools:** A powerful debugging tool for the renderer process.
  - **Build Configuration:**
    - Use a build system like webpack or Parcel to bundle your application code.
    - Configure the build system to optimize code for production.
  - **Linting and Formatting:**
    - Use a linter like ESLint to enforce code style and prevent errors.
    - Use a formatter like Prettier to automatically format code.
  - **Deployment Best Practices:**
    - Sign your application code to prevent tampering.
    - Distribute your application through a secure channel.
  - **CI/CD Integration:**
    - Integrate your application with a CI/CD system to automate the build, test, and deployment process.