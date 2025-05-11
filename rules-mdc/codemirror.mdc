---
description: This rule provides guidelines for using CodeMirror effectively, covering code organization, performance, security, testing, and common pitfalls. It aims to ensure robust and maintainable code editor implementations.
globs: *.js,*.ts,*.html,*.css,*.vue,*.svelte,*.jsx,*.tsx
---
# CodeMirror Best Practices and Coding Standards

This document outlines best practices and coding standards for developing text editors and code-related applications using CodeMirror. Adhering to these guidelines will help ensure maintainability, performance, and a positive user experience.

## 1. Code Organization and Structure

### 1.1. Directory Structure

*   **Project Root:**
    *   `.cursor/rules/`: (if using cursor rules) stores project-specific rules. Each rule should correspond to a specific aspect of the project or library.
    *   `src/`: Contains the main source code for your CodeMirror integration.
    *   `dist/`: Stores the built or compiled output.
    *   `node_modules/`: Contains project dependencies (managed by npm or yarn).
    *   `package.json`: Defines project metadata and dependencies.
    *   `webpack.config.js` or `vite.config.js`: Configuration file for the bundler (if using one).
    *   `.eslintrc.js`, `.prettierrc.js`: Configuration files for linting and formatting.
    *   `tests/`: Contains unit, integration, and end-to-end tests.
    *   `README.md`: Project documentation.

*   **`src/` directory:**
    *   `components/`: Reusable CodeMirror components (e.g., custom toolbars, panels).
    *   `modes/`: Custom language modes (if any).
    *   `addons/`: Custom CodeMirror addons.
    *   `utils/` or `helpers/`: Utility functions.
    *   `config/`: Configuration files (e.g., CodeMirror options).
    *   `styles/`: Custom CSS or styling modules for CodeMirror.
    *   `index.js` or `index.ts`: Entry point for the CodeMirror integration.

### 1.2. File Naming Conventions

*   **Components:** Use PascalCase (e.g., `EditorToolbar.js`, `SyntaxHighlightingAddon.ts`).
*   **Modules:** Use camelCase (e.g., `editorConfig.js`, `stringUtils.ts`).
*   **Styles:** Use kebab-case (e.g., `editor-styles.css`, `toolbar-theme.scss`).
*   **Test Files:** Append `.test.js` or `.spec.ts` to the corresponding component/module name (e.g., `EditorToolbar.test.js`).

### 1.3. Module Organization

*   **Modular Design:** Break down CodeMirror-related functionality into small, reusable modules.
*   **Single Responsibility Principle:** Each module should have a clear and specific purpose.
*   **Dependency Management:** Explicitly declare dependencies within each module.  Use ES modules (`import`/`export`) or CommonJS (`require`/`module.exports`).
*   **Avoid Global State:** Minimize the use of global variables or shared mutable state. Pass data and configuration options explicitly to functions and components.

### 1.4. Component Architecture

*   **Presentational and Container Components:**  Separate concerns by creating presentational components (responsible for rendering UI) and container components (responsible for data fetching and state management).
*   **Component Composition:** Build complex UIs by composing smaller, reusable components.
*   **Props and Events:** Use props to pass data down to components and events to communicate actions up to parent components.

### 1.5. Code Splitting

*   **Dynamic Imports:** Use dynamic imports (`import()`) to load CodeMirror modes and addons on demand.
*   **Route-Based Splitting:** If the CodeMirror editor is only used on specific routes in your application, load it only when those routes are visited.
*   **Webpack or Vite Configuration:** Configure your bundler to automatically split your code into smaller chunks.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Observer Pattern:** Use the Observer pattern to notify other parts of the application when the editor's content changes.
*   **Strategy Pattern:** Implement different editing strategies based on the selected language mode.
*   **Factory Pattern:** Use a factory function to create CodeMirror instances with specific configurations.

### 2.2. Recommended Approaches

*   **Configuration Objects:** Encapsulate CodeMirror options in a configuration object to improve readability and maintainability.
*   **Custom Addons:** Create custom CodeMirror addons to extend the editor's functionality.
*   **Event Handling:** Use CodeMirror's event system to respond to user interactions and editor changes.

### 2.3. Anti-patterns

*   **Direct DOM Manipulation:** Avoid directly manipulating the CodeMirror DOM elements. Use the CodeMirror API instead.
*   **Overly Complex Modes:** Keep language modes focused and well-structured.  Delegate complex parsing logic to external libraries if necessary.
*   **Ignoring Performance:** Be mindful of performance implications when adding new features or customizations.

### 2.4. State Management

*   **Local State:** Use component state for simple, editor-specific data.
*   **Redux or Context API:** For larger applications, consider using a state management library like Redux or the React Context API to manage the editor's state centrally.
*   **Immutability:** Treat the editor's state as immutable to simplify debugging and improve performance.

### 2.5. Error Handling

*   **Try-Catch Blocks:** Use try-catch blocks to handle potential errors when interacting with the CodeMirror API.
*   **Error Boundaries:** In React applications, use error boundaries to catch errors that occur during rendering.
*   **Logging:** Log errors and warnings to aid in debugging.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Lazy Loading:** Load CodeMirror modes and addons only when they are needed.
*   **Debouncing:** Debounce event handlers to prevent excessive updates.
*   **Virtualization:** If you are displaying a large number of lines, consider using a virtualization technique to render only the visible lines.
*   **Minimize DOM Updates:** Reduce the number of DOM updates by batching changes and using efficient rendering techniques.

### 3.2. Memory Management

*   **Remove Event Listeners:** Remove event listeners when components are unmounted to prevent memory leaks.
*   **Clear Intervals and Timeouts:** Clear any intervals or timeouts that are no longer needed.
*   **Object Reuse:**  Reuse objects and data structures whenever possible to reduce memory allocation.

### 3.3. Rendering Optimization

*   **ShouldComponentUpdate:** In React applications, use `shouldComponentUpdate` or `React.memo` to prevent unnecessary re-renders.
*   **Immutable Data Structures:** Use immutable data structures to efficiently detect changes and trigger re-renders.

### 3.4. Bundle Size Optimization

*   **Tree Shaking:** Configure your bundler to remove unused code (tree shaking).
*   **Minification:** Minify your code to reduce its size.
*   **Gzip Compression:** Use Gzip compression to reduce the size of your assets during transfer.

### 3.5. Lazy Loading Strategies

*   **Mode-Specific Loading:** Only load the language mode when the user opens a file of that type.
*   **Addon-Specific Loading:** Load addons only when the user needs their functionality.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Cross-Site Scripting (XSS):** Be careful when displaying user-generated content in the editor. Sanitize the input to prevent XSS attacks.
*   **Code Injection:** Avoid using `eval()` or `Function()` to execute untrusted code.

### 4.2. Input Validation

*   **Sanitize Input:** Sanitize user input before displaying it in the editor.
*   **Validate Data:** Validate data received from external sources before using it in the editor.

### 4.3. Authentication and Authorization

*   **Secure APIs:** Use secure APIs for data access and authentication.
*   **Role-Based Access Control:** Implement role-based access control to restrict access to sensitive features.

### 4.4. Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Secure Storage:** Store data in a secure location with appropriate access controls.

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **API Keys:** Protect API keys and prevent them from being exposed in the client-side code.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Components:** Unit test individual CodeMirror components and modules.
*   **Mock Dependencies:** Mock external dependencies to isolate the components being tested.
*   **Assertion Libraries:** Use assertion libraries like Jest or Chai to write expressive tests.

### 5.2. Integration Testing

*   **Test Interactions:** Test the interactions between different CodeMirror components and modules.
*   **Real DOM:** Use a real DOM environment for integration testing.

### 5.3. End-to-End Testing

*   **Simulate User Actions:** Simulate user actions to test the entire CodeMirror workflow.
*   **Automated Testing:** Use end-to-end testing frameworks like Cypress or Puppeteer to automate the testing process.

### 5.4. Test Organization

*   **Test Directory:** Create a dedicated `tests/` directory for all tests.
*   **Test Files:** Place test files alongside the corresponding components/modules.
*   **Test Suites:** Organize tests into logical suites based on functionality.

### 5.5. Mocking and Stubbing

*   **Mock CodeMirror Instances:** Mock CodeMirror instances to control their behavior during testing.
*   **Stub API Calls:** Stub API calls to prevent external dependencies from interfering with the tests.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Not Sanitizing User Input:** Failing to sanitize user input can lead to XSS vulnerabilities.
*   **Ignoring Performance Issues:** Neglecting performance considerations can result in a slow and unresponsive editor.
*   **Direct DOM Manipulation:**  Directly manipulating CodeMirror's DOM can break its internal workings.

### 6.2. Edge Cases

*   **Handling Large Files:** CodeMirror can struggle with extremely large files. Consider using virtualization or other optimization techniques.
*   **Internationalization (i18n):** Ensure that the editor supports different languages and character sets.
*   **Accessibility (a11y):**  Make the editor accessible to users with disabilities by providing keyboard navigation, screen reader support, and appropriate ARIA attributes.

### 6.3. Version-Specific Issues

*   **Check Release Notes:** Review the CodeMirror release notes for any breaking changes or known issues.
*   **Test Upgrades:** Thoroughly test CodeMirror upgrades to ensure compatibility with your existing code.

### 6.4. Compatibility Concerns

*   **Browser Compatibility:** Test the editor in different browsers to ensure compatibility.
*   **Framework Compatibility:**  Ensure that CodeMirror integrates correctly with your chosen framework (e.g., React, Vue, Angular).

### 6.5. Debugging Strategies

*   **Browser Developer Tools:** Use the browser developer tools to inspect the CodeMirror DOM, network requests, and console output.
*   **Logging:** Add logging statements to your code to track the flow of execution and identify potential issues.
*   **Debugging Tools:** Use a debugger to step through your code and examine variables.

## 7. Tooling and Environment

### 7.1. Recommended Tools

*   **Code Editor:** VS Code, Sublime Text, or Atom with CodeMirror-specific plugins.
*   **Bundler:** Webpack, Vite, or Parcel.
*   **Linter:** ESLint or JSHint.
*   **Formatter:** Prettier.
*   **Testing Framework:** Jest, Mocha, or Jasmine.
*   **End-to-End Testing Framework:** Cypress or Puppeteer.

### 7.2. Build Configuration

*   **Module Bundling:** Use a module bundler to combine your code and dependencies into optimized bundles.
*   **Source Maps:** Generate source maps to aid in debugging.
*   **Code Optimization:** Configure your build process to optimize code for production (e.g., minification, tree shaking).

### 7.3. Linting and Formatting

*   **Consistent Style:** Use a linter and formatter to enforce a consistent coding style.
*   **Code Quality:** Configure the linter to catch potential code quality issues (e.g., unused variables, syntax errors).

### 7.4. Deployment Best Practices

*   **Optimize Assets:** Optimize your assets (e.g., images, CSS, JavaScript) for production.
*   **Content Delivery Network (CDN):** Use a CDN to deliver CodeMirror assets to users from geographically distributed servers.
*   **Caching:** Configure browser caching to reduce the number of requests to your server.

### 7.5. CI/CD Integration

*   **Automated Builds:** Automate the build process using a CI/CD pipeline.
*   **Automated Tests:** Run automated tests as part of the CI/CD pipeline.
*   **Deployment Automation:** Automate the deployment process to reduce the risk of errors.