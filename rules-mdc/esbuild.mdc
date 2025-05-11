---
description: This rule provides comprehensive best practices for using esbuild, focusing on performance, code organization, and security in build configurations and development workflows.
globs: *.js,*.jsx,*.ts,*.tsx,*.css,*.json
---
# esbuild Best Practices

This document outlines best practices for using esbuild as a JavaScript bundler. It covers various aspects, including code organization, performance optimization, security considerations, and common pitfalls to avoid.

## Library Information:

- Name: esbuild
- Tags: build-tool, javascript, bundler, performance

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

-   **Source Code Directory (`src`):**  House all your source code within a `src` directory. This promotes a clear separation of concerns between your source and build artifacts.
-   **Configuration Files (`config` or `.config`):** Store esbuild configuration files (e.g., `esbuild.config.js`, `build.js`) in a dedicated `config` or `.config` directory.  This makes it easy to identify and manage build-related settings.
-   **Assets Directory (`assets`):** Place static assets like images, fonts, and other non-code files in an `assets` directory. This keeps them separate from your code and simplifies asset management.
-   **Output Directory (`dist` or `build`):**  esbuild commonly uses `dist` or `build` directories for storing the bundled output.  Configure esbuild to output to one of these standard directories.

Example:


project-root/
├── src/
│   ├── components/
│   │   ├── MyComponent.tsx
│   │   └── ...
│   ├── utils/
│   │   ├── helpers.ts
│   │   └── ...
│   ├── index.tsx
│   └── ...
├── config/
│   ├── esbuild.config.js
│   └── ...
├── assets/
│   ├── images/
│   │   ├── logo.png
│   │   └── ...
│   ├── fonts/
│   │   ├── OpenSans.woff2
│   │   └── ...
├── dist/
│   ├── bundle.js
│   ├── styles.css
│   └── ...
├── package.json
├── tsconfig.json
└── ...


### 1.2. File Naming Conventions

-   **Consistent Case:** Use either camelCase or PascalCase for JavaScript/TypeScript files, but consistently within your project.  PascalCase is generally preferred for React components.
-   **Descriptive Names:** Choose names that clearly reflect the file's purpose (e.g., `userProfile.tsx`, `apiClient.ts`).
-   **Module-Specific Names:** If a file exports a single primary entity (e.g., a React component), use the same name for the file (e.g., `MyComponent.tsx` exports `MyComponent`).
-   **CSS Modules:** Use `.module.css` or `.module.scss` for CSS Modules to scope styles locally to a component.
-   **Configuration Files:** use `esbuild.config.js` or `build.js` for esbuild's configuration to make it easily identifiable.

### 1.3. Module Organization Best Practices

-   **Feature-Based Modules:** Organize code by feature or functionality.  Each feature should have its own directory containing all related code (components, utils, styles, etc.).
-   **Reusable Modules:**  Extract reusable code into separate modules (e.g., utility functions, API clients).  Place these modules in a `utils` or `services` directory.
-   **Avoid Circular Dependencies:**  Circular dependencies can lead to unexpected behavior and bundling issues. Use tools like `madge` to detect and eliminate them.
-   **Explicit Exports:** Be explicit about what you export from each module using `export` statements. This improves code clarity and tree-shaking.

### 1.4. Component Architecture Recommendations

-   **Component-Based Architecture:**  Adopt a component-based architecture (e.g., using React, Vue, or Svelte).  This promotes code reusability, testability, and maintainability.
-   **Atomic Design:**  Consider using the Atomic Design methodology to structure components into atoms, molecules, organisms, templates, and pages.
-   **Separation of Concerns:**  Separate presentation logic (UI) from business logic (data fetching, state management).  Use techniques like custom hooks to extract and reuse business logic.

### 1.5. Code Splitting Strategies

-   **Dynamic Imports:** Use dynamic imports (`import()`) to load code on demand.  This can significantly reduce the initial bundle size and improve page load performance.  esbuild supports dynamic imports out of the box.
-   **Route-Based Splitting:** Split your application into separate bundles for each route or page.  This ensures that users only download the code they need for the current page.
-   **Vendor Splitting:**  Separate vendor libraries (e.g., React, Lodash) into a separate bundle.  This allows browsers to cache vendor code separately from your application code.
-   **Entry Points:** Create multiple entry points for distinct parts of your application (e.g. a landing page vs. an admin panel). esbuild will bundle these into separate output files.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Module Pattern:** Use the module pattern to encapsulate code and create private variables and functions.
-   **Factory Pattern:** Use factory functions to create objects or components. This allows you to abstract the creation process and easily configure objects with different options.
-   **Higher-Order Components (HOCs):**  (React-specific) Use HOCs to add functionality to existing components.
-   **Custom Hooks:** (React-specific) Use custom hooks to extract and reuse stateful logic.

### 2.2. Recommended Approaches for Common Tasks

-   **Environment Variables:** Use environment variables to configure your application for different environments (development, staging, production).  Access environment variables using `process.env`.
-   **Path Aliases:** Configure path aliases to simplify imports.  For example, you can alias `@components` to `src/components`.  esbuild can be configured to understand these aliases.
-   **CSS Preprocessing:** Integrate CSS preprocessors like Sass or Less using esbuild plugins.  This allows you to use features like variables, mixins, and nesting in your CSS.
-   **Minification and Tree Shaking:** Always enable minification and tree shaking for production builds to reduce bundle size. esbuild does this automatically with the `--minify` flag.

### 2.3. Anti-patterns and Code Smells

-   **Global Variables:** Avoid using global variables as they can lead to naming conflicts and make it difficult to reason about your code.
-   **Long Component Files:**  Break down large component files into smaller, more manageable components.
-   **Deeply Nested Components:**  Avoid deeply nested component structures as they can make it difficult to understand the component hierarchy.
-   **Over-reliance on `any`:** In TypeScript, avoid using `any` excessively as it defeats the purpose of type checking.  Use more specific types whenever possible.
-   **Direct DOM Manipulation (in React/Vue):** Avoid directly manipulating the DOM. Rely on the framework's virtual DOM for efficient updates.

### 2.4. State Management Best Practices

-   **Component State:** Use component state (e.g., `useState` in React) for simple, localized state management.
-   **Context API:** (React-specific) Use the Context API to share state between components without prop drilling.
-   **Redux/Zustand/Recoil:** Use a state management library like Redux, Zustand, or Recoil for more complex application state.
-   **Immutability:**  Maintain immutability when updating state to avoid unexpected side effects and improve performance (especially with React).

### 2.5. Error Handling Patterns

-   **Try-Catch Blocks:** Use `try-catch` blocks to handle synchronous errors.
-   **Async/Await Error Handling:** Use `try-catch` blocks with `async/await` to handle asynchronous errors.
-   **Error Boundaries:** (React-specific) Use error boundaries to catch errors that occur during rendering and prevent the entire application from crashing.
-   **Centralized Error Logging:** Implement a centralized error logging system to track errors in your application.

## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Minification:** Use esbuild's built-in minification (`--minify`) to reduce the size of your JavaScript and CSS files.
-   **Tree Shaking:**  esbuild automatically performs tree shaking to remove unused code. Ensure that your code is written in a way that allows for efficient tree shaking (e.g., using ES modules with explicit exports).
-   **Code Splitting:** Implement code splitting using dynamic imports and route-based splitting to reduce the initial bundle size.
-   **Image Optimization:** Optimize images using tools like ImageOptim or TinyPNG to reduce their file size.
-   **Caching:**  Configure your server to cache static assets (JavaScript, CSS, images) to improve loading times for returning users.
-   **Compression:** Enable gzip or Brotli compression on your server to reduce the size of files transmitted over the network.
-   **Target Specific Environments:** Use the `target` option to specify the target JavaScript environment.  This allows esbuild to generate code that is optimized for the specific environment.
-   **Incremental Builds:** Utilize esbuild's `--watch` option and the context API for incremental builds, which significantly speeds up development by only recompiling changed files.

### 3.2. Memory Management

-   **Avoid Memory Leaks:** Be mindful of memory leaks, especially in long-running applications.  Remove event listeners and clear timers when they are no longer needed.
-   **Use WeakRefs:** Consider using `WeakRef` in situations where you need to hold a reference to an object without preventing it from being garbage collected.
-   **Profile Your Code:** Use browser developer tools to profile your code and identify memory bottlenecks.

### 3.3. Rendering Optimization (If Applicable)

-   **Virtualization:** Use virtualization techniques (e.g., `react-window`, `react-virtualized`) to efficiently render large lists or tables.
-   **Debouncing and Throttling:** Use debouncing and throttling to limit the frequency of expensive operations, such as event handlers or API calls.
-   **Memoization:** Use memoization techniques (e.g., `React.memo`, `useMemo`) to cache the results of expensive calculations.

### 3.4. Bundle Size Optimization

-   **Analyze Bundle Size:** Use tools like `esbuild-visualizer` or `webpack-bundle-analyzer` to analyze your bundle size and identify large dependencies.
-   **Reduce Dependency Size:** Look for opportunities to reduce the size of your dependencies.  Consider using smaller alternatives or only importing the specific parts of a library that you need.
-   **Remove Dead Code:**  Ensure that tree shaking is working effectively to remove unused code.

### 3.5. Lazy Loading Strategies

-   **Lazy-Load Components:** Use dynamic imports to lazy-load components that are not immediately needed.
-   **Lazy-Load Images:** Use lazy-loading for images that are below the fold to improve initial page load time.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

-   **Cross-Site Scripting (XSS):** Prevent XSS attacks by properly escaping user input and using a Content Security Policy (CSP).
-   **Injection Attacks:** Prevent injection attacks (e.g., SQL injection, command injection) by validating and sanitizing user input.
-   **Dependency Vulnerabilities:** Regularly audit your dependencies for known vulnerabilities using tools like `npm audit` or `yarn audit`.  Update to the latest versions of your dependencies to patch vulnerabilities.

### 4.2. Input Validation

-   **Validate All Input:** Validate all user input, both on the client-side and the server-side.
-   **Use Strong Validation Rules:** Use strong validation rules to ensure that input is in the expected format and range.
-   **Sanitize Input:** Sanitize input to remove potentially malicious characters or code.

### 4.3. Authentication and Authorization

-   **Use Strong Authentication:** Use strong authentication methods, such as multi-factor authentication (MFA).
-   **Implement Proper Authorization:** Implement proper authorization to ensure that users only have access to the resources they are authorized to access.
-   **Securely Store Credentials:** Securely store user credentials using hashing and salting.

### 4.4. Data Protection

-   **Encrypt Sensitive Data:** Encrypt sensitive data both in transit and at rest.
-   **Use HTTPS:** Use HTTPS to encrypt communication between the client and the server.
-   **Protect API Keys:** Protect API keys and other sensitive configuration data by storing them in environment variables or a secure configuration store.

### 4.5. Secure API Communication

-   **Use HTTPS:** Always use HTTPS for API communication.
-   **Validate API Responses:** Validate API responses to ensure that they are in the expected format.
-   **Implement Rate Limiting:** Implement rate limiting to prevent abuse of your API.

## 5. Testing Approaches

### 5.1. Unit Testing

-   **Test Individual Components:** Unit tests should focus on testing individual components or modules in isolation.
-   **Use Mocking and Stubbing:** Use mocking and stubbing to isolate the component under test from its dependencies.
-   **Test Edge Cases:** Test edge cases and error conditions to ensure that the component handles them correctly.

### 5.2. Integration Testing

-   **Test Interactions Between Components:** Integration tests should focus on testing the interactions between different components or modules.
-   **Test API Integrations:** Test integrations with external APIs to ensure that they are working correctly.

### 5.3. End-to-End Testing

-   **Test User Flows:** End-to-end tests should simulate user flows to ensure that the application is working correctly from the user's perspective.
-   **Use a Testing Framework:** Use a testing framework like Cypress or Playwright to automate end-to-end tests.

### 5.4. Test Organization

-   **Co-locate Tests with Code:** Store test files in the same directory as the code they are testing.
-   **Use Descriptive Test Names:** Use descriptive test names to clearly indicate what each test is verifying.

### 5.5. Mocking and Stubbing

-   **Use a Mocking Library:** Use a mocking library like Jest or Sinon to create mocks and stubs.
-   **Mock External Dependencies:** Mock external dependencies, such as API calls or database connections.
-   **Stub Function Behavior:** Stub the behavior of functions to control their return values or side effects.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Forgetting to Bundle:** Failing to enable bundling can result in performance issues due to multiple HTTP requests.
-   **Not Minifying:**  Not minifying your code will result in larger file sizes and slower loading times.
-   **Incorrect `tsconfig.json` Configuration:** Incorrectly configuring your `tsconfig.json` file can lead to compilation errors or unexpected behavior.
-   **Not Handling Environment Variables:** Failing to properly handle environment variables can result in incorrect configuration in different environments.
-   **Not Using Path Aliases:**  Not using path aliases can make imports more verbose and difficult to maintain.

### 6.2. Edge Cases

-   **Circular Dependencies:** Circular dependencies can lead to unexpected behavior and bundling issues.
-   **Dynamic Imports with Variable Paths:**  esbuild's support for dynamic imports with variable paths is limited. Be aware of the restrictions and consider alternative approaches if needed.
-   **Plugin Compatibility:** Ensure that plugins are compatible with the version of esbuild you are using.

### 6.3. Version-Specific Issues

-   **Breaking Changes:** Be aware of breaking changes in new versions of esbuild. Consult the release notes before upgrading.

### 6.4. Compatibility Concerns

-   **Browser Compatibility:**  Ensure that your code is compatible with the target browsers.  Use Babel or other transpilers if necessary.
-   **Node.js Compatibility:**  If you are building a Node.js application, ensure that your code is compatible with the target version of Node.js.

### 6.5. Debugging Strategies

-   **Source Maps:**  Enable source maps to make it easier to debug your code in the browser developer tools.
-   **Console Logging:** Use `console.log` statements to debug your code.
-   **Debugger Statements:**  Use `debugger` statements to pause execution at specific points in your code.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **VS Code:** A popular code editor with excellent support for JavaScript, TypeScript, and esbuild.
-   **ESLint:** A linter for JavaScript and TypeScript that can help you identify and fix code quality issues.
-   **Prettier:** A code formatter that can automatically format your code to a consistent style.
-   **esbuild-visualizer:** A tool to visualize the contents of your esbuild bundle.

### 7.2. Build Configuration Best Practices

-   **Use a Configuration File:** Store your esbuild configuration in a dedicated configuration file (e.g., `esbuild.config.js`).
-   **Separate Configurations for Different Environments:** Create separate configurations for different environments (development, staging, production).
-   **Use Environment Variables:** Use environment variables to configure your build process.

### 7.3. Linting and Formatting

-   **Configure ESLint:** Configure ESLint to enforce coding style and identify potential errors.
-   **Use Prettier:** Use Prettier to automatically format your code to a consistent style.
-   **Integrate Linting and Formatting into Your Workflow:** Integrate linting and formatting into your development workflow using tools like Husky or lint-staged.

### 7.4. Deployment

-   **Use a Build Process:** Use a build process to bundle, minify, and optimize your code before deployment.
-   **Deploy to a CDN:** Deploy static assets to a content delivery network (CDN) for faster loading times.
-   **Use HTTPS:** Always use HTTPS to encrypt communication between the client and the server.

### 7.5. CI/CD Integration

-   **Automate Build and Testing:** Automate your build and testing process using a continuous integration and continuous delivery (CI/CD) pipeline.
-   **Run Linting and Formatting Checks:**  Run linting and formatting checks as part of your CI/CD pipeline.
-   **Deploy Automatically:**  Automate the deployment process to deploy new versions of your application automatically.

This comprehensive guide provides a solid foundation for using esbuild effectively. Remember to adapt these best practices to your specific project needs and continuously learn as the esbuild ecosystem evolves.