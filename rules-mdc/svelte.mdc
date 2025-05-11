---
description: Comprehensive Svelte best practices covering code structure, performance, security, testing, and common pitfalls. This rule provides guidance on writing maintainable, efficient, and secure Svelte applications.
globs: *.svelte
---
- **Manipulate the DOM with HTML:** Utilize Svelte's HTML-centric approach to manipulate the DOM.
- **Solve challenges with HTML/CSS first:** Prioritize solving problems with HTML and CSS before resorting to JavaScript.
- **Write short components:** Aim for concise and focused components to improve readability and maintainability.
- **Write concise reactive `$:` blocks:** Keep reactive statements short and easy to understand.
- **Scope CSS to the actual component:** Ensure that CSS styles are scoped to the component to avoid conflicts.
- **Avoid hard coding data or dimensions:** Use dynamic data and responsive design techniques instead of hard-coded values.
- **Keep objects immutable:** Treat objects as immutable to prevent unexpected side effects and improve performance.
- **Use two-way binding effectively:** Leverage two-way binding for efficient data synchronization but be mindful of its potential impact on performance for complex scenarios.

## 1. Code Organization and Structure:

   - **Directory Structure Best Practices:**
      - `src/`: Contains all the application source code.
      - `src/components/`: Holds reusable Svelte components, categorized further by feature or domain (e.g., `src/components/Button/`, `src/components/Form/`).
      - `src/lib/`: Contains utility functions, helper modules, and reusable logic that is not specific to a component.
      - `src/routes/`: For SvelteKit applications, this directory defines the application's routes.
      - `src/stores/`: Stores for global state management (if using Svelte's stores).
      - `static/`: Static assets like images, fonts, and other resources.
   - **File Naming Conventions:**
      - Components: Use PascalCase (e.g., `MyComponent.svelte`).
      - Utility functions: Use camelCase (e.g., `formatDate.js`).
      - Stores: Use descriptive names related to the data they hold (e.g., `userStore.js`).
   - **Module Organization:**
      - Group related components and utilities into modules within their respective directories.
      - Use `index.js` or `index.svelte` files to export multiple components or functions from a directory, providing a cleaner import experience.
   - **Component Architecture:**
      - Favor a component-based architecture where UI is broken down into small, reusable components.
      - Consider using a composition pattern where complex components are built by composing simpler ones.
      - Separate concerns: Keep components focused on presentation logic and delegate data fetching or business logic to services or stores.
   - **Code Splitting Strategies:**
      - Use dynamic imports (`import()`) to load components or modules on demand, reducing the initial bundle size.
      - Leverage SvelteKit's built-in code splitting capabilities for route-based splitting.
      - Consider splitting large components into smaller, lazily loaded sub-components.

## 2. Common Patterns and Anti-patterns:

   - **Design Patterns Specific to Svelte:**
      - **Store pattern:** Use Svelte's stores for managing application state and reactivity.
      - **Action pattern:** Use Svelte's actions for manipulating DOM elements or integrating with third-party libraries.
      - **Component composition:** Build complex UIs by composing smaller, reusable components.
   - **Recommended Approaches for Common Tasks:**
      - **Form handling:** Use Svelte's two-way binding (`bind:value`) for simple forms.  For more complex scenarios, consider libraries like Formik or Svelte Formly.
      - **Data fetching:** Use `fetch` or libraries like Axios for fetching data from APIs.  Handle loading and error states appropriately.
      - **Event handling:** Use Svelte's event directives (e.g., `on:click`) for handling DOM events.
   - **Anti-patterns and Code Smells to Avoid:**
      - **Overusing global state:** Avoid putting everything in a global store.  Use component-level state when appropriate.
      - **Directly manipulating the DOM outside of actions:**  Rely on Svelte's reactivity system to update the DOM.
      - **Writing overly complex components:** Break down large components into smaller, more manageable ones.
   - **State Management Best Practices:**
      - Use Svelte's built-in stores for managing global or application-level state.
      - Consider using reactive declarations (`$:`) to derive state from other state variables.
      - Keep state updates predictable and avoid modifying state directly.
      - For complex state management needs, explore libraries like Redux or Zustand (though often unnecessary in Svelte).
   - **Error Handling Patterns:**
      - Use try-catch blocks to handle potential errors during data fetching or other asynchronous operations.
      - Display user-friendly error messages in the UI.
      - Log errors to the console or a logging service for debugging purposes.
      - Implement global error handling to catch unhandled exceptions.

## 3. Performance Considerations:

   - **Optimization Techniques:**
      - **Minimize DOM updates:** Svelte is very efficient at updating the DOM, but unnecessary updates can still impact performance.  Use reactive declarations judiciously.
      - **Use `{#each}` blocks efficiently:** Key your `{#each}` blocks with unique identifiers to help Svelte efficiently update the list.
      - **Avoid unnecessary component re-renders:** Use the `$:` syntax effectively to only update components when necessary.
   - **Memory Management:**
      - Avoid memory leaks by properly cleaning up event listeners and subscriptions when components are destroyed (using `onDestroy`).
      - Be mindful of large data structures in stores, and consider using techniques like pagination or virtualization to manage them efficiently.
   - **Rendering Optimization:**
      - Use the `svelte:options` tag with the `immutable` option for components that receive immutable data as props.
      - Use `shouldUpdate` to prevent rendering for unchanged immutable props.
   - **Bundle Size Optimization:**
      - Use production builds with minification and tree shaking to remove unused code.
      - Use dynamic imports for code splitting.
      - Optimize images and other assets.
   - **Lazy Loading Strategies:**
      - Use dynamic imports to lazy load components or modules that are not immediately needed.
      - Implement lazy loading for images or other resources that are below the fold.

## 4. Security Best Practices:

   - **Common Vulnerabilities and How to Prevent Them:**
      - **Cross-Site Scripting (XSS):** Sanitize user input before displaying it in the UI.  Svelte automatically escapes HTML entities, but be careful when using `@html` or `{@debug}`.
      - **Cross-Site Request Forgery (CSRF):** Use CSRF tokens to protect against CSRF attacks.
      - **SQL Injection:** If interacting with a database, use parameterized queries to prevent SQL injection attacks.
   - **Input Validation:**
      - Validate user input on both the client-side and server-side.
      - Use appropriate data types and validation rules to prevent invalid data from being processed.
   - **Authentication and Authorization Patterns:**
      - Use secure authentication mechanisms like OAuth or JWT.
      - Implement proper authorization checks to ensure that users only have access to the resources they are allowed to access.
   - **Data Protection Strategies:**
      - Encrypt sensitive data at rest and in transit.
      - Use secure storage mechanisms for storing sensitive data.
   - **Secure API Communication:**
      - Use HTTPS for all API communication.
      - Validate API responses to ensure that they are valid and not malicious.

## 5. Testing Approaches:

   - **Unit Testing Strategies:**
      - Use a testing framework like Jest or Mocha to write unit tests for individual components and utility functions.
      - Mock external dependencies to isolate the code being tested.
   - **Integration Testing:**
      - Use a testing framework like Cypress or Playwright to write integration tests that verify the interaction between different components or modules.
      - Test the integration with APIs and other external services.
   - **End-to-End Testing:**
      - Use a testing framework like Cypress or Playwright to write end-to-end tests that simulate user interactions with the application.
      - Test the entire application flow from start to finish.
   - **Test Organization:**
      - Organize tests into directories that mirror the application's directory structure.
      - Use descriptive names for test files and test cases.
   - **Mocking and Stubbing:**
      - Use mocking libraries like Jest's `jest.fn()` or Sinon.js to mock external dependencies.
      - Use stubbing to replace complex or slow dependencies with simpler implementations for testing purposes.

## 6. Common Pitfalls and Gotchas:

   - **Frequent Mistakes Developers Make:**
      - **Incorrect reactivity usage:**  Failing to understand how Svelte's reactivity system works can lead to unexpected behavior.
      - **Over-reliance on `$: `:** Although reactive declarations are powerful, overusing them can make code harder to read and debug. Consider if a regular variable or function would be more appropriate.
      - **Mutating props directly:**  Modifying props directly inside a component can lead to unexpected side effects.
   - **Edge Cases to Be Aware Of:**
      - **Transition and animation behavior:**  Understanding how transitions and animations interact with component updates is important for creating smooth user experiences.
      - **Server-side rendering (SSR) considerations:**  When using SvelteKit, be aware of the differences between client-side and server-side execution environments.
   - **Version-Specific Issues:**
      - Consult the Svelte changelog and migration guides when upgrading to a new version to avoid compatibility issues.
   - **Compatibility Concerns:**
      - Ensure that your code is compatible with the target browsers and devices.
      - Use polyfills or transpilers to support older browsers if necessary.
   - **Debugging Strategies:**
      - Use the Svelte Devtools browser extension for inspecting component state and reactivity.
      - Use `console.log` or the `debugger` statement to debug code execution.
      - Use SvelteKit's built-in error handling and logging features.

## 7. Tooling and Environment:

   - **Recommended Development Tools:**
      - **VS Code with the Svelte extension:** Provides syntax highlighting, code completion, and other useful features.
      - **Svelte Devtools:** A browser extension for inspecting Svelte component state and reactivity.
      - **ESLint and Prettier:** For linting and formatting code.
   - **Build Configuration:**
      - Use a build tool like Vite or Rollup to bundle your Svelte code for production.
      - Configure the build tool to perform optimizations like minification and tree shaking.
   - **Linting and Formatting:**
      - Use ESLint with the `eslint-plugin-svelte3` plugin to lint your Svelte code.
      - Use Prettier with the `prettier-plugin-svelte` plugin to format your Svelte code.
   - **Deployment Best Practices:**
      - Deploy your Svelte application to a CDN or a serverless platform like Netlify or Vercel.
      - Configure your server to serve the application's static assets with proper caching headers.
   - **CI/CD Integration:**
      - Use a CI/CD platform like GitHub Actions or GitLab CI to automate the build, testing, and deployment process.
      - Run tests and linting checks on every commit to ensure code quality.