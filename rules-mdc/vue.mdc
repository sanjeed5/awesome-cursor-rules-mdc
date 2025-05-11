---
description: Comprehensive guidelines for Vue.js development, covering code structure, performance, security, testing, and tooling best practices. This rule provides actionable guidance to enhance code quality, maintainability, and developer productivity in Vue.js projects.
globs: *.vue, *.js, *.ts
---
# Vue.js Best Practices and Coding Standards

This document outlines best practices for Vue.js development, covering various aspects to ensure high-quality, maintainable, and performant code.

## 1. Code Organization and Structure

### 1.1. Directory Structure

*   **Component-Based Structure:** Organize components into logical folders based on their functionality or feature. This improves code readability and maintainability.
    
    src/
    ├── components/
    │   ├── Button/
    │   │   ├── Button.vue
    │   │   └── Button.spec.js
    │   ├── Input/
    │   │   ├── Input.vue
    │   │   └── Input.spec.js
    │   └── ...
    ├── views/
    │   ├── Home.vue
    │   ├── About.vue
    │   └── ...
    ├── services/
    │   ├── api.js
    │   └── auth.js
    ├── store/
    │   ├── index.js  # Vuex store
    │   ├── modules/
    │   │   ├── user.js
    │   │   └── ...
    ├── App.vue
    └── main.js
    
*   **Feature-Based Structure:** Alternatively, organize files by feature, grouping components, routes, and store modules related to a specific feature.
    
    src/
    ├── features/
    │   ├── user-profile/
    │   │   ├── components/
    │   │   │   ├── UserProfile.vue
    │   │   │   └── ...
    │   │   ├── routes.js
    │   │   ├── store.js
    │   │   └── ...
    │   ├── shopping-cart/
    │   │   ├── ...
    │   └── ...
    ├── App.vue
    └── main.js
    

### 1.2. File Naming Conventions

*   **Component Files:** Use PascalCase for component file names (e.g., `MyComponent.vue`).
*   **Other Files:** Use camelCase or kebab-case for other JavaScript/TypeScript files (e.g., `apiService.js`, `my-helper.js`).
*   **Consistency:** Maintain a consistent naming convention throughout the project.

### 1.3. Module Organization

*   **ES Modules:** Utilize ES modules (`import`/`export`) for modular code organization.
*   **Single Responsibility Principle:** Each module should have a single, well-defined responsibility.
*   **Avoid Circular Dependencies:** Prevent circular dependencies between modules to avoid unexpected behavior and improve maintainability.

### 1.4. Component Architecture

*   **Component Composition:** Favor component composition over inheritance for increased flexibility and reusability.
*   **Presentational and Container Components:** Separate presentational (dumb) components from container (smart) components. Presentational components focus on rendering UI, while container components handle data fetching and logic.
*   **Single File Components (SFCs):** Leverage Vue's SFCs for encapsulating component logic, template, and styling.

### 1.5. Code Splitting Strategies

*   **Route-Based Splitting:** Use dynamic imports and Vue's `async` component feature to split the application into chunks based on routes.
*   **Component-Based Splitting:** Split large components into smaller, lazy-loaded components to improve initial load time.
*   **Vendor Splitting:** Separate vendor dependencies into a separate chunk to allow for browser caching and prevent unnecessary reloads.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to Vue

*   **Provide/Inject:** Use `provide` and `inject` for dependency injection between components, especially when dealing with deeply nested components.
*   **Renderless Components:** Create renderless components that encapsulate logic and provide data to be rendered by slot-using components.
*   **Higher-Order Components (HOCs):** Use HOCs to reuse component logic or add functionality to existing components.

### 2.2. Recommended Approaches for Common Tasks

*   **Form Handling:** Use `v-model` for two-way data binding in forms. Consider using a form validation library like Vuelidate or VeeValidate for robust form validation.
*   **API Requests:** Use a dedicated service module for handling API requests. Use `async/await` for cleaner asynchronous code.
*   **State Management:** Utilize Vuex for centralized state management in larger applications. For simpler applications, consider using Vue's reactivity system directly or a lightweight state management solution like Pinia.
*   **Event Handling:** Use component events (`$emit`) for communication between parent and child components. For communication between unrelated components, use a global event bus (with caution) or a state management solution.

### 2.3. Anti-patterns and Code Smells to Avoid

*   **Mutating Props Directly:** Avoid mutating props directly within a component. Instead, emit an event to the parent component to update the prop value.
*   **Overusing Global State:** Avoid storing too much data in global state. Use local component state whenever possible.
*   **Direct DOM Manipulation:** Avoid directly manipulating the DOM using `document` APIs. Use Vue's template directives and component APIs to update the DOM reactively.
*   **Magic Numbers and Strings:** Avoid using magic numbers and strings directly in the code. Use constants to improve readability and maintainability.
*   **Complex Computed Properties:** Keep computed properties simple and focused. Complex computations should be moved to methods or utility functions.

### 2.4. State Management Best Practices

*   **Single Source of Truth:** Maintain a single source of truth for application state using Vuex or Pinia.
*   **Mutations for State Updates:** Only use mutations to update the state in Vuex. Mutations should be synchronous and atomic.
*   **Actions for Asynchronous Operations:** Use actions to handle asynchronous operations like API requests. Actions can commit mutations to update the state.
*   **Getters for Derived State:** Use getters to derive state from the store. Getters should be pure functions and should not modify the state.
*   **Modularity:** Organize the store into modules to improve maintainability and scalability.

### 2.5. Error Handling Patterns

*   **Centralized Error Handling:** Implement a centralized error handling mechanism to catch and log errors consistently.
*   **Error Boundary Components:** Use error boundary components to catch errors within specific parts of the application and prevent crashes.
*   **User-Friendly Error Messages:** Provide user-friendly error messages to guide users when errors occur.
*   **Logging:** Log errors to a server or error tracking service for monitoring and debugging.
*   **Try-Catch Blocks:** Use `try-catch` blocks to handle potential errors in asynchronous operations or complex computations.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Virtual DOM Optimization:** Vue's virtual DOM implementation is already highly optimized, but avoid unnecessary re-renders by using `v-if` instead of `v-show` when elements are rarely displayed.
*   **Computed Properties and Watchers:** Use computed properties and watchers judiciously. Avoid performing expensive computations in computed properties that are frequently re-evaluated. Debounce or throttle watchers to limit the number of updates.
*   **List Rendering Optimization:** Use the `:key` attribute when rendering lists with `v-for` to help Vue track changes efficiently.  Ensure the keys are unique and stable.
*   **Functional Components:** Use functional components for simple, stateless components to improve rendering performance.
*   **Avoid Inline Templates:**  Use pre-compiled templates in single-file components instead of inline templates (using `<script type="text/x-template">`) for better performance.

### 3.2. Memory Management

*   **Remove Event Listeners:**  When a component is destroyed, remove any event listeners that were added manually (e.g., using `addEventListener`).
*   **Unsubscribe from Observables:**  If using RxJS or other observable libraries, unsubscribe from observables when the component is destroyed to prevent memory leaks.
*   **Release References:**  Release references to large objects or data structures when they are no longer needed to allow the garbage collector to reclaim memory.

### 3.3. Rendering Optimization

*   **Asynchronous Updates:**  Use `Vue.nextTick()` or `setTimeout()` to defer updates that are not immediately needed, allowing the browser to complete rendering tasks.
*   **Debouncing and Throttling:**  Debounce or throttle event handlers that trigger frequent updates to prevent excessive re-renders.
*   **`v-once` Directive:**  Use the `v-once` directive for elements that will never change to improve rendering performance.
*   **Avoid Deeply Nested Components:**  Deeply nested component hierarchies can impact rendering performance. Consider flattening the hierarchy or using techniques like scoped slots to optimize rendering.

### 3.4. Bundle Size Optimization

*   **Code Splitting:** Implement code splitting to reduce the initial bundle size and improve loading time.
*   **Tree Shaking:**  Use a modern build tool like Webpack or Rollup to perform tree shaking and remove unused code from the final bundle.
*   **Minification and Compression:**  Minify and compress the code to reduce the bundle size.
*   **Image Optimization:**  Optimize images by compressing them and using appropriate formats (e.g., WebP) to reduce file sizes.
*   **Lazy Loading:**  Lazy load images, components, and other resources to improve initial load time.

### 3.5. Lazy Loading Strategies

*   **Lazy Loading Components:** Use dynamic imports to lazy load components only when they are needed.
*   **Lazy Loading Images:** Use a lazy loading library to load images only when they are visible in the viewport.
*   **Lazy Loading Routes:** Lazy load routes using Vue Router's `component: () => import('./MyComponent.vue')` syntax.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user input and using Vue's built-in template directives, which automatically escape HTML entities.
*   **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by implementing CSRF tokens in forms and API requests.
*   **SQL Injection:** Prevent SQL injection attacks by using parameterized queries or an ORM with built-in protection.
*   **Man-in-the-Middle (MitM) Attacks:** Use HTTPS to encrypt communication between the client and server and protect against MitM attacks.
*   **Clickjacking:** Prevent clickjacking attacks by setting the `X-Frame-Options` header to `DENY` or `SAMEORIGIN`.

### 4.2. Input Validation

*   **Server-Side Validation:** Always perform server-side validation to ensure data integrity and prevent malicious input.
*   **Client-Side Validation:** Implement client-side validation to provide immediate feedback to users and reduce server load. Use libraries like Vuelidate or VeeValidate.
*   **Sanitization:** Sanitize user input to remove potentially harmful characters or code.

### 4.3. Authentication and Authorization Patterns

*   **JSON Web Tokens (JWT):** Use JWTs for authentication and authorization. Store JWTs securely in the client-side (e.g., using HTTP-only cookies or local storage with encryption).
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to different parts of the application based on user roles.
*   **OAuth 2.0:** Use OAuth 2.0 for third-party authentication and authorization.
*   **Secure Password Storage:** Store passwords securely using a strong hashing algorithm like bcrypt or Argon2.

### 4.4. Data Protection Strategies

*   **Encryption:** Encrypt sensitive data both in transit and at rest.
*   **Data Masking:** Mask sensitive data in the UI to prevent unauthorized access.
*   **Data Minimization:** Collect only the necessary data and avoid storing sensitive data unnecessarily.
*   **Regular Security Audits:** Conduct regular security audits to identify and address potential vulnerabilities.

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **API Authentication:** Implement authentication for all API endpoints using JWTs or other authentication mechanisms.
*   **Rate Limiting:** Implement rate limiting to prevent abuse and denial-of-service attacks.
*   **Input Validation:** Validate all API input to prevent injection attacks.
*   **Output Encoding:** Encode API output to prevent XSS attacks.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   **Component Testing:** Write unit tests for individual Vue components to verify their behavior in isolation. Use a testing library like Jest or Mocha with Vue Test Utils.
*   **Function Testing:** Write unit tests for utility functions and other non-component code.
*   **Test-Driven Development (TDD):** Consider using TDD to write tests before writing the code.

### 5.2. Integration Testing

*   **Component Integration:** Write integration tests to verify the interaction between multiple components.
*   **Module Integration:** Write integration tests to verify the interaction between different modules of the application.
*   **End-to-End Integration:** Write end-to-end integration tests to verify the entire application flow from the user's perspective. Tools like Cypress, Playwright, or Selenium can be used for E2E testing.

### 5.3. End-to-End Testing

*   **User Flow Testing:** Simulate user flows to test the application's functionality from end to end.
*   **Visual Regression Testing:** Use visual regression testing to detect unintended visual changes in the UI.
*   **Accessibility Testing:** Test the application's accessibility to ensure it is usable by people with disabilities.

### 5.4. Test Organization

*   **Test Suites:** Organize tests into suites based on the component or module being tested.
*   **Test Cases:** Write clear and concise test cases with descriptive names.
*   **Arrange-Act-Assert:** Follow the Arrange-Act-Assert pattern in each test case.

### 5.5. Mocking and Stubbing

*   **Mock Dependencies:** Mock external dependencies like API services or third-party libraries to isolate the code being tested.
*   **Stub Component Behavior:** Stub the behavior of child components to focus on testing the parent component's logic.
*   **Use Mocking Libraries:** Use a mocking library like Jest's `jest.fn()` to create mock functions and objects.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

*   **Forgetting to Use `:key` in `v-for`:** Always use the `:key` attribute when rendering lists with `v-for` to ensure efficient DOM updates.
*   **Incorrectly Using `v-if` and `v-show`:** Understand the difference between `v-if` and `v-show` and use them appropriately.  `v-if` conditionally renders the element, while `v-show` toggles the element's visibility.
*   **Mutating Props Directly:** Avoid mutating props directly. Emit an event to the parent component to update the prop value.
*   **Not Handling Edge Cases:**  Consider edge cases and write tests to cover them.

### 6.2. Edge Cases to Be Aware Of

*   **Empty Arrays or Objects:**  Handle cases where data is empty or null.
*   **Unexpected API Responses:**  Handle cases where the API returns an error or unexpected data.
*   **User Input Errors:**  Handle cases where the user enters invalid or malicious input.

### 6.3. Version-Specific Issues

*   **Breaking Changes:** Be aware of breaking changes in new Vue.js versions and update the code accordingly.
*   **Deprecated APIs:** Avoid using deprecated APIs and migrate to the recommended alternatives.
*   **Compatibility Issues:** Ensure compatibility with the target browsers and devices.

### 6.4. Compatibility Concerns

*   **Browser Compatibility:** Test the application in different browsers and devices to ensure compatibility.
*   **Accessibility:** Ensure the application is accessible to users with disabilities.
*   **Responsive Design:** Implement responsive design to ensure the application looks good on different screen sizes.

### 6.5. Debugging Strategies

*   **Vue Devtools:** Use the Vue Devtools browser extension to inspect components, state, and events.
*   **Console Logging:** Use `console.log()` to debug code and track variables.
*   **Debugger Statements:** Use `debugger` statements to pause the execution of code and inspect variables.
*   **Error Logging:** Log errors to a server or error tracking service for monitoring and debugging.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **VS Code with Vetur or Volar:** Use VS Code with the Vetur (Vue tooling) or Volar extension for syntax highlighting, code completion, and other features.
*   **Vue CLI:** Use Vue CLI for scaffolding projects, building, and serving the application.
*   **Vue Devtools:** Use the Vue Devtools browser extension for debugging Vue applications.
*   **ESLint:** Use ESLint with the `eslint-plugin-vue` plugin for linting Vue code.
*   **Prettier:** Use Prettier for formatting Vue code.

### 7.2. Build Configuration

*   **Webpack or Rollup:** Use Webpack or Rollup for building the application.
*   **Babel:** Use Babel for transpiling JavaScript code to ensure compatibility with older browsers.
*   **PostCSS:** Use PostCSS for processing CSS code and adding vendor prefixes.

### 7.3. Linting and Formatting

*   **ESLint:** Configure ESLint with the `eslint-plugin-vue` plugin to enforce coding standards and prevent errors.
*   **Prettier:** Configure Prettier to automatically format code according to a consistent style.
*   **Husky and lint-staged:** Use Husky and lint-staged to run linters and formatters before committing code.

### 7.4. Deployment Best Practices

*   **Build for Production:** Build the application for production with the `--mode production` flag.
*   **Optimize Assets:** Optimize assets like images and fonts to reduce file sizes.
*   **Use a CDN:** Use a content delivery network (CDN) to serve static assets.
*   **Configure Caching:** Configure caching headers to improve performance.
*   **Use HTTPS:** Use HTTPS for all communication.

### 7.5. CI/CD Integration

*   **Automated Builds:** Configure a CI/CD pipeline to automatically build and deploy the application whenever changes are pushed to the repository.
*   **Automated Testing:** Run automated tests in the CI/CD pipeline to ensure code quality.
*   **Automated Deployment:** Automate the deployment process to reduce manual effort and prevent errors.


By following these best practices, you can create high-quality, maintainable, and performant Vue.js applications.