---
description: This rule provides comprehensive best practices and coding standards for Nuxt.js projects, covering code organization, performance, security, testing, and common pitfalls. It aims to ensure maintainable, scalable, and secure Nuxt.js applications.
globs: *.vue,*.js,*.ts,*.mjs,*.mts,*.jsx,*.tsx,*.config.js,*.config.ts
---
- **Enable ESLint support:** Use the `@nuxt/eslint` module for project-aware ESLint configuration. This ensures code quality and consistency.
  - `npx nuxi module add eslint` to add the module.
  - Customize the generated `eslint.config.mjs` file as needed.
- **Adopt Nuxt.js Modules:** Leverage Nuxt.js modules to encapsulate functionalities and maintain a clean codebase.  Explore existing modules before implementing custom solutions (e.g., `@nuxt/auth` for server-side authentication).
- **Convention over Configuration:** Adhere to Nuxt.js conventions to simplify development and collaboration.  Avoid deviating from conventions unless absolutely necessary.
- **Efficiently Utilize Nuxt Layouts:** Create reusable layouts for components shared across multiple pages to ensure consistency and save development time. Layouts are located in the `layouts/` directory.
- **Manage State with Pinia:** Use Pinia for state management. Organize store modules based on features or functionalities for better maintainability.
- **Divide Pages into Components:** Break down pages into small, reusable components to enhance maintainability, testability, and reusability. Each component should have a single responsibility.
- **Leverage Nuxt Plugins Wisely:** Use Nuxt plugins to run code before Vue.js initializes or add global functionality.  Be mindful of plugin performance impact. Plugins are located in the `plugins/` directory.
- **Optimize for SEO and Performance:** Utilize Nuxt.js's server-side rendering (SSR) for SEO.  Implement lazy loading for images and optimize assets to minimize initial load time.  Use tools like Lighthouse to identify performance bottlenecks.
- **Implement Error Handling and Validation:** Implement robust error handling and validation mechanisms to provide a seamless user experience. Use Nuxt.js middleware to intercept requests and responses for error handling and data validation.
- **Document Your Code:** Provide clear and concise documentation for components, modules, and custom functions using tools like JSDoc.
- **Embrace Testing:** Write unit tests, integration tests, and end-to-end tests using tools like Jest, Vue Test Utils and Vitest.

## 1. Code Organization and Structure:

- **Directory Structure:**
  - `components/`: Reusable Vue components.
  - `composables/`: Reusable composable functions.
  - `layouts/`: Application layouts.
  - `middleware/`: Route middleware.
  - `pages/`: Application pages (route definitions).
  - `plugins/`: Nuxt.js plugins.
  - `server/`: API routes and server-side logic.
  - `static/`: Static assets (e.g., images, fonts).
  - `store/`: Pinia stores (optional, but recommended).
  - `utils/`: Utility functions.
- **File Naming Conventions:**
  - Components: `PascalCase.vue` (e.g., `MyComponent.vue`)
  - Composables: `usePascalCase.js` or `usePascalCase.ts` (e.g., `useCounter.js`)
  - Layouts: `kebab-case.vue` (e.g., `default.vue` or `custom-layout.vue`)
  - Pages: `kebab-case.vue` (e.g., `index.vue`, `about.vue`, `product-details.vue`)
  - Plugins: `kebab-case.js` or `kebab-case.ts` (e.g., `analytics.js`)
  - Stores: `kebab-case.js` or `kebab-case.ts` (e.g., `user-store.js`)
  - Utility functions: `camelCase.js` or `camelCase.ts` (e.g., `formatDate.js`)
- **Module Organization:**
  - Group related functionalities into separate modules.
  - Use the `@nuxt/modules` array in `nuxt.config.js` or `nuxt.config.ts` to register modules.
  - Create custom modules to encapsulate complex logic.
- **Component Architecture:**
  - Favor composition over inheritance.
  - Use functional components for simple UI elements.
  - Design components for reusability and testability.
  - Consider using slots for flexible component composition.
- **Code Splitting:**
  - Utilize dynamic imports for route-based code splitting.
  - Split large components into smaller chunks using `import()`.
  - Analyze bundle size with tools like Webpack Bundle Analyzer.

## 2. Common Patterns and Anti-patterns:

- **Design Patterns:**
  - **Composition API:** Use the Composition API for organizing component logic.
  - **Store Pattern (Pinia):** Implement a centralized state management system with Pinia.
  - **Middleware Pattern:** Use middleware for authentication, authorization, and data validation.
  - **Plugin Pattern:** Create plugins for global functionality and third-party library integrations.
- **Recommended Approaches:**
  - **API Communication:** Use the `useFetch` or `useAsyncData` composables for API calls within components.
  - **Form Handling:** Utilize Vue's built-in form handling features with `v-model` and validation libraries like VeeValidate.
  - **Authentication:** Implement a secure authentication flow using a library like `@nuxt/auth` or a custom solution.
  - **Authorization:** Implement role-based access control (RBAC) using middleware and Pinia stores.
- **Anti-patterns:**
  - **Mutating props directly:** Avoid modifying parent component data directly from child components. Use `emit` instead.
  - **Overusing global state:** Limit the use of global state to essential application data.  Consider component-level state for local data.
  - **Ignoring error handling:** Always handle potential errors in API calls and other asynchronous operations.
  - **Writing overly complex components:** Break down large components into smaller, more manageable pieces.
- **State Management Best Practices:**
  - **Single Source of Truth:** Maintain a single, consistent source of truth for application state in Pinia stores.
  - **Immutability:** Treat state as immutable. Use functions to update the store rather than directly manipulating data.
  - **Clear Naming Conventions:** Use descriptive names for store modules, actions, and mutations.
  - **Modularization:** Divide stores into modules based on features or functionalities.
- **Error Handling Patterns:**
  - **Centralized Error Handling:** Implement a global error handler to catch unhandled exceptions.
  - **Error Boundaries:** Use error boundaries to isolate component failures and prevent cascading errors.
  - **User-Friendly Error Messages:** Provide clear and helpful error messages to users.

## 3. Performance Considerations:

- **Optimization Techniques:**
  - **Lazy Loading:** Implement lazy loading for images, components, and routes.
  - **Code Splitting:** Split the application into smaller chunks for faster initial load times.
  - **Tree Shaking:** Remove unused code during the build process.
  - **Caching:** Cache API responses and static assets to reduce server load.
  - **Image Optimization:** Optimize images using tools like `nuxt/image`. Use appropriate image formats (WebP). Resize the image to appropriate size. Consider using a CDN for image delivery.
- **Memory Management:**
  - **Avoid Memory Leaks:** Clean up event listeners and timers when components are unmounted.
  - **Use Weak References:** Use weak references for DOM elements when possible.
  - **Minimize Object Creation:** Avoid creating unnecessary objects and arrays.
- **Rendering Optimization:**
  - **Virtualization:** Use virtualization for large lists to improve rendering performance.
  - **Memoization:** Memoize expensive calculations to avoid redundant computations. Use `computed` properties effectively to avoid unnecessary re-renders.
  - **Debouncing and Throttling:** Use debouncing and throttling for event handlers to reduce the number of function calls.
- **Bundle Size Optimization:**
  - **Analyze Bundle Size:** Use Webpack Bundle Analyzer to identify large dependencies.
  - **Remove Unused Dependencies:** Remove unused dependencies to reduce bundle size.
  - **Use Smaller Alternatives:** Consider using smaller alternatives to large libraries.
  - **Optimize Dependencies:** Review dependencies and ensure you're using the most efficient ones.
- **Lazy Loading Strategies:**
  - **Route-based Lazy Loading:** Load components only when the corresponding route is accessed.
  - **Component-based Lazy Loading:** Load components only when they are visible in the viewport.

## 4. Security Best Practices:

- **Common Vulnerabilities:**
  - **Cross-Site Scripting (XSS):** Prevent XSS attacks by properly sanitizing user input and using Vue's built-in HTML escaping.
  - **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by implementing CSRF tokens.
  - **SQL Injection:** Avoid raw SQL queries. Use an ORM (Object-Relational Mapper) to prevent SQL injection.
  - **Authentication and Authorization Flaws:** Implement secure authentication and authorization mechanisms.
  - **Insecure Direct Object References (IDOR):** Implement proper access control to prevent unauthorized access to resources.
- **Input Validation:**
  - **Server-Side Validation:** Always validate user input on the server-side.
  - **Client-Side Validation:** Provide client-side validation for a better user experience (but don't rely on it as the sole source of validation).
  - **Sanitize Input:** Sanitize user input to remove potentially harmful characters.
- **Authentication and Authorization Patterns:**
  - **JWT (JSON Web Tokens):** Use JWT for authentication and authorization.
  - **OAuth 2.0:** Implement OAuth 2.0 for third-party authentication.
  - **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
- **Data Protection Strategies:**
  - **Encryption:** Encrypt sensitive data at rest and in transit.
  - **Hashing:** Hash passwords and other sensitive data using strong hashing algorithms.
  - **Data Masking:** Mask sensitive data in logs and other non-production environments.
- **Secure API Communication:**
  - **HTTPS:** Always use HTTPS for API communication.
  - **API Rate Limiting:** Implement API rate limiting to prevent abuse.
  - **Authentication and Authorization:** Require authentication and authorization for all API endpoints.

## 5. Testing Approaches:

- **Unit Testing:**
  - **Test Individual Components:** Test individual components in isolation.
  - **Mock Dependencies:** Mock external dependencies to isolate components during testing.
  - **Verify Component Behavior:** Verify that components render correctly and behave as expected.
- **Integration Testing:**
  - **Test Component Interactions:** Test the interactions between components.
  - **Test Data Flow:** Test the data flow between components and stores.
  - **Test API Integrations:** Test the integration with external APIs.
- **End-to-End Testing:**
  - **Simulate User Interactions:** Simulate user interactions to test the application's functionality.
  - **Test the Entire Application Flow:** Test the entire application flow from start to finish.
  - **Use a Browser Automation Tool:** Use a browser automation tool like Cypress or Playwright.
- **Test Organization:**
  - **Organize Tests by Feature:** Organize tests by feature or functionality.
  - **Use Descriptive Test Names:** Use descriptive test names to clearly indicate what each test is testing.
  - **Keep Tests Isolated:** Keep tests isolated from each other to avoid interference.
- **Mocking and Stubbing:**
  - **Use Mock Objects:** Use mock objects to replace external dependencies during testing.
  - **Use Stubs:** Use stubs to replace complex functions with simplified versions.
  - **Avoid Over-Mocking:** Avoid mocking too much code, as this can make tests less effective.

## 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes:**
  - **Incorrect `this` Context:** Be mindful of the `this` context in Vue components and use arrow functions or `bind` to maintain the correct context.
  - **Asynchronous Data Handling:** Properly handle asynchronous data loading with `async/await` or Promises.
  - **Forgetting to Unsubscribe:** Unsubscribe from event listeners and timers when components are unmounted to prevent memory leaks.
  - **Overusing `forceUpdate`:** Avoid using `forceUpdate` unless absolutely necessary, as it can negatively impact performance.
- **Edge Cases:**
  - **Server-Side Rendering (SSR):** Be aware of the differences between client-side and server-side rendering.
  - **Browser Compatibility:** Test the application in different browsers to ensure compatibility.
  - **Accessibility:** Consider accessibility when designing and developing the application.
- **Version-Specific Issues:**
  - **Nuxt 2 vs Nuxt 3:** Be aware of the differences between Nuxt 2 and Nuxt 3.
  - **Vue 2 vs Vue 3:** Be aware of the differences between Vue 2 and Vue 3.
  - **Dependency Updates:** Carefully review dependency updates for potential breaking changes.
- **Compatibility Concerns:**
  - **Browser Support:** Ensure compatibility with the target browsers.
  - **Device Compatibility:** Test the application on different devices.
  - **Operating System Compatibility:** Ensure compatibility with the target operating systems.
- **Debugging Strategies:**
  - **Use Browser Developer Tools:** Use browser developer tools to inspect the application's state and network activity.
  - **Use Vue Devtools:** Use Vue Devtools to inspect Vue components and data.
  - **Use Logging:** Use logging to track the application's behavior.

## 7. Tooling and Environment:

- **Recommended Development Tools:**
  - **VS Code:** Visual Studio Code is a popular code editor with excellent Vue.js support.
  - **Vue Devtools:** Vue Devtools is a browser extension that provides debugging tools for Vue.js applications.
  - **ESLint:** ESLint is a linter that enforces coding standards.
  - **Prettier:** Prettier is a code formatter that automatically formats code.
- **Build Configuration:**
  - **`nuxt.config.js` or `nuxt.config.ts`:** Configure the application's build settings in `nuxt.config.js` or `nuxt.config.ts`.
  - **Webpack:** Nuxt uses Webpack to bundle the application.
  - **Vite:** Nuxt 3 uses Vite to bundle the application by default, providing faster build and development times.
- **Linting and Formatting:**
  - **ESLint:** Use ESLint to enforce coding standards.
  - **Prettier:** Use Prettier to automatically format code.
  - **Husky:** Use Husky to run linters and formatters before commits.
- **Deployment Best Practices:**
  - **Server-Side Rendering (SSR):** Deploy the application to a server that supports SSR.
  - **Static Site Generation (SSG):** Generate a static site for content-heavy applications.
  - **CDN:** Use a CDN to deliver static assets.
- **CI/CD Integration:**
  - **Continuous Integration (CI):** Use a CI tool like Jenkins, GitLab CI, or GitHub Actions to automate the build and testing process.
  - **Continuous Deployment (CD):** Use a CD tool to automate the deployment process.

By following these best practices, you can build robust, maintainable, and scalable Nuxt.js applications.