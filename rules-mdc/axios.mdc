---
description: This rule provides comprehensive guidelines for using Axios effectively, covering best practices, security considerations, and performance optimization. It aims to improve code quality, maintainability, and overall application resilience when working with Axios.
globs: *.js,*.jsx,*.ts,*.tsx
---
- **Avoid hardcoding URLs**: Store API endpoints in configuration files or environment variables to enhance maintainability and avoid redundancy.
- **Centralize API Logic**: Create dedicated modules or services to encapsulate Axios instances and request logic. This promotes reusability and simplifies debugging.
- **Handle Errors Gracefully**: Implement comprehensive error handling using try-catch blocks, response interceptors, and centralized error logging. Display user-friendly error messages.
- **Use Asynchronous Functions**: Leverage `async/await` or Promises for cleaner and more manageable asynchronous code when making Axios requests.
- **Optimize with Request Interceptors**: Utilize request interceptors for tasks like adding authentication tokens, logging requests, or transforming request data. Response interceptors can be used for error handling or data transformation.
- **Ensure HTTP Requests When Necessary**: Verify that requests are only made when truly needed, preventing unnecessary network traffic and improving performance. Consider caching strategies.

### 1. Code Organization and Structure:

   - **Directory Structure Best Practices**:
     - Create an `api` or `services` directory to house all API-related code.
     - Organize API modules based on resources or features (e.g., `api/users.js`, `api/products.js`).
     - Separate Axios instance configuration into a dedicated file (e.g., `api/axios-config.js`).

   - **File Naming Conventions**:
     - Use descriptive names for API modules (e.g., `userApi.js`, `productService.js`).
     - Follow a consistent naming convention (e.g., camelCase for variables, PascalCase for components).

   - **Module Organization**:
     - Export individual API functions from each module (e.g., `getUser`, `createUser`).
     - Use named exports for better code readability and maintainability.

   - **Component Architecture**:
     - Create reusable components for handling data fetching and display.
     - Decouple components from specific API calls to improve reusability.
     - Utilize custom hooks to encapsulate data fetching logic.

   - **Code Splitting Strategies**:
     - Implement lazy loading for API modules that are not immediately required.
     - Use dynamic imports to load modules on demand.
     - Consider route-based code splitting for larger applications.

### 2. Common Patterns and Anti-patterns:

   - **Design Patterns Specific to Axios**:
     - **Singleton Pattern**: Use a singleton pattern for the Axios instance to ensure a single configuration across the application.
     - **Adapter Pattern**: Create an adapter layer to transform API responses into a consistent format.
     - **Factory Pattern**: Use a factory pattern to create different Axios instances with specific configurations.

   - **Recommended Approaches for Common Tasks**:
     - **Authentication**: Use request interceptors to add authentication headers (e.g., JWT tokens).
     - **Error Handling**: Implement a centralized error handling function to log errors and display user-friendly messages.
     - **Data Transformation**: Use response interceptors to transform data before it reaches the components.

   - **Anti-patterns and Code Smells to Avoid**:
     - **Hardcoding URLs**: Avoid hardcoding API endpoints directly in the code.
     - **Duplicated Request Logic**: Don't repeat API request logic across multiple components.
     - **Ignoring Errors**: Never ignore errors returned by Axios; always handle them appropriately.

   - **State Management Best Practices**:
     - Use a state management library (e.g., Redux, Zustand, Recoil) to manage API data.
     - Store API responses in a normalized format to improve performance.
     - Utilize selectors to derive data from the store efficiently.

   - **Error Handling Patterns**:
     - Implement a global error boundary to catch unexpected errors.
     - Display user-friendly error messages based on the error type.
     - Log errors to a central logging service for monitoring and debugging.

### 3. Performance Considerations:

   - **Optimization Techniques**:
     - **Caching**: Implement caching mechanisms to avoid redundant API calls.
     - **Request Debouncing**: Debounce frequent API requests to reduce server load.
     - **Request Cancellation**: Cancel pending requests when the component unmounts or the user navigates away.

   - **Memory Management**:
     - Clear Axios instances when they are no longer needed.
     - Avoid creating unnecessary Axios instances.

   - **Rendering Optimization**:
     - Use memoization techniques to prevent unnecessary re-renders.
     - Implement pagination or virtualized lists for large datasets.

   - **Bundle Size Optimization**:
     - Use tree shaking to remove unused code from the Axios library.
     - Minify and compress the JavaScript bundle.

   - **Lazy Loading Strategies**:
     - Lazy load API modules that are not immediately required.
     - Use dynamic imports to load modules on demand.

### 4. Security Best Practices:

   - **Common Vulnerabilities and How to Prevent Them**:
     - **Cross-Site Scripting (XSS)**: Sanitize user input to prevent XSS attacks.
     - **Cross-Site Request Forgery (CSRF)**: Implement CSRF protection mechanisms.
     - **Man-in-the-Middle (MITM) Attacks**: Use HTTPS to encrypt communication between the client and the server.

   - **Input Validation**:
     - Validate user input on both the client and server sides.
     - Use appropriate validation libraries to prevent injection attacks.

   - **Authentication and Authorization Patterns**:
     - Use a secure authentication protocol (e.g., OAuth 2.0, OpenID Connect).
     - Implement role-based access control (RBAC) to restrict access to sensitive data.

   - **Data Protection Strategies**:
     - Encrypt sensitive data at rest and in transit.
     - Use appropriate data masking techniques to protect sensitive information.

   - **Secure API Communication**:
     - Use HTTPS for all API communication.
     - Implement rate limiting to prevent abuse.
     - Monitor API traffic for suspicious activity.

### 5. Testing Approaches:

   - **Unit Testing Strategies**:
     - Mock Axios requests using libraries like `axios-mock-adapter` or `nock`.
     - Test individual API functions in isolation.
     - Verify that requests are made with the correct parameters and headers.

   - **Integration Testing**:
     - Test the interaction between API modules and components.
     - Verify that data is fetched and displayed correctly.
     - Use a testing framework like Jest or Mocha.

   - **End-to-End Testing**:
     - Test the entire application workflow from the user's perspective.
     - Use a testing framework like Cypress or Selenium.

   - **Test Organization**:
     - Organize tests by module or feature.
     - Use descriptive names for test cases.
     - Follow a consistent testing style.

   - **Mocking and Stubbing**:
     - Use mocking to isolate components from external dependencies.
     - Use stubbing to replace API responses with predefined data.

### 6. Common Pitfalls and Gotchas:

   - **Frequent Mistakes Developers Make**:
     - Forgetting to handle errors.
     - Using incorrect HTTP methods.
     - Hardcoding URLs.
     - Not setting appropriate headers.

   - **Edge Cases to Be Aware Of**:
     - Network errors.
     - Server downtime.
     - Rate limiting.
     - API versioning.

   - **Version-Specific Issues**:
     - Be aware of breaking changes between Axios versions.
     - Consult the Axios documentation for specific version information.

   - **Compatibility Concerns**:
     - Ensure that Axios is compatible with the target browsers and environments.
     - Use polyfills if necessary.

   - **Debugging Strategies**:
     - Use the browser's developer tools to inspect network requests.
     - Log Axios requests and responses to the console.
     - Use a debugging tool like VS Code's debugger.

### 7. Tooling and Environment:

   - **Recommended Development Tools**:
     - VS Code with the ESLint and Prettier extensions.
     - Axios DevTools for Chrome and Firefox.
     - Postman or Insomnia for testing API endpoints.

   - **Build Configuration**:
     - Use a build tool like Webpack or Parcel to bundle the application.
     - Configure the build tool to optimize the bundle size.

   - **Linting and Formatting**:
     - Use ESLint with a consistent set of rules to enforce code quality.
     - Use Prettier to format the code automatically.

   - **Deployment Best Practices**:
     - Deploy the application to a secure hosting environment.
     - Configure HTTPS for all API communication.
     - Monitor the application for errors and performance issues.

   - **CI/CD Integration**:
     - Integrate automated testing and deployment into the CI/CD pipeline.
     - Use a CI/CD tool like Jenkins, GitLab CI, or CircleCI.