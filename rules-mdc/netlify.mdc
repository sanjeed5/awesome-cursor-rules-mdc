---
description: This rule file outlines best practices for Netlify development, covering code structure, performance, security, testing, and deployment. It aims to provide a comprehensive guide for building robust and scalable applications on Netlify.
globs: *.js,*.jsx,*.ts,*.tsx,*.html,*.css,*.scss,*.md
---
# Netlify Library Best Practices and Coding Standards

This document outlines best practices for developing applications using Netlify, encompassing code organization, performance considerations, security measures, testing strategies, and deployment procedures. Adhering to these guidelines will ensure the creation of robust, scalable, and maintainable applications on the Netlify platform.

## 1. Code Organization and Structure

### 1.1 Directory Structure

A well-defined directory structure is crucial for maintainability and collaboration.  Here's a recommended structure:


project-root/
├── .netlify/          # Netlify-specific configuration (auto-generated)
├── functions/        # Serverless functions
│   └── api/          # API endpoint functions
│       └── hello.js
├── src/              # Source code
│   ├── components/   # Reusable UI components
│   │   ├── Button.jsx
│   │   └── Header.jsx
│   ├── pages/        # Pages for different routes
│   │   ├── index.jsx
│   │   └── about.jsx
│   ├── styles/       # CSS, SCSS, or other styling
│   │   ├── global.css
│   │   └── components/
│   ├── utils/        # Utility functions
│   │   └── api.js
│   ├── App.jsx       # Main application component
│   ├── index.jsx     # Entry point for React
│   └── routes.js     # Routing configuration
├── static/           # Static assets (images, fonts, etc.)
│   ├── img/
│   └── fonts/
├── public/           # Public files to be deployed
│   └── index.html
├── tests/            # Tests
│   ├── unit/
│   └── integration/
├── netlify.toml      # Netlify configuration file
├── package.json      # Node.js dependencies
└── README.md


### 1.2 File Naming Conventions

*   **Components:** Use PascalCase for component file names (e.g., `MyComponent.jsx`).
*   **Styles:** Use kebab-case for style file names (e.g., `my-component.css`).
*   **Functions:** Use camelCase or kebab-case for function file names (e.g., `helloWorld.js` or `hello-world.js`).
*   **Images:** Use descriptive names (e.g., `product-image.jpg`).

### 1.3 Module Organization

*   **Group Related Code:**  Place related functions, components, and styles into separate modules/directories.
*   **Single Responsibility Principle:** Each module should have a clear and single purpose.
*   **Avoid Circular Dependencies:**  Carefully manage dependencies between modules to prevent circular dependencies.
*   **Use ES Modules:** Utilize ES modules (`import/export`) for better code organization and tree shaking.

### 1.4 Component Architecture

*   **Component-Based Approach:** Build your UI using reusable components.
*   **Presentational and Container Components:** Separate concerns by using presentational components (UI only) and container components (data fetching and logic).
*   **Atomic Design:**  Consider using atomic design principles (atoms, molecules, organisms, templates, pages) for a scalable component architecture.

### 1.5 Code Splitting

*   **Dynamic Imports:**  Use dynamic imports (`import()`) to load modules on demand.
*   **Route-Based Splitting:** Split your application based on routes so that only the necessary code is loaded for a given route. This will improve the initial loading time for the end user.
*   **Component-Based Splitting:** Use React.lazy or similar mechanisms to load components only when they are needed.
*   **Webpack or Parcel Bundler Configuration:** Configure your bundler (Webpack, Parcel, etc.) for optimal code splitting.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **JAMstack Architecture:**  Embrace the JAMstack architecture (JavaScript, APIs, Markup) for faster and more secure websites.
*   **Serverless Functions:** Utilize Netlify Functions for backend logic (API endpoints, background tasks).  Follow the function handler structure and consider using middleware for common tasks like authentication.
*   **Git-Based Workflow:** Use Git for version control and continuous deployment with Netlify.
*   **Environment Variables:** Store sensitive information (API keys, database credentials) in environment variables, and access them from your serverless functions. Ensure you configure these in Netlify's UI and use `process.env.VARIABLE_NAME`.

### 2.2 Recommended Approaches

*   **Form Handling:** Use libraries like Formik or React Hook Form for form handling.
*   **State Management:**  Choose a state management solution appropriate for the complexity of your application (e.g., Context API, Redux, Zustand).
*   **API Calls:** Use `fetch` or libraries like Axios for making API calls.  Implement error handling and loading states.
*   **Image Optimization:** Optimize images before deployment to improve performance.  Consider using Netlify Large Media for automatic image optimization.

### 2.3 Anti-patterns

*   **Committing Secrets:** Avoid committing API keys or other secrets to your Git repository.
*   **Over-reliance on Client-Side Rendering:**  Minimize client-side rendering for better SEO and initial load time. Use static site generation (SSG) or server-side rendering (SSR) where appropriate.
*   **Ignoring Performance Budgets:**  Establish performance budgets and monitor your application's performance regularly. Use tools such as Lighthouse, WebPageTest, and Chrome DevTools.
*   **Unnecessary Dependencies:**  Avoid including unnecessary dependencies in your project.
*   **Direct DOM Manipulation:** Avoid direct DOM manipulation in React components; let React handle the updates.

### 2.4 State Management

*   **Context API:**  Use React's Context API for simple state management.
*   **Redux:**  Use Redux for more complex state management requirements.
*   **Zustand:** Consider Zustand for a simple and unopinionated state management solution.
*   **Avoid Global State:** Minimize the use of global state to avoid performance issues and make code easier to reason about.
*   **Immutable Updates:** Use immutable updates when updating state to improve performance and prevent unexpected side effects.

### 2.5 Error Handling

*   **Try-Catch Blocks:**  Use try-catch blocks in your serverless functions to handle errors gracefully.
*   **Centralized Error Handling:**  Implement a centralized error handling mechanism to log errors and display user-friendly messages.
*   **ErrorBoundary:**  Use React's `ErrorBoundary` component to catch errors in your UI.
*   **Logging:** Log errors to a service like Sentry, Rollbar, or Netlify's built-in logging for monitoring and debugging.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Code Splitting:**  Implement code splitting to reduce initial load time.
*   **Lazy Loading:** Lazy load images and other assets to improve performance.
*   **Image Optimization:** Optimize images using tools like ImageOptim, TinyPNG, or Netlify Large Media.
*   **Caching:**  Utilize browser caching and CDN caching to reduce server load and improve response times. Netlify automatically handles CDN caching.
*   **Minification:**  Minify your CSS and JavaScript files to reduce their size.
*   **Compression:** Use Gzip or Brotli compression to reduce the size of your files.

### 3.2 Memory Management

*   **Avoid Memory Leaks:**  Be careful to avoid memory leaks in your JavaScript code. Properly clean up event listeners and timers.
*   **Use Efficient Data Structures:** Use appropriate data structures for your data to optimize memory usage.
*   **Limit Large Data Sets:**  Avoid loading large datasets into memory if possible.

### 3.3 Rendering Optimization (if applicable)

*   **Virtual DOM:** React's Virtual DOM provides significant rendering optimization. Ensure you are using React efficiently, minimizing unnecessary re-renders.
*   **Memoization:** Use `React.memo` to prevent re-rendering components unless their props change.
*   **Pure Components:**  Use `PureComponent` or implement `shouldComponentUpdate` for class components to prevent unnecessary re-renders.

### 3.4 Bundle Size Optimization

*   **Analyze Bundle Size:** Use tools like Webpack Bundle Analyzer or Parcel Size Analyzer to identify large dependencies.
*   **Tree Shaking:**  Enable tree shaking in your bundler to remove unused code.
*   **Code Minimization:** Utilize code minification in your build process.
*   **Remove Unused Dependencies:** Review your dependencies and remove any unused ones.

### 3.5 Lazy Loading

*   **Intersection Observer:**  Use the Intersection Observer API to lazy load images and other assets when they come into view.
*   **React Lazy:** Use React.lazy for component-level lazy loading.
*   **Dynamic Imports:**  Use dynamic imports for modules that are not needed immediately.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user inputs and escaping output.
*   **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by using CSRF tokens.
*   **Injection Attacks:**  Prevent SQL injection and other injection attacks by using parameterized queries and validating user inputs.
*   **Authentication and Authorization:**  Implement secure authentication and authorization mechanisms to protect sensitive data.
*   **Exposed API Keys:** Avoid exposing API keys in client-side code. Store them in environment variables and access them from your serverless functions.

### 4.2 Input Validation

*   **Client-Side Validation:** Use client-side validation to provide immediate feedback to users.
*   **Server-Side Validation:**  Always perform server-side validation to ensure data integrity and prevent malicious inputs.
*   **Sanitization:** Sanitize user inputs to remove potentially harmful characters.
*   **Escaping:** Escape output to prevent XSS attacks.

### 4.3 Authentication and Authorization

*   **Netlify Identity:** Use Netlify Identity for simple authentication and authorization.
*   **Third-Party Authentication:**  Use third-party authentication providers like Auth0, Firebase Authentication, or AWS Cognito.
*   **JSON Web Tokens (JWT):** Use JWTs to securely transmit user information between the client and server.
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.

### 4.4 Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Secure Storage:** Store sensitive data in secure storage locations.
*   **Regular Backups:** Perform regular backups of your data.
*   **Data Minimization:** Only collect and store the data that is necessary.
*   **Compliance:** Ensure your application complies with relevant data protection regulations (e.g., GDPR, CCPA).

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **API Keys:** Use API keys to authenticate requests to your API.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your API.
*   **CORS:** Configure CORS (Cross-Origin Resource Sharing) to allow only authorized domains to access your API.

## 5. Testing Approaches

### 5.1 Unit Testing

*   **Test Individual Components:** Unit tests should focus on testing individual components in isolation.
*   **Mock Dependencies:** Mock dependencies to isolate the component being tested.
*   **Test All Code Paths:** Test all code paths, including error handling.
*   **Use a Testing Framework:** Use a testing framework like Jest, Mocha, or Jasmine.

### 5.2 Integration Testing

*   **Test Interactions:** Integration tests should focus on testing the interactions between different components and modules.
*   **Test API Calls:** Test the integration with your API endpoints.
*   **Use a Testing Framework:** Use a testing framework like Cypress, Puppeteer, or Selenium.

### 5.3 End-to-End Testing

*   **Test User Flows:** End-to-end tests should focus on testing complete user flows.
*   **Simulate User Interactions:** Simulate user interactions, such as clicking buttons and filling out forms.
*   **Use a Testing Framework:** Use a testing framework like Cypress, Puppeteer, or Selenium.

### 5.4 Test Organization

*   **Separate Test Files:** Create separate test files for each component or module.
*   **Organize Tests by Type:** Organize tests by type (unit, integration, end-to-end).
*   **Use Descriptive Test Names:** Use descriptive test names to make it easier to understand what each test is testing.

### 5.5 Mocking and Stubbing

*   **Mock API Calls:** Mock API calls to avoid making real API calls during testing.
*   **Stub Dependencies:** Stub dependencies to isolate the component being tested.
*   **Use a Mocking Library:** Use a mocking library like Jest's `jest.mock` or Sinon.js.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   **Not Using Environment Variables:**  Failing to use environment variables for sensitive information.
*   **Over-complicating Serverless Functions:** Creating serverless functions that are too large and complex.
*   **Not Handling Errors:**  Failing to handle errors gracefully in serverless functions.
*   **Ignoring Performance:**  Ignoring performance considerations and creating slow-loading applications.
*   **Not Testing:**  Failing to test your application thoroughly.

### 6.2 Edge Cases

*   **Long Build Times:**  Long build times can be a problem for large applications. Optimize your build process to reduce build times.
*   **Function Cold Starts:**  Serverless functions can experience cold starts, which can increase response times. Use techniques like keep-alive to reduce cold start times.
*   **CDN Invalidation:**  Ensure your CDN invalidates cache correctly when you deploy new versions of your application.

### 6.3 Version-Specific Issues

*   **Netlify CLI Updates:** Stay up-to-date with the latest Netlify CLI to avoid compatibility issues.
*   **Node.js Versions:**  Ensure your Node.js version is compatible with Netlify Functions.
*   **Dependency Updates:**  Regularly update your dependencies to address security vulnerabilities and bug fixes.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:**  Test your application in different browsers to ensure compatibility.
*   **Device Compatibility:**  Test your application on different devices (desktop, mobile, tablet) to ensure compatibility.
*   **Accessibility:** Ensure your application is accessible to users with disabilities by following accessibility guidelines (WCAG).

### 6.5 Debugging Strategies

*   **Netlify Logs:** Use Netlify's built-in logging to debug issues.
*   **Console Logging:**  Use `console.log` statements to debug your code.
*   **Debugging Tools:** Use browser debugging tools like Chrome DevTools or Firefox Developer Tools.
*   **Remote Debugging:**  Use remote debugging tools to debug serverless functions.

## 7. Tooling and Environment

### 7.1 Recommended Tools

*   **VS Code:** VS Code with extensions like ESLint, Prettier, and Netlify.
*   **Netlify CLI:** Netlify Command Line Interface for local development and deployment.
*   **Git:** Git for version control.
*   **npm or Yarn:** npm or Yarn for package management.
*   **Webpack, Parcel, or Rollup:**  Webpack, Parcel, or Rollup for bundling your code.

### 7.2 Build Configuration

*   **Netlify TOML:** Use the `netlify.toml` file to configure your build process. Set environment variables, build commands, and deploy contexts.
*   **Build Commands:**  Define build commands to transpile your code, optimize assets, and generate static files.
*   **Deploy Contexts:** Use deploy contexts to create different environments (e.g., production, staging, development).

### 7.3 Linting and Formatting

*   **ESLint:** Use ESLint to enforce code style and prevent errors.
*   **Prettier:**  Use Prettier to automatically format your code.
*   **Husky and lint-staged:** Use Husky and lint-staged to run linters and formatters before committing code.

### 7.4 Deployment

*   **Git-Based Deployment:**  Deploy your application by pushing your code to a Git repository connected to Netlify.
*   **Netlify CLI Deployment:**  Deploy your application using the Netlify CLI.
*   **Atomic Deploys:** Netlify provides atomic deploys, ensuring that your site is always in a consistent state during updates.
*   **Rollbacks:** Easily roll back to previous deployments if necessary.

### 7.5 CI/CD Integration

*   **GitHub Actions:**  Integrate with GitHub Actions for automated testing and deployment.
*   **CircleCI:** Integrate with CircleCI for continuous integration and continuous deployment.
*   **Travis CI:** Integrate with Travis CI for continuous integration and continuous deployment.

By following these best practices, you can create robust, scalable, secure, and maintainable applications using Netlify.