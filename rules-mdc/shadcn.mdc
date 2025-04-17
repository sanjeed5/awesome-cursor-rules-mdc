---
description: This rule provides comprehensive best practices for developing with Shadcn UI, covering code organization, performance, security, and testing.
globs: *.js,*.jsx,*.ts,*.tsx
---
# Shadcn UI Best Practices

This document outlines best practices for developing with Shadcn UI, covering code organization, common patterns, performance considerations, security, testing, common pitfalls, and tooling.

## 1. Code Organization and Structure

- **Directory Structure:**
    - Organize components into logical directories based on functionality or domain. For example, place form-related components in a `components/forms` directory.
    - Separate components into their own files.  Each component should have a dedicated file named after the component (e.g., `Button.tsx`).
    - Consider using an `index.ts` file within each directory to export all components from that directory, simplifying imports.
    - Structure directories to reflect the UI hierarchy.  For example, `/components/layout` for layout related components and `/components/ui` for reusable UI elements.

- **File Naming Conventions:**
    - Use PascalCase for component file names (e.g., `MyComponent.tsx`).
    - Use camelCase for variable and function names (e.g., `handleClick`).
    - Use descriptive names that clearly indicate the purpose of the component or function.

- **Module Organization:**
    - Break down complex components into smaller, reusable modules.
    - Keep components focused on a single responsibility.
    - Utilize shared utility functions and constants to avoid code duplication.  Create a `utils` directory for helper functions.
    - Use `components` directory to store UI components.

- **Component Architecture:**
    - Favor composition over inheritance. Create flexible components that can be customized through props.
    - Design components with clear separation of concerns: presentational components (UI) and container components (logic).
    - Use functional components with hooks for managing state and side effects.

- **Code Splitting Strategies:**
    - Implement lazy loading for non-critical components to improve initial load time.
    - Utilize React.lazy and Suspense for code splitting at the component level.
    - Configure your bundler (e.g., Webpack, Parcel) to automatically split code into smaller chunks.
    - Consider route-based code splitting for larger applications.

## 2. Common Patterns and Anti-patterns

- **Design Patterns Specific to Shadcn UI:**
    - Leverage the existing components provided by Shadcn UI whenever possible.
    - Customize components using styling solutions like Tailwind CSS's utility classes or CSS variables.
    - Create compound components by combining existing Shadcn UI components to build more complex UI elements.

- **Recommended Approaches for Common Tasks:**
    - Use Shadcn UI's form components (e.g., `Input`, `Select`) for handling user input.
    - Implement accessible components by following ARIA guidelines and using appropriate HTML semantics.
    - Use the `cn` utility (classnames library) provided by Shadcn UI to manage CSS class names effectively.

- **Anti-patterns and Code Smells to Avoid:**
    - Directly modifying the Shadcn UI component code.
    - Overusing custom CSS, as Shadcn UI is built with Tailwind CSS.
    - Neglecting accessibility considerations.
    - Creating overly complex components with too many responsibilities.

- **State Management Best Practices:**
    - Use React's built-in `useState` hook for simple component-level state.
    - Consider using a state management library like Zustand, Redux, or Recoil for more complex application state.
    - Avoid mutating state directly; always use the setState function or a state management library's update methods.

- **Error Handling Patterns:**
    - Implement error boundaries to catch errors in components and prevent the entire application from crashing.
    - Use try-catch blocks to handle errors in asynchronous operations and API calls.
    - Provide informative error messages to users.
    - Log errors to a monitoring service for debugging and analysis.

## 3. Performance Considerations

- **Optimization Techniques:**
    - Minimize re-renders by using `React.memo` for functional components and `shouldComponentUpdate` for class components.
    - Optimize event handlers by using useCallback to prevent unnecessary re-creation of functions.
    - Debounce or throttle expensive operations to reduce the frequency of execution.

- **Memory Management:**
    - Avoid memory leaks by properly cleaning up event listeners and timers in the `useEffect` hook.
    - Release unused resources, such as large data structures, when they are no longer needed.

- **Rendering Optimization:**
    - Use virtualized lists or grids for rendering large datasets.
    - Batch DOM updates to minimize reflows and repaints.
    - Use CSS containment to isolate rendering changes to specific parts of the DOM.

- **Bundle Size Optimization:**
    - Remove unused code and dependencies using tree shaking.
    - Minify JavaScript and CSS files to reduce their size.
    - Compress images using tools like ImageOptim or TinyPNG.

- **Lazy Loading Strategies:**
    - Implement lazy loading for images and other media assets.
    - Use the Intersection Observer API to detect when elements are visible in the viewport and load them on demand.

## 4. Security Best Practices

- **Common Vulnerabilities and How to Prevent Them:**
    - Prevent cross-site scripting (XSS) attacks by sanitizing user input and escaping HTML entities.
    - Protect against cross-site request forgery (CSRF) attacks by using anti-CSRF tokens.
    - Avoid storing sensitive information, such as API keys or passwords, in client-side code.

- **Input Validation:**
    - Validate user input on both the client-side and server-side.
    - Use a validation library like Zod or Yup to define data schemas and enforce validation rules.
    - Sanitize user input to remove potentially harmful characters or code.

- **Authentication and Authorization Patterns:**
    - Use a secure authentication protocol, such as OAuth 2.0 or OpenID Connect.
    - Implement role-based access control (RBAC) to restrict access to sensitive resources.
    - Store user credentials securely using hashing and salting.

- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to protect data transmitted between the client and server.
    - Implement data masking to hide sensitive information from unauthorized users.

- **Secure API Communication:**
    - Use HTTPS for all API requests.
    - Implement rate limiting to prevent abuse and denial-of-service attacks.
    - Validate API responses to ensure data integrity.

## 5. Testing Approaches

- **Unit Testing Strategies:**
    - Write unit tests for individual components and functions.
    - Use a testing framework like Jest or Mocha.
    - Test component behavior with different props and inputs.

- **Integration Testing:**
    - Write integration tests to verify that components work together correctly.
    - Test the interaction between components and APIs.

- **End-to-End Testing:**
    - Write end-to-end tests to simulate user interactions and verify that the application functions as expected.
    - Use a testing framework like Cypress or Playwright.

- **Test Organization:**
    - Organize tests into separate files based on the component or feature being tested.
    - Use descriptive test names that clearly indicate the purpose of the test.

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components and functions during testing.
    - Mock external dependencies, such as APIs or third-party libraries.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes Developers Make:**
    - Forgetting to handle edge cases.
    - Overcomplicating components.
    - Neglecting accessibility.
    - Ignoring performance considerations.

- **Edge Cases to Be Aware Of:**
    - Handling different screen sizes and devices.
    - Dealing with slow network connections.
    - Handling invalid or unexpected user input.

- **Version-Specific Issues:**
    - Be aware of breaking changes between Shadcn UI versions.
    - Consult the Shadcn UI changelog for migration instructions.

- **Compatibility Concerns:**
    - Ensure that your application is compatible with the target browsers and devices.
    - Test your application on different browsers and devices.

- **Debugging Strategies:**
    - Use browser developer tools to inspect the DOM and debug JavaScript code.
    - Use console logging to track the flow of execution and identify errors.
    - Use a debugger to step through code and inspect variables.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - Visual Studio Code (VS Code) with extensions for React, TypeScript, and Tailwind CSS.
    - A browser with developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
    - A terminal for running commands and scripts.

- **Build Configuration:**
    - Use a build tool like Webpack, Parcel, or Rollup to bundle your application.
    - Configure your build tool to optimize code for production.

- **Linting and Formatting:**
    - Use ESLint to enforce code style and identify potential errors.
    - Use Prettier to automatically format code.
    - Configure your editor to automatically lint and format code on save.

- **Deployment Best Practices:**
    - Deploy your application to a reliable hosting provider.
    - Use a content delivery network (CDN) to serve static assets.
    - Configure your server to serve compressed files.

- **CI/CD Integration:**
    - Use a continuous integration and continuous deployment (CI/CD) pipeline to automate the build, test, and deployment process.
    - Integrate your CI/CD pipeline with your version control system.