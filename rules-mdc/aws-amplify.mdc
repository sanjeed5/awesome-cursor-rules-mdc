---
description: This rule provides best practices and coding standards for aws-amplify projects, covering code organization, common patterns, performance, security, testing, pitfalls, and tooling. It aims to guide developers in building robust, scalable, and maintainable aws-amplify applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
- # aws-amplify Best Practices

  This document outlines best practices and coding standards for developing applications using the aws-amplify library. It aims to guide developers in creating robust, scalable, and maintainable applications.

- ## 1. Code Organization and Structure

  - ### Directory Structure

    - **src/**: Contains all the application source code.
      - **components/**: Reusable UI components.
        - `ComponentName/`: Each component in its own directory.
          - `ComponentName.tsx`: The component file.
          - `ComponentName.module.css`: Component-specific styles (using CSS Modules).
          - `ComponentName.test.tsx`: Unit tests for the component.
          - `index.ts`: (Optional) Exports the component for easier imports.
      - **pages/**: Defines the application's routes/pages (especially important for Next.js and similar frameworks).
        - `index.tsx`: The home page.
        - `[routeName].tsx`: Dynamic routes.
      - **services/**: Contains business logic, data fetching, and interactions with aws-amplify services.
        - `authService.ts`: Authentication-related functions.
        - `dataService.ts`: Data fetching and manipulation (e.g., interacting with AppSync).
        - `storageService.ts`: Storage-related functions (e.g., interacting with S3).
      - **models/**: Defines data models used in the application.  This often corresponds to your Amplify DataStore or GraphQL schema definitions.
        - `Todo.ts`: Example data model for a `Todo` item.
      - **utils/**: Utility functions and helpers.
        - `apiClient.ts`: A wrapper around the `Amplify` object for making API requests.
        - `dateUtils.ts`: Date formatting functions.
      - **config/**: Application configuration files.
        - `aws-exports.js`: Automatically generated aws-amplify configuration.
      - **hooks/**: Custom React hooks for reusable logic.
        - `useAuth.ts`: A hook for managing authentication state.
    - **amplify/**:  Generated directory by Amplify CLI, containing backend definitions.
      - **backend/**: Code and configurations that define the AWS resources.
        -  `api/`:  GraphQL API definitions (schema.graphql, resolvers, etc.).
        -  `auth/`:  Authentication configuration (Cognito User Pool, Identity Pool).
        -  `storage/`:  Storage configurations (S3 bucket).
        -  `function/`:  Lambda functions.

  - ### File Naming Conventions

    - Components: `ComponentName.tsx`
    - Styles: `ComponentName.module.css`
    - Services: `serviceName.ts`
    - Models: `ModelName.ts`
    - Hooks: `useHookName.ts`
    - Tests: `FileName.test.tsx` or `FileName.spec.tsx`

  - ### Module Organization

    - Group related functionalities into modules.
    - Use clear and descriptive names for modules.
    - Avoid circular dependencies between modules.
    - Export only necessary functions and variables from each module.

  - ### Component Architecture

    - **Presentational Components:** Responsible for how things look.  Receive data and callbacks via props.
    - **Container Components:** Concerned with how things work.  Fetch data, manage state, and pass data/callbacks to presentational components.
    - Use composition over inheritance.
    - Keep components small and focused on a single responsibility.
    - Use React Hooks for managing state and side effects.

  - ### Code Splitting

    - Use dynamic imports (`React.lazy`) for components that are not immediately needed.
    - Split routes into separate bundles.
    - Leverage webpack, Parcel, or other bundlers to automatically split code.

- ## 2. Common Patterns and Anti-patterns

  - ### Design Patterns

    - **Provider Pattern:** Use React Context to provide aws-amplify client and authentication state to the entire application.
    - **Hook Composition:** Create custom hooks that encapsulate aws-amplify logic for reusability.
    - **Repository Pattern:** Abstract data access logic behind a repository interface.

  - ### Recommended Approaches

    - **Authentication:** Use `Amplify.configure` to initialize Amplify with the configuration from `aws-exports.js`.
      - Implement sign-in, sign-up, sign-out using `Auth` API.
      - Utilize Higher Order Components or custom Hooks to protect routes based on authentication status.
    - **Data Fetching:** Use `DataStore` or GraphQL APIs for data fetching.  Consider using pagination for large datasets.
    - **Storage:** Use `Storage` API for uploading and retrieving files from S3.

  - ### Anti-patterns and Code Smells

    - **Tight Coupling:** Avoid directly using `Amplify` APIs in components. Abstract them into services or hooks.
    - **Over-fetching:**  Fetch only the necessary data from APIs. Use GraphQL fragments to specify required fields.
    - **Ignoring Errors:**  Always handle errors when interacting with aws-amplify services.
    - **Storing Sensitive Data in Client:** Never store API keys or secrets in the client-side code. Use Lambda functions with environment variables.
    - **Over-reliance on Amplify UI Components:** While useful for rapid prototyping, customize or replace Amplify UI components for complex designs and accessibility requirements.

  - ### State Management

    - **Component State:**  Use `useState` or `useReducer` for local component state.
    - **Context API:**  Use React Context for application-wide state (e.g., authentication status, user data).
    - **Redux/Zustand/Recoil:**  Consider using a state management library for complex applications with global state dependencies.
    - **Amplify DataStore:** Excellent solution for offline-first applications needing real-time data synchronization with the cloud.  Use its built-in mechanisms for local persistence and conflict resolution.

  - ### Error Handling

    - Use `try...catch` blocks to handle errors when calling aws-amplify APIs.
    - Display user-friendly error messages.
    - Log errors for debugging purposes.
    - Implement retry mechanisms for transient errors.

- ## 3. Performance Considerations

  - ### Optimization Techniques

    - **Caching:** Cache API responses to reduce network requests. Use `Cache` API for storing data in the browser.
    - **Lazy Loading:** Load components and data only when needed.
    - **Debouncing and Throttling:**  Limit the rate at which functions are executed (e.g., event handlers).
    - **Memoization:**  Cache the results of expensive function calls using `useMemo` or `React.memo`.
    - **Image Optimization:** Optimize images before uploading them to S3. Use responsive images with different sizes for different devices.

  - ### Memory Management

    - Avoid memory leaks by cleaning up subscriptions and timers in `useEffect` hooks.
    - Release unused resources.

  - ### Rendering Optimization

    - Use `React.memo` to prevent unnecessary re-renders of components.
    - Implement shouldComponentUpdate (if using class components) for fine-grained control over rendering.
    - Virtualize long lists to render only visible items.

  - ### Bundle Size Optimization

    - Use code splitting to reduce the initial bundle size.
    - Remove unused code and dependencies.
    - Minify and compress code.
    - Analyze bundle size with tools like `webpack-bundle-analyzer` or `Source Map Explorer`.

  - ### Lazy Loading

    - Use `React.lazy` and `Suspense` for lazy-loading components.
    - Implement lazy loading for images and other assets.

- ## 4. Security Best Practices

  - ### Common Vulnerabilities

    - **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection mechanisms.
    - **Injection Attacks:** Protect against SQL injection, NoSQL injection, and command injection attacks.
    - **Authentication and Authorization Flaws:** Secure authentication and authorization mechanisms.

  - ### Input Validation

    - Validate all user input on both the client and server sides.
    - Use strong validation rules to prevent invalid data from being stored in the database.

  - ### Authentication and Authorization

    - Use Amazon Cognito for authentication and authorization.
    - Implement role-based access control (RBAC) to restrict access to resources.
    - Secure API endpoints with authentication and authorization checks.
    - Follow the principle of least privilege when granting permissions.

  - ### Data Protection

    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all communication.
    - Implement data masking and redaction techniques to protect sensitive data from unauthorized access.
    - Regularly back up data.

  - ### Secure API Communication

    - Use HTTPS for all API requests.
    - Validate API responses.
    - Implement rate limiting to prevent abuse.

- ## 5. Testing Approaches

  - ### Unit Testing

    - Test individual components and functions in isolation.
    - Use mocking and stubbing to isolate components from external dependencies.
    - Write tests that cover all possible scenarios and edge cases.
    - Use testing frameworks like Jest and React Testing Library.

  - ### Integration Testing

    - Test the interaction between different components and modules.
    - Test the integration with aws-amplify services.
    - Use integration testing frameworks like Cypress or Playwright.

  - ### End-to-End Testing

    - Test the entire application flow from start to finish.
    - Simulate real user interactions.
    - Use end-to-end testing frameworks like Cypress or Selenium.

  - ### Test Organization

    - Keep tests close to the code they are testing.
    - Use descriptive names for tests.
    - Organize tests into logical groups.

  - ### Mocking and Stubbing

    - Use mocking libraries like Jest to mock aws-amplify APIs and services.
    - Use stubbing to replace complex dependencies with simple implementations for testing purposes.

- ## 6. Common Pitfalls and Gotchas

  - ### Frequent Mistakes

    - Misconfiguring aws-amplify.
    - Ignoring error messages.
    - Not handling edge cases.
    - Over-complicating code.
    - Not testing thoroughly.

  - ### Edge Cases

    - Handling offline scenarios.
    - Dealing with slow network connections.
    - Handling unexpected API responses.

  - ### Version-Specific Issues

    - Be aware of breaking changes between aws-amplify versions.
    - Consult the aws-amplify documentation and release notes for migration guides.

  - ### Compatibility Concerns

    - Ensure compatibility between aws-amplify and other libraries and frameworks used in the project.
    - Check for known compatibility issues and workarounds.

  - ### Debugging Strategies

    - Use browser developer tools for debugging client-side code.
    - Use CloudWatch logs for debugging server-side code (Lambda functions).
    - Use the aws-amplify CLI to diagnose configuration issues.

- ## 7. Tooling and Environment

  - ### Recommended Tools

    - Visual Studio Code (VS Code) or other preferred IDE
    - aws-amplify CLI
    - Node.js and npm or yarn
    - Git for version control
    - Jest and React Testing Library for unit testing
    - Cypress or Playwright for end-to-end testing

  - ### Build Configuration

    - Use a build tool like webpack, Parcel, or esbuild.
    - Configure the build tool to optimize code for production.
    - Use environment variables for configuration settings.

  - ### Linting and Formatting

    - Use ESLint and Prettier to enforce code style and prevent errors.
    - Configure ESLint and Prettier to automatically format code on save.

  - ### Deployment

    - Use aws-amplify Console for deploying the application.
    - Configure CI/CD pipelines to automatically deploy changes to production.
    - Use environment variables for configuration settings.

  - ### CI/CD Integration

    - Integrate with CI/CD tools like GitHub Actions, CircleCI, or Travis CI.
    - Automate the build, test, and deployment processes.
    - Use environment variables for configuration settings.

- **Code-First Approach with TypeScript:** Embrace the code-first approach with AWS Amplify Gen 2, using TypeScript for both frontend and backend development to improve developer productivity and catch errors early.  Leverage AWS CDK for defining infrastructure as code.

- **Single Responsibility Principle:** Adhere to the Single Responsibility Principle for AWS Lambda functions to improve code maintainability and testability.

- **Avoid Function Chaining:** Reduce complexity and improve debugging by avoiding function chaining in Lambda functions.  If code reuse is needed, utilize separate TypeScript files for shared logic.

- **Start with Templates:** Accelerate development by using pre-built templates instead of starting projects from scratch.

- **Utilize Cloud Sandboxes:** Leverage per-developer cloud sandbox environments for isolated testing and iteration without interfering with other team members.

- **Take advantage of Amplify's opinionated nature** When you read the docs, you will start to understand how opinionated Amplify is. Whether you choose Amplify or not should include this aspect of its design, and you shouldn't choose this tool if it doesn't align with the direction your team is taking with infrastructure. It is best used as 'Glue' for an All-AWS roadmap.