---
description: This rule provides comprehensive best practices for using Apollo Client in your projects, covering code organization, performance, security, testing, and common pitfalls. It aims to guide developers in building robust and maintainable GraphQL-powered applications.
globs: *.js,*.jsx,*.ts,*.tsx,*.gql,*.graphql
---
- **Code Organization and Structure:**
  - **Directory Structure:**
    - Organize GraphQL queries and mutations in a dedicated directory (e.g., `graphql` or `queries`).
    - Group related queries and mutations into subdirectories based on features or modules.
    - Consider a separate directory for GraphQL types/schemas if you're managing them locally.
    - Example:
      
      src/
        graphql/
          user/
            getUser.gql
            createUser.gql
          product/
            getProduct.gql
            updateProduct.gql
      
  - **File Naming Conventions:**
    - Use descriptive names for GraphQL query and mutation files (e.g., `getUser.gql`, `updateProduct.gql`).
    - Employ a consistent naming scheme (e.g., `[operationName].[operationType].gql`).
  - **Module Organization:**
    - Encapsulate Apollo Client logic (e.g., initialization, hooks) within reusable modules.
    - Create custom hooks for common data fetching patterns (e.g., `useUser`, `useProduct`).
  - **Component Architecture:**
    - Favor a component-based architecture for UI development.
    - Decouple data fetching logic from UI components using hooks or render props.
    - Utilize higher-order components (HOCs) or render props for cross-cutting concerns (e.g., authentication, error handling).
  - **Code Splitting:**
    - Use dynamic imports to lazy-load components and queries, improving initial load time.
    - Split large queries into smaller fragments to reduce bundle size.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Repository Pattern:** Abstract data access logic behind a repository interface.
    - **Facade Pattern:** Provide a simplified interface to a complex subsystem (e.g., Apollo Client).
    - **Hook Composition:** Combine multiple custom hooks to create more complex data fetching logic.
  - **Recommended Approaches:**
    - Use the `useQuery` and `useMutation` hooks for data fetching and manipulation.
    - Leverage Apollo Client's cache for improved performance and offline capabilities.
    - Implement optimistic updates to provide a better user experience.
    - Use GraphQL fragments to reuse common data structures across multiple queries and mutations.
  - **Anti-patterns:**
    - Avoid directly manipulating the Apollo Client cache outside of mutations.
    - Don't fetch the entire schema unnecessarily; only fetch what you need.
    - Avoid deeply nested components with complex data fetching logic.
  - **State Management:**
    - Use Apollo Client's cache as the primary source of truth for application state.
    - Integrate with external state management libraries (e.g., Redux, Zustand) when necessary for complex state requirements.
    - Consider using Apollo Link State for managing local client-side state.
  - **Error Handling:**
    - Implement global error handling with Apollo Link.
    - Display user-friendly error messages to the user.
    - Retry failed queries or mutations automatically.

- **Performance Considerations:**
  - **Optimization Techniques:**
    - Use the `fetchPolicy` option in `useQuery` to control cache behavior.
    - Implement pagination for large datasets.
    - Use batching and caching at the server-side to reduce network requests.
  - **Memory Management:**
    - Unsubscribe from subscriptions when components unmount to prevent memory leaks.
    - Avoid storing large amounts of data in the Apollo Client cache.
  - **Rendering Optimization:**
    - Use React's `memo` or `useMemo` to prevent unnecessary re-renders.
    - Implement virtualization for rendering large lists.
  - **Bundle Size Optimization:**
    - Use code splitting to reduce the initial bundle size.
    - Remove unused dependencies.
    - Minify and compress your code.
  - **Lazy Loading:**
    - Lazy-load components and queries using dynamic imports.

- **Security Best Practices:**
  - **Common Vulnerabilities:**
    - **GraphQL Injection:** Prevent injection attacks by validating user input and using parameterized queries.
    - **Denial of Service (DoS):** Limit query complexity and depth to prevent DoS attacks.
    - **Information Disclosure:** Avoid exposing sensitive data in error messages.
  - **Input Validation:**
    - Validate all user input on both the client and server sides.
    - Use GraphQL's built-in validation capabilities.
  - **Authentication and Authorization:**
    - Implement proper authentication and authorization mechanisms.
    - Use JWT (JSON Web Tokens) or other secure authentication protocols.
    - Follow the principle of least privilege when granting access to data.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all API communication.
  - **Secure API Communication:**
    - Use Apollo Link to add security headers to API requests.
    - Implement CORS (Cross-Origin Resource Sharing) to prevent cross-site scripting (XSS) attacks.

- **Testing Approaches:**
  - **Unit Testing:**
    - Test individual components and hooks in isolation.
    - Mock Apollo Client's API to control data fetching behavior.
  - **Integration Testing:**
    - Test the integration of components with Apollo Client.
    - Use a test GraphQL server to simulate real API interactions.
  - **End-to-end Testing:**
    - Test the entire application flow from the user interface to the backend.
    - Use tools like Cypress or Puppeteer for end-to-end testing.
  - **Test Organization:**
    - Organize tests in a directory structure that mirrors the source code.
    - Use descriptive names for test files and test cases.
  - **Mocking and Stubbing:**
    - Use mocking libraries (e.g., Jest, Sinon) to mock Apollo Client's API.
    - Create stub GraphQL responses for testing different scenarios.

- **Common Pitfalls and Gotchas:**
  - **Frequent Mistakes:**
    - Incorrect cache configuration leading to stale data.
    - Over-fetching data in GraphQL queries.
    - Ignoring error handling in `useQuery` and `useMutation`.
  - **Edge Cases:**
    - Handling network errors and offline scenarios.
    - Dealing with large datasets and complex queries.
    - Managing cache invalidation and updates.
  - **Version-Specific Issues:**
    - Be aware of breaking changes between Apollo Client versions.
    - Consult the Apollo Client documentation for migration guides.
  - **Compatibility Concerns:**
    - Ensure compatibility with different browsers and devices.
    - Test your application on different platforms.
  - **Debugging Strategies:**
    - Use Apollo Client DevTools to inspect queries, mutations, and cache state.
    - Log API requests and responses for debugging purposes.

- **Tooling and Environment:**
  - **Recommended Tools:**
    - Apollo Client DevTools for debugging and inspecting the cache.
    - GraphQL Playground or GraphiQL for exploring and testing GraphQL APIs.
    - Apollo VS Code extension for syntax highlighting and autocompletion.
  - **Build Configuration:**
    - Use a build tool like Webpack or Parcel to bundle your code.
    - Configure Babel to transpile your code to compatible JavaScript versions.
  - **Linting and Formatting:**
    - Use ESLint and Prettier to enforce code style and best practices.
    - Configure linting rules specific to Apollo Client and GraphQL.
  - **Deployment:**
    - Use a deployment platform like Netlify, Vercel, or AWS Amplify.
    - Configure environment variables for API endpoints and other sensitive data.
  - **CI/CD Integration:**
    - Use a CI/CD pipeline to automate testing and deployment.
    - Integrate linters, formatters, and test runners into the pipeline.

- **Specific Best Practices based on Exa Search Results:**
  - Ensure all queries for a specific type return an ID for normalization and caching (RedwoodJS Community).
  - Store queries in separate `.gql` files and use tools like `graphql-import` for better organization (Reddit).
  - Use Apollo Client's `watchQuery` method for real-time data updates (Medium).
  - Follow GraphQL query best practices for efficient API design (Apollo GraphQL).

- **Additional considerations**
  - When creating queries and mutations, follow best practices to get the most out of both GraphQL and Apollo tooling.

- **Example Directory Structure**

src/
  components/
    UserCard.tsx
    ProductDetails.tsx
  graphql/
    queries/
      getUser.gql
      getProduct.gql
    mutations/
      createUser.gql
      updateProduct.gql
  hooks/
    useUser.ts
    useProduct.ts
  utils/
    apolloClient.ts // Initialize Apollo Client here


This is a comprehensive guide for using Apollo Client. Please adapt these guidelines to your specific project requirements and coding style.