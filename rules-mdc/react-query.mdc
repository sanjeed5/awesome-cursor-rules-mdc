---
description: This rule enforces best practices for using react-query in React applications, covering code organization, performance, security, and testing.
globs: *.js,*.jsx,*.ts,*.tsx
---
# react-query Best Practices

This document outlines the best practices for using react-query in React applications, covering various aspects such as code organization, performance considerations, security, and testing.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

*   **Feature-based Organization:** Group react-query hooks and related components within feature-specific directories.  This improves modularity and maintainability.
    
    src/
    ├── features/
    │   ├── users/
    │   │   ├── components/
    │   │   │   ├── UserList.tsx
    │   │   │   └── UserDetails.tsx
    │   │   ├── hooks/
    │   │   │   ├── useUsersQuery.ts
    │   │   │   └── useCreateUserMutation.ts
    │   │   ├── api/
    │   │   │   └── usersApi.ts
    │   │   └── types/
    │   │       └── user.ts
    │   ├── products/
    │   └── ...
    ├── ...
    

*   **Dedicated API Service Layer:** Abstract API interaction logic into separate modules.  This allows for easier testing and decoupling of components from specific API implementations. Consider using a dedicated `api` directory within each feature.

### 1.2 File Naming Conventions

*   **Consistent Naming:** Follow a consistent naming convention for react-query hooks.  Prefix hooks with `use` and postfix with `Query` or `Mutation` to clearly indicate their purpose (e.g., `usePostsQuery`, `useUpdatePostMutation`).

*   **Descriptive Names:** Use descriptive names for files and variables to improve code readability.  For example, `useFetchUsersQuery.ts` is more informative than `useUsers.ts`.

### 1.3 Module Organization

*   **Custom Hooks for Reusability:** Encapsulate react-query logic within custom hooks to promote reusability and separation of concerns.

    typescript
    // src/features/users/hooks/useUsersQuery.ts
    import { useQuery } from '@tanstack/react-query';
    import { fetchUsers } from '../api/usersApi';

    export const useUsersQuery = () => {
      return useQuery('users', fetchUsers);
    };
    

*   **Separate Query and Mutation Files:** Organize queries and mutations into separate files or modules. This enhances code readability and maintainability.

### 1.4 Component Architecture

*   **Presentational and Container Components:** Separate presentational components (UI) from container components (data fetching and state management).  This improves testability and reusability.

*   **Composition:** Use component composition to build complex UIs from smaller, reusable components.  Leverage React Context for shared state when appropriate.

### 1.5 Code Splitting Strategies

*   **Route-Based Splitting:** Split your application into separate bundles based on routes.  This reduces the initial load time and improves perceived performance. React.lazy and React.Suspense can assist with this.

*   **Component-Based Splitting:** Split large components into smaller chunks that can be loaded on demand.  This can improve the performance of individual pages or components.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to react-query

*   **Custom Hooks for Data Fetching:** As highlighted earlier, encapsulating react-query logic within custom hooks. This promotes reusability and separation of concerns. These hooks will typically return the result of a `useQuery` or `useMutation` call.

*   **Optimistic Updates:** Implement optimistic updates to improve perceived performance.  This involves updating the UI before the API request completes, and then reverting the changes if the request fails.  react-query provides utilities like `onMutate` to handle this.

*   **Pessimistic Updates:** Update the UI only after a successful response from the API. Simpler to implement but provides a less snappy user experience.

### 2.2 Recommended Approaches for Common Tasks

*   **Prefetching Data:** Prefetch data for routes or components that the user is likely to visit next.  This can significantly improve the user experience.  Use `queryClient.prefetchQuery`.

*   **Pagination and Infinite Scrolling:** Implement pagination and infinite scrolling to handle large datasets efficiently.  react-query provides hooks like `useInfiniteQuery` to simplify this.

*   **Dependent Queries:** Fetch data based on the result of a previous query. Use the `enabled` option in `useQuery` to conditionally execute queries.

    typescript
    const { data: user } = useQuery(['user', userId], () => fetchUser(userId));

    const { data: posts } = useQuery(['posts', user?.id], () => fetchPosts(user.id), {
      enabled: !!user,
    });
    

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Directly Calling API in Components:** Avoid making API calls directly within components.  This makes testing difficult and tightly couples components to specific API implementations.

*   **Ignoring Error Handling:** Always handle errors properly.  Display user-friendly error messages and provide options for retrying requests.

*   **Over-fetching Data:** Fetch only the data that is required by the component.  Use GraphQL or API query parameters to reduce the amount of data transferred.

*   **Deeply Nested Queries:** Avoid deeply nesting queries, as this can lead to performance issues and make the code difficult to understand. Consider combining queries or using a different approach.

### 2.4 State Management Best Practices

*   **Local vs. Global State:** Determine whether data should be stored in local component state or in a global state management solution.  Use local state for component-specific data and global state for data that needs to be shared across multiple components.

*   **react-query as a State Manager:** Leverage react-query's built-in caching and state management capabilities.  Avoid using external state management libraries for data that is already managed by react-query.

### 2.5 Error Handling Patterns

*   **Centralized Error Handling:** Implement centralized error handling to provide consistent error messages and logging.

*   **Retry Logic:** Implement retry logic to automatically retry failed requests. react-query provides options for configuring retry behavior.

*   **Error Boundaries:** Use Error Boundaries to catch errors that occur during rendering. This prevents the entire application from crashing.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Query Invalidation:** Invalidate queries when data changes. This ensures that the UI is always up-to-date.

*   **Stale-While-Revalidate:** Use the `staleTime` and `cacheTime` options to configure how long data should be considered fresh.  `staleWhileRevalidate` allows the UI to display cached data while fetching fresh data in the background.

*   **Window Focus Refetching:**  Configure refetching on window focus to keep data fresh when the user switches back to the application.

*   **Polling/Refetch Intervals:** Use `refetchInterval` to periodically refetch data.  This is useful for data that changes frequently.

### 3.2 Memory Management

*   **Query Cache Management:** Understand how react-query manages its cache.  Configure the `cacheTime` option to control how long data is stored in the cache.

*   **Garbage Collection:** Ensure that unused queries are garbage collected properly.  Use the `gcTime` option to configure how long inactive queries should be kept in the cache.

### 3.3 Rendering Optimization

*   **Memoization:** Use `React.memo` to prevent unnecessary re-renders of components.  This is especially important for components that display data fetched from react-query.

*   **Virtualization:** Use virtualization techniques (e.g., `react-window`, `react-virtualized`) to efficiently render large lists of data.

### 3.4 Bundle Size Optimization

*   **Tree Shaking:** Ensure that your build process is configured for tree shaking.  This removes unused code from the final bundle.

*   **Code Splitting:** As mentioned earlier, use code splitting to reduce the initial load time.

### 3.5 Lazy Loading Strategies

*   **Lazy Load Components:** Use `React.lazy` to lazy load components that are not immediately needed.

*   **Lazy Load Data:** Fetch data only when it is needed.  Use dependent queries to fetch data based on user interactions.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks. Use a library like DOMPurify to sanitize HTML.

*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection to prevent attackers from performing actions on behalf of the user. Use a library or framework that provides CSRF protection.

*   **Injection Attacks:** Protect against injection attacks by validating user input and using parameterized queries.

### 4.2 Input Validation

*   **Client-Side Validation:** Implement client-side validation to provide immediate feedback to the user.

*   **Server-Side Validation:** Always validate user input on the server-side to prevent malicious data from being stored in the database.

### 4.3 Authentication and Authorization Patterns

*   **JSON Web Tokens (JWT):** Use JWTs for authentication. Store the JWT in a secure cookie or in local storage (with caution). Use `httpOnly` flag on cookies containing JWTs when possible to prevent client-side script access.

*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to different parts of the application. Use middleware or custom hooks to check user roles.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit. Use HTTPS to encrypt data in transit. Encrypt sensitive data in the database.

*   **Data Masking:** Mask sensitive data in logs and reports. This prevents sensitive data from being exposed to unauthorized users.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication. This encrypts data in transit and prevents eavesdropping.

*   **API Rate Limiting:** Implement API rate limiting to prevent abuse.

*   **CORS:** Configure CORS properly to prevent cross-origin requests from unauthorized domains.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test Custom Hooks:** Unit test custom react-query hooks to ensure they are working correctly.  Mock the API calls using libraries like `msw` (Mock Service Worker).

*   **Test Components in Isolation:** Unit test components in isolation to ensure they render correctly and handle user interactions properly. Use libraries like `react-testing-library`.

### 5.2 Integration Testing

*   **Test Data Flow:** Integration test the data flow between components and APIs.  Verify that data is fetched correctly and displayed properly.

*   **Test Error Handling:** Integration test error handling scenarios to ensure that errors are handled properly.

### 5.3 End-to-End Testing

*   **Simulate User Interactions:** Use end-to-end testing frameworks like Cypress or Playwright to simulate user interactions and verify that the application is working correctly from the user's perspective.

*   **Test Critical Paths:** Focus on testing critical user flows, such as login, registration, and checkout.

### 5.4 Test Organization

*   **Colocate Tests with Components:** Colocate tests with the components they are testing.  This makes it easier to find and maintain tests.

*   **Use Descriptive Test Names:** Use descriptive test names to clearly indicate what each test is verifying.

### 5.5 Mocking and Stubbing

*   **Mock API Calls:** Use mocking libraries like `msw` to mock API calls during testing. This allows you to test components in isolation without relying on a real API.

*   **Stub External Dependencies:** Stub external dependencies to isolate components and make tests more predictable.

## 6. Common Pitfalls and Gotchas

*   **Forgetting to Invalidate Queries:** Failing to invalidate queries when data changes can lead to stale data being displayed in the UI.

*   **Incorrect Cache Configuration:** Incorrectly configuring the `cacheTime` and `staleTime` options can lead to performance issues or stale data.

*   **Not Handling Errors Properly:** Not handling errors properly can lead to unexpected behavior and a poor user experience.

*   **Over-relying on Default Configuration:** Customizing react-query to match specific needs is essential.

*   **Ignoring Devtools:** The react-query devtools are invaluable for debugging and understanding what is happening under the hood.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **VS Code:** Use VS Code with extensions like ESLint, Prettier, and TypeScript to improve developer productivity.

*   **React Developer Tools:** Use the React Developer Tools browser extension to inspect React components and state.

*   **react-query Devtools:** Use the react-query Devtools to inspect the react-query cache and track query and mutation status.

### 7.2 Build Configuration

*   **Webpack or Parcel:** Use a bundler like Webpack or Parcel to bundle your code for production. Configure the bundler for tree shaking and code splitting.

*   **Babel:** Use Babel to transpile your code to older versions of JavaScript. This ensures that your code is compatible with older browsers.

### 7.3 Linting and Formatting

*   **ESLint:** Use ESLint to enforce coding standards and prevent errors. Configure ESLint to use a popular style guide like Airbnb or Google.

*   **Prettier:** Use Prettier to automatically format your code. This ensures that your code is consistently formatted and easy to read.

### 7.4 Deployment Best Practices

*   **CDN:** Use a CDN to serve static assets. This improves performance and reduces the load on your server.

*   **Caching:** Configure caching properly on your server and CDN. This reduces the number of requests to your server and improves performance.

### 7.5 CI/CD Integration

*   **Automated Testing:** Integrate automated testing into your CI/CD pipeline. This ensures that your code is tested automatically before it is deployed.

*   **Automated Deployment:** Automate the deployment process to reduce the risk of errors and improve efficiency.