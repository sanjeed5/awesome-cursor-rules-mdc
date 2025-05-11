---
description: This rule provides comprehensive guidance on tRPC best practices, covering code organization, performance, security, testing, and common pitfalls to ensure robust and maintainable tRPC applications.
globs: *.ts,*.tsx
---
# tRPC Best Practices: A Comprehensive Guide

This document outlines best practices for developing robust, maintainable, and efficient applications using tRPC (TypeScript Remote Procedure Call). It covers various aspects, from code organization to security considerations, providing actionable guidance for developers.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

*   **Feature-Based Organization:** Organize your code around features or modules, rather than technical layers (e.g., `components`, `utils`, `services`).  This promotes modularity and maintainability. For example:

    
    src/
    ├── features/
    │   ├── user/
    │   │   ├── components/
    │   │   │   ├── UserProfile.tsx
    │   │   │   └── UserSettings.tsx
    │   │   ├── api/
    │   │   │   ├── userRouter.ts  // tRPC router for user-related procedures
    │   │   │   └── userSchema.ts // Zod schemas for input validation
    │   │   ├── hooks/
    │   │   │   └── useUser.ts    // Custom hooks for data fetching and state management
    │   │   └── types/
    │   │   │   └── user.ts        // TypeScript types related to users
    │   ├── product/
    │   │   └── ...
    ├── utils/
    │   ├── api.ts      // tRPC client initialization
    │   └── db.ts       // Database connection/abstraction
    ├── app/
    │   ├── api/
    │   │   └── root.ts   // Root tRPC router combining all feature routers
    │   └── context.ts // tRPC context creation
    └── index.ts      // Server entry point
    

*   **Separation of Concerns:**  Separate concerns into different directories and modules.  For instance, keep your tRPC router definitions separate from your business logic and data access layers.
*   **Grouping Similar Functionality:**  Keep related files together within a feature directory.  This makes it easier to understand and maintain the code.

### 1.2 File Naming Conventions

*   **Descriptive Names:** Use descriptive names for files and directories that clearly indicate their purpose.
*   **Consistent Case:**  Maintain a consistent casing convention (e.g., camelCase for variables, PascalCase for components). Use `kebab-case` for file names e.g. `user-profile.tsx` or `user.router.ts`
*   **Suffixes:** Use suffixes to indicate the type of file (e.g., `.router.ts` for tRPC routers, `.schema.ts` for Zod schemas, `.component.tsx` for React components).

### 1.3 Module Organization

*   **Small Modules:**  Keep modules small and focused. A module should have a single responsibility. Aim for modules that are easy to understand and test.
*   **Explicit Exports:** Use explicit exports to control what is exposed from a module. This helps to prevent accidental exposure of internal implementation details.
*   **Circular Dependencies:** Avoid circular dependencies between modules. Circular dependencies can lead to unexpected behavior and make it difficult to reason about the code.

### 1.4 Component Architecture

*   **Presentational and Container Components:** Separate presentational (dumb) components from container (smart) components. Presentational components focus on rendering UI, while container components handle data fetching and state management. Use the term `view` instead of `component` in the file suffix can make a separation between React components and presentational components, e.g. `user-profile.view.tsx`
*   **Reusable Components:** Design components to be reusable across different parts of the application. This reduces code duplication and improves maintainability.
*   **Component Composition:** Favor component composition over inheritance. Composition allows you to create more flexible and reusable components.

### 1.5 Code Splitting Strategies

*   **Route-Based Splitting:** Split your code based on routes. This allows you to load only the code that is needed for a specific route. This can be easily achieved with React's `lazy` and `Suspense` APIs.
*   **Component-Based Splitting:** Split your code based on components. This allows you to load only the code that is needed for a specific component.
*   **Dynamic Imports:** Use dynamic imports to load code on demand. This can be useful for loading large libraries or components that are not needed immediately. 

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to tRPC

*   **Router Composition:** Compose tRPC routers to create a hierarchical API structure. This allows you to organize your API into logical groups.

    typescript
    // src/app/api/routers/userRouter.ts
    import { publicProcedure, router } from "../trpc";
    import { z } from "zod";

    export const userRouter = router({
      getById: publicProcedure
        .input(z.string())
        .query(async ({ input }) => {
          // ... fetch user by ID
        }),
      create: publicProcedure
        .input(z.object({ name: z.string() }))
        .mutation(async ({ input }) => {
          // ... create a new user
        }),
    });

    // src/app/api/root.ts
    import { userRouter } from "./routers/userRouter";
    import { productRouter } from "./routers/productRouter";

    export const appRouter = router({
      user: userRouter,
      product: productRouter,
    });

    export type AppRouter = typeof appRouter;
    

*   **Middleware Chaining:** Use middleware to handle cross-cutting concerns such as authentication, authorization, and logging. Middleware can be chained to create a pipeline of operations.

    typescript
    // src/app/api/trpc.ts
    import { initTRPC, TRPCError } from "@trpc/server";
    import { Context } from "./context";

    const t = initTRPC.context<Context>().create();
    const isAuthed = t.middleware(({ ctx, next }) => {
      if (!ctx.user) {
        throw new TRPCError({ code: "UNAUTHORIZED" });
      }
      return next({
        ctx: {
          user: ctx.user,
        },
      });
    });

    export const router = t.router;
    export const publicProcedure = t.procedure;
    export const protectedProcedure = t.procedure.use(isAuthed);
    

*   **Input Validation with Zod:** Use Zod for input validation to ensure that data received from the client is valid. This helps to prevent errors and security vulnerabilities.

    typescript
    import { z } from "zod";
    import { publicProcedure } from "../trpc";

    export const createUserProcedure = publicProcedure
      .input(z.object({ name: z.string().min(3), email: z.string().email() }))
      .mutation(async ({ input }) => {
        // ... create a new user
      });
    

### 2.2 Recommended Approaches for Common Tasks

*   **Authentication:** Use tRPC middleware to authenticate users. Verify user credentials and set the user object in the context.
*   **Authorization:** Use tRPC middleware to authorize users. Check if the user has the required permissions to access a resource.
*   **Error Handling:** Use tRPC's built-in error handling mechanisms to handle errors gracefully. Throw `TRPCError` exceptions with appropriate error codes and messages.
*   **Data Fetching:**  Use a data fetching library (e.g., Prisma, Drizzle ORM, Supabase) to interact with your database. Abstract data access logic into separate modules or services.

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Over-fetching:** Avoid fetching more data than you need.  Use projections or GraphQL-style queries to fetch only the required fields.
*   **Under-fetching:** Avoid making multiple API calls to fetch related data. Batch requests or use a data loader pattern to fetch related data in a single call.
*   **Tight Coupling:** Avoid tight coupling between tRPC routers and your business logic.  Abstract business logic into separate modules or services.
*   **Ignoring Errors:** Never ignore errors. Always handle errors gracefully and provide meaningful feedback to the client.
*   **Direct Database Access in Routers:** Avoid accessing the database directly within tRPC router procedures. Instead, abstract data access into separate services or repositories.
*   **Complex Business Logic in Routers:** Keep tRPC router procedures focused on routing and input validation. Move complex business logic to separate functions or modules.

### 2.4 State Management Best Practices

*   **Centralized State Management:** Use a centralized state management library (e.g., Zustand, Redux, Jotai) to manage application state. This makes it easier to share state between components and to reason about the application's state.
*   **Immutable State:** Use immutable state to prevent unexpected side effects. This makes it easier to reason about the application's state and to debug issues.
*   **Controlled Components:** Use controlled components to manage form state. This gives you more control over the form and makes it easier to validate input.
*   **Server State Management:** Use a library like TanStack Query or SWR to manage server state (data fetched from the API). These libraries provide caching, optimistic updates, and other features that make it easier to manage server state.

### 2.5 Error Handling Patterns

*   **Centralized Error Handling:** Use a centralized error handling mechanism to handle errors consistently across the application.
*   **Error Boundaries:** Use error boundaries to prevent errors from crashing the application. Error boundaries can catch errors that occur during rendering and display a fallback UI.
*   **Logging:** Log errors to a central logging service. This makes it easier to track down and fix issues.
*   **User-Friendly Error Messages:** Display user-friendly error messages to the user. Avoid displaying technical details or stack traces.
*   **Retry Mechanism:** Implement a retry mechanism for transient errors. This can improve the resilience of the application.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Caching:** Implement caching to reduce the number of API calls. Cache data on the server and the client.
*   **Compression:** Use compression to reduce the size of API responses. This can improve the performance of the application, especially on slow networks.
*   **Code Splitting:** Split your code into smaller chunks to reduce the initial load time. Load code on demand as needed.
*   **Debouncing and Throttling:** Use debouncing and throttling to reduce the number of API calls triggered by user input.
*   **Efficient Data Structures:** Use efficient data structures to store and process data. Choose the right data structure for the task at hand.

### 3.2 Memory Management

*   **Avoid Memory Leaks:** Be careful to avoid memory leaks. Memory leaks can cause the application to slow down and eventually crash.
*   **Garbage Collection:** Understand how garbage collection works in JavaScript. This can help you to avoid memory leaks and to optimize memory usage.
*   **Use Weak References:** Use weak references to avoid keeping objects in memory longer than necessary.

### 3.3 Rendering Optimization

*   **Memoization:** Use memoization to avoid re-rendering components unnecessarily. Memoize components that receive the same props multiple times.
*   **Virtualization:** Use virtualization to render large lists efficiently. Virtualization only renders the items that are visible on the screen.
*   **Batch Updates:** Batch updates to reduce the number of re-renders. React's `useState` hook batches updates automatically.

### 3.4 Bundle Size Optimization

*   **Tree Shaking:** Use tree shaking to remove unused code from the bundle. This can significantly reduce the bundle size.
*   **Code Minification:** Minify your code to reduce the bundle size. Minification removes whitespace and comments from the code.
*   **Image Optimization:** Optimize images to reduce the bundle size. Use appropriate image formats and compress images.
*   **Dependency Analysis:** Analyze your dependencies to identify large or unnecessary dependencies. Consider replacing large dependencies with smaller alternatives.

### 3.5 Lazy Loading Strategies

*   **Lazy Load Images:** Lazy load images to improve the initial load time. Only load images when they are visible on the screen.
*   **Lazy Load Components:** Lazy load components to improve the initial load time. Only load components when they are needed.
*   **Lazy Load Modules:** Lazy load modules to improve the initial load time. Only load modules when they are needed.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by escaping user input and using a content security policy (CSP).
*   **Cross-Site Request Forgery (CSRF):** Prevent CSRF attacks by using CSRF tokens. Most frameworks have built-in support for CSRF protection.
*   **SQL Injection:** Prevent SQL injection attacks by using parameterized queries or an ORM.
*   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms to protect sensitive data.
*   **Rate Limiting:** Implement rate limiting to prevent brute-force attacks.
*   **Denial-of-Service (DoS):** Implement measures to prevent DoS attacks, such as rate limiting and input validation.

### 4.2 Input Validation

*   **Validate All Input:** Validate all input from the client, including query parameters, request bodies, and headers.
*   **Use Strong Types:** Use strong types to define the expected format of input data. This can help to prevent type-related errors and security vulnerabilities. Tools like Zod are invaluable here.
*   **Sanitize Input:** Sanitize input to remove potentially harmful characters or code. This can help to prevent XSS attacks.
*   **Whitelist Input:** Whitelist allowed input values. This is more secure than blacklisting disallowed values.

### 4.3 Authentication and Authorization Patterns

*   **JWT (JSON Web Tokens):** Use JWTs for authentication. JWTs are a standard way to securely transmit information between parties as a JSON object.
*   **OAuth 2.0:** Use OAuth 2.0 for authorization. OAuth 2.0 is a standard protocol for delegating access to resources.
*   **Role-Based Access Control (RBAC):** Use RBAC to control access to resources based on user roles. This makes it easier to manage permissions and to enforce security policies.
*   **Attribute-Based Access Control (ABAC):** Use ABAC to control access to resources based on user attributes. This provides more fine-grained control over access to resources.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit. Use strong encryption algorithms.
*   **Hashing:** Hash passwords and other sensitive data. Use a strong hashing algorithm with a salt.
*   **Data Masking:** Mask sensitive data in logs and other outputs. This prevents sensitive data from being exposed accidentally.
*   **Data Redaction:** Redact sensitive data from API responses. This prevents sensitive data from being exposed to unauthorized users.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication. HTTPS encrypts the communication between the client and the server, protecting it from eavesdropping and tampering.
*   **CORS (Cross-Origin Resource Sharing):** Configure CORS to restrict access to your API from unauthorized domains. This helps to prevent cross-site scripting attacks.
*   **Rate Limiting:** Implement rate limiting to prevent brute-force attacks and DoS attacks.
*   **API Keys:** Use API keys to authenticate API clients. This helps to prevent unauthorized access to your API.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test Individual Functions and Modules:** Unit tests should focus on testing individual functions and modules in isolation.
*   **Mock Dependencies:** Mock dependencies to isolate the code under test. This prevents external dependencies from interfering with the test.
*   **Test Edge Cases:** Test edge cases to ensure that the code handles unexpected input correctly.
*   **Test Error Handling:** Test error handling to ensure that the code handles errors gracefully.
*   **Use Test-Driven Development (TDD):** Consider using TDD to write tests before writing code. This can help to improve the design of the code and to ensure that it is testable.

### 5.2 Integration Testing

*   **Test Interactions Between Modules:** Integration tests should focus on testing the interactions between modules. This ensures that the modules work together correctly.
*   **Use Real Dependencies:** Use real dependencies in integration tests. This provides more realistic testing conditions.
*   **Test Data Flow:** Test the data flow between modules to ensure that data is passed correctly.

### 5.3 End-to-End Testing

*   **Test the Entire Application:** End-to-end tests should focus on testing the entire application, from the user interface to the backend API.
*   **Use a Test Automation Framework:** Use a test automation framework (e.g., Cypress, Playwright) to automate end-to-end tests. This makes it easier to run the tests and to verify that the application is working correctly.
*   **Test User Flows:** Test common user flows to ensure that users can complete important tasks.

### 5.4 Test Organization

*   **Organize Tests by Feature:** Organize tests by feature. This makes it easier to find and run tests for a specific feature.
*   **Keep Tests Separate from Code:** Keep tests separate from the code. This prevents tests from being included in the production bundle.
*   **Use a Consistent Naming Convention:** Use a consistent naming convention for tests. This makes it easier to identify and understand the tests.

### 5.5 Mocking and Stubbing

*   **Use Mocking Frameworks:** Use mocking frameworks (e.g., Jest, Sinon) to create mocks and stubs. This makes it easier to isolate the code under test.
*   **Mock External Dependencies:** Mock external dependencies to prevent them from interfering with the test.
*   **Stub API Responses:** Stub API responses to control the data that is returned by the API.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Not Validating Input:** Failing to validate input is a common mistake that can lead to errors and security vulnerabilities.
*   **Ignoring Errors:** Ignoring errors can make it difficult to debug issues and can lead to unexpected behavior.
*   **Over-Complicating Code:** Over-complicating code can make it difficult to understand and maintain.
*   **Not Writing Tests:** Not writing tests can lead to bugs and can make it difficult to refactor the code.
*   **Incorrect Context Usage:** Misunderstanding how the tRPC context works, especially when dealing with middleware and authentication.

### 6.2 Edge Cases to Be Aware Of

*   **Empty Input:** Handle empty input gracefully. Provide default values or display an error message.
*   **Invalid Input:** Handle invalid input gracefully. Display an error message to the user.
*   **Network Errors:** Handle network errors gracefully. Display an error message to the user and provide a way to retry the request.
*   **Server Errors:** Handle server errors gracefully. Display an error message to the user and log the error on the server.
*   **Concurrency Issues:** Be aware of concurrency issues when dealing with shared resources. Use locks or other synchronization mechanisms to prevent data corruption.

### 6.3 Version-Specific Issues

*   **Breaking Changes:** Be aware of breaking changes when upgrading tRPC. Review the release notes carefully and update your code accordingly.
*   **Deprecated Features:** Be aware of deprecated features. Replace deprecated features with the recommended alternatives.
*   **Compatibility Issues:** Be aware of compatibility issues with other libraries. Test your code thoroughly after upgrading tRPC or other libraries.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:** Ensure that your code is compatible with the browsers that you support.
*   **Node.js Version Compatibility:** Ensure that your code is compatible with the Node.js versions that you support.
*   **TypeScript Version Compatibility:** Ensure that your code is compatible with the TypeScript version that you are using.

### 6.5 Debugging Strategies

*   **Use Debugging Tools:** Use debugging tools (e.g., Chrome DevTools, VS Code debugger) to step through your code and inspect variables.
*   **Add Logging Statements:** Add logging statements to your code to track the flow of execution and to identify errors.
*   **Use a Debugger:** Use a debugger to pause the execution of your code and to inspect the state of the application.
*   **Reproduce the Issue:** Try to reproduce the issue in a controlled environment. This makes it easier to identify the cause of the issue.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **VS Code:** Use VS Code as your IDE. VS Code provides excellent support for TypeScript and JavaScript development.
*   **ESLint:** Use ESLint to enforce code style and to prevent errors.
*   **Prettier:** Use Prettier to format your code automatically.
*   **TypeScript Compiler:** Use the TypeScript compiler to compile your code.
*   **npm or Yarn:** Use npm or Yarn to manage your dependencies.
*   **Testing Framework:** Jest is a great testing framework

### 7.2 Build Configuration

*   **Use a Build Tool:** Use a build tool (e.g., Webpack, Parcel, Rollup) to bundle your code for production.
*   **Configure the Build Tool:** Configure the build tool to optimize the bundle size, to minify the code, and to perform tree shaking.
*   **Use Environment Variables:** Use environment variables to configure the build process.

### 7.3 Linting and Formatting

*   **Configure ESLint:** Configure ESLint to enforce code style and to prevent errors. Use a consistent code style across the entire project.
*   **Configure Prettier:** Configure Prettier to format your code automatically. This ensures that the code is consistently formatted.
*   **Use a Pre-Commit Hook:** Use a pre-commit hook to run ESLint and Prettier before committing code. This prevents code with style violations or errors from being committed.

### 7.4 Deployment Best Practices

*   **Use a Deployment Platform:** Use a deployment platform (e.g., Vercel, Netlify, AWS) to deploy your application.
*   **Configure the Deployment Platform:** Configure the deployment platform to automatically deploy your application when changes are pushed to the repository.
*   **Use Environment Variables:** Use environment variables to configure the deployment environment.
*   **Monitor the Application:** Monitor the application to identify and resolve issues.

### 7.5 CI/CD Integration

*   **Use a CI/CD Tool:** Use a CI/CD tool (e.g., GitHub Actions, GitLab CI, CircleCI) to automate the build, test, and deployment process.
*   **Configure the CI/CD Tool:** Configure the CI/CD tool to run tests, build the application, and deploy the application automatically.
*   **Use Environment Variables:** Use environment variables to configure the CI/CD environment.

By adhering to these best practices, you can build robust, maintainable, and efficient applications using tRPC.