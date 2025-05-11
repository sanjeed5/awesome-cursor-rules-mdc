---
description: Comprehensive best practices for developing with Supabase, covering code organization, performance, security, testing, and common pitfalls. This rule provides actionable guidance for developers to build robust and scalable applications using Supabase.
globs: *.js,*.jsx,*.ts,*.tsx,*.sql
---
# Supabase Best Practices: A Comprehensive Guide

This document outlines best practices for developing with Supabase, covering various aspects from code organization to security and performance.

## 1. Code Organization and Structure

A well-organized codebase is crucial for maintainability and scalability. Here's how to structure your Supabase project:

### 1.1. Directory Structure Best Practices

Adopt a modular structure that separates concerns:


project-root/
├── src/
│   ├── components/        # Reusable UI components
│   ├── pages/             # Page-level components (routes)
│   ├── services/          # Supabase client initialization and data fetching
│   │   ├── supabase.ts   # Supabase client initialization
│   │   ├── auth.ts       # Authentication-related functions
│   │   ├── database.ts   # Database interaction functions
│   ├── utils/             # Utility functions (e.g., date formatting)
│   ├── types/             # TypeScript types and interfaces
│   ├── hooks/             # Custom React hooks
│   ├── styles/            # Global styles and theme
│   └── App.tsx            # Main application component
├── migrations/        # Database migrations
├── tests/              # Unit and integration tests
├── .env                # Environment variables
└── package.json          # Project dependencies


### 1.2. File Naming Conventions

*   **Components:** Use PascalCase (e.g., `UserProfile.tsx`).
*   **Functions:** Use camelCase (e.g., `fetchUserData`).
*   **Variables:** Use camelCase (e.g., `userName`).
*   **Files:** Use kebab-case (e.g., `user-profile.tsx`).

### 1.3. Module Organization

*   Group related functionalities into modules.
*   Use clear and descriptive module names.
*   Export only what is necessary from each module to minimize the API surface.

### 1.4. Component Architecture

*   Favor small, reusable components.
*   Use a component-based approach (e.g., React, Vue.js) to build UIs.
*   Separate presentational components from container components to improve reusability and testability.

### 1.5. Code Splitting Strategies

*   Use dynamic imports to lazy-load modules.
*   Split large components into smaller chunks.
*   Implement route-based code splitting to load only the necessary code for each route.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to Supabase

*   **Repository Pattern:** Abstract database interactions behind a repository interface for better testability and maintainability.
*   **Observer Pattern:** Utilize Supabase's real-time capabilities to implement reactive UIs.

### 2.2. Recommended Approaches for Common Tasks

*   **Data Fetching:** Create reusable functions for fetching data from Supabase tables.
*   **Authentication:** Use Supabase Auth for user authentication and authorization.
*   **Real-time Updates:** Leverage Supabase Realtime for real-time data synchronization.
*   **File Storage:** Utilize Supabase Storage for managing file uploads and downloads.

### 2.3. Anti-patterns and Code Smells to Avoid

*   **Direct Database Access from UI Components:** This tightly couples the UI to the database and makes testing difficult. Use a service layer to abstract database interactions.
*   **Overusing Supabase Functions and Policies:** Keep business logic in your application code whenever possible to avoid vendor lock-in and improve testability.
*   **Manually Creating Tables Without an ORM:** Always use an ORM to manage your database schema and migrations.
*   **Ignoring Error Handling:** Implement proper error handling to prevent unexpected crashes and provide informative error messages to users.
*   **Storing Sensitive Data in Plain Text:** Never store sensitive data like passwords or API keys in plain text. Use encryption and secure storage mechanisms.

### 2.4. State Management Best Practices

*   Choose a state management library (e.g., Redux, Zustand, Recoil) based on your project's complexity.
*   Use local component state for simple UI state.
*   Centralize application state in a global store for complex data dependencies.
*   Use asynchronous actions to handle data fetching and updates.

### 2.5. Error Handling Patterns

*   Use try-catch blocks to handle exceptions.
*   Implement a global error handler to catch unhandled exceptions.
*   Log errors to a monitoring service for tracking and analysis.
*   Display user-friendly error messages to the user.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Indexing:** Add indexes to frequently queried columns to improve query performance.
*   **Query Optimization:** Use efficient SQL queries and avoid N+1 query problems.
*   **Caching:** Implement caching strategies to reduce database load.
*   **Connection Pooling:** Use connection pooling to reuse database connections and reduce overhead.

### 3.2. Memory Management

*   Avoid memory leaks by properly cleaning up resources.
*   Use garbage collection to reclaim unused memory.
*   Optimize data structures to reduce memory usage.

### 3.3. Rendering Optimization

*   Use memoization techniques to prevent unnecessary re-renders.
*   Virtualize long lists to improve rendering performance.
*   Optimize images and other assets to reduce file sizes.

### 3.4. Bundle Size Optimization

*   Use tree shaking to remove unused code.
*   Minify code to reduce file sizes.
*   Compress assets to reduce transfer times.

### 3.5. Lazy Loading Strategies

*   Lazy load images and other assets that are not immediately visible.
*   Implement infinite scrolling to load data in chunks as the user scrolls.
*   Use code splitting to load only the necessary code for each route.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

*   **SQL Injection:** Prevent SQL injection by using parameterized queries and prepared statements.
*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection to prevent unauthorized requests.
*   **Authentication and Authorization Issues:** Secure your authentication and authorization mechanisms to prevent unauthorized access.

### 4.2. Input Validation

*   Validate all user input to prevent malicious data from entering your system.
*   Use server-side validation in addition to client-side validation.
*   Sanitize user input to remove potentially harmful characters.

### 4.3. Authentication and Authorization Patterns

*   Use Supabase Auth for user authentication and authorization.
*   Implement role-based access control (RBAC) to manage user permissions.
*   Use row-level security (RLS) to control data access at the row level.

### 4.4. Data Protection Strategies

*   Encrypt sensitive data at rest and in transit.
*   Use secure storage mechanisms to store API keys and other secrets.
*   Implement data masking to protect sensitive data from unauthorized access.

### 4.5. Secure API Communication

*   Use HTTPS to encrypt API traffic.
*   Implement API rate limiting to prevent abuse.
*   Validate API requests to prevent malicious input.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   Write unit tests for individual functions and components.
*   Use mocking and stubbing to isolate units of code.
*   Aim for high code coverage.

### 5.2. Integration Testing

*   Write integration tests to verify the interaction between different parts of your system.
*   Test the integration between your application and Supabase.

### 5.3. End-to-end Testing

*   Write end-to-end tests to simulate user interactions and verify the overall functionality of your application.
*   Use tools like Cypress or Playwright to automate end-to-end tests.

### 5.4. Test Organization

*   Organize tests into separate directories based on functionality.
*   Use descriptive test names.
*   Keep tests concise and focused.

### 5.5. Mocking and Stubbing

*   Use mocking to replace external dependencies with controlled substitutes.
*   Use stubbing to replace specific method calls with predefined values.
*   Avoid over-mocking, as it can make tests less reliable.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

*   Not using an ORM for database schema management.
*   Over-relying on Supabase functions and policies for business logic.
*   Using Supabase-only features without considering open-source alternatives.
*   Ignoring error handling and security best practices.

### 6.2. Edge Cases to Be Aware Of

*   Handling large datasets efficiently.
*   Dealing with real-time data synchronization conflicts.
*   Managing user sessions and authentication tokens securely.

### 6.3. Version-Specific Issues

*   Be aware of breaking changes in Supabase and its dependencies.
*   Test your application thoroughly after upgrading Supabase or its dependencies.
*   Refer to the Supabase documentation for migration guides.

### 6.4. Compatibility Concerns

*   Ensure compatibility between your application and the Supabase client library.
*   Test your application on different browsers and devices.

### 6.5. Debugging Strategies

*   Use browser developer tools to debug client-side code.
*   Use server-side logging to track errors and performance issues.
*   Use the Supabase dashboard to monitor database activity.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   Supabase CLI: For local development and database management.
*   VS Code: For code editing and debugging.
*   Drizzle ORM: For database schema management.
*   Postman/Insomnia: For testing API endpoints.

### 7.2. Build Configuration

*   Use a build tool like Webpack or Parcel to bundle your code.
*   Configure your build tool to optimize for production.
*   Use environment variables to configure your application.

### 7.3. Linting and Formatting

*   Use ESLint to enforce code style and prevent errors.
*   Use Prettier to format your code automatically.
*   Integrate linting and formatting into your development workflow.

### 7.4. Deployment Best Practices

*   Use a CI/CD pipeline to automate deployments.
*   Deploy your application to a production-ready environment.
*   Monitor your application for errors and performance issues.

### 7.5. CI/CD Integration

*   Use a CI/CD tool like GitHub Actions or GitLab CI to automate your build, test, and deployment processes.
*   Configure your CI/CD pipeline to run tests and linting before deployment.
*   Use environment variables to configure your application in the CI/CD environment.

## 8. Database Workflow Design

*   **Avoid Direct Changes in Production**: Once your application is live, refrain from making database changes using the Supabase Dashboard. Instead, utilize migration tools and enforce access control to prevent unauthorized modifications.
*   **Multiple Environments**: Adopt a multi-stage development workflow (`local -> staging -> prod`). This approach allows for thorough testing and validation at each stage before changes are deployed to production.
*   **Point-in-Time Recovery**: As your database grows, consider moving to Point-in-Time Recovery (PITR) to minimize the impact on database performance during maintenance and ensure data safety.
*   **Database Migrations**: Use database migration tools to manage schema changes. This practice helps maintain consistency across different environments and simplifies version control.
*   **Access Control**: Be mindful of who has access to your production environment. Limit access to experienced team members and set clear internal workflows to mitigate risks.
*   **Security**: Regularly review and update your security measures. Ensure that tables with sensitive data have appropriate access levels and that database secrets and API keys are stored securely.
*   **Performance Monitoring**: Utilize Supabase's observability tooling to monitor database performance and optimize queries, indexes, and connection management.

## 9. Additional Best Practices

*   **Understand Shared Responsibilities:** When using Supabase, be aware of the shared responsibilities model. Supabase manages infrastructure, but you are responsible for application architecture, security, and data management.
*   **Proper Indexing:** Essential for query optimization. Indices should be created based on common query patterns to reduce search time.
*   **Load Testing**: Before deploying changes to production, perform load testing in a staging environment. Tools such as `k6` can simulate traffic and help identify potential bottlenecks.
*   **Resource Upgrades**: Monitor resource usage and upgrade your database when necessary. For significant traffic surges, contact support in advance for assistance.
*   **Database Optimization**: Regularly optimize your database by adding filters on large queries, using caching strategies, and managing connections efficiently.
*   **Regular Backups:** Schedule regular backups of your database to protect against data loss.
*   **Use of Postgres Features**: As Supabase is built around Postgres, understand and leverage Postgres features for performance and scalability.

## 10. Conclusion

By following these best practices, you can build robust, scalable, and secure applications with Supabase. Remember to stay up-to-date with the latest Supabase documentation and community resources to continuously improve your development practices.

@file ./supabase_code_examples.mdc