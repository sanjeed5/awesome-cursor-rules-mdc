---
description: This rule provides comprehensive best practices for developing applications using the notion-api library, covering code organization, performance, security, testing, and common pitfalls.
globs: *.js,*.ts,*.jsx,*.tsx
---
# Notion API Library Best Practices

This document outlines the recommended best practices for developing applications using the `notion-api` library. Following these guidelines will help you create maintainable, performant, secure, and testable Notion integrations.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

Adopt a clear and consistent directory structure to improve code maintainability and collaboration. Here's a suggested structure for projects using `notion-api`:


project-root/
 ├── src/
 │   ├── components/        # Reusable UI components (if applicable)
 │   ├── pages/             # Page-specific components/logic (if applicable)
 │   ├── services/          # Notion API interaction logic
 │   │   ├── notion-client.js  # or notion-client.ts: Initializes the Notion client
 │   │   ├── database-service.js # or database-service.ts: Handles database interactions
 │   │   ├── page-service.js     # or page-service.ts: Handles page interactions
 │   │   └── utils.js          # or utils.ts: Utility functions for API calls, data formatting
 │   ├── models/             # Data models for Notion objects
 │   ├── utils/              # General utility functions
 │   ├── config/             # Configuration files
 │   └── app.js              # or app.ts: Main application entry point
 ├── tests/              # Unit and integration tests
 ├── .env                 # Environment variables
 ├── package.json          # Project dependencies
 └── README.md            # Project documentation


### 1.2. File Naming Conventions

Use descriptive and consistent file names:

*   **Components:** `ComponentName.jsx` or `ComponentName.tsx`
*   **Services:** `service-name.js` or `service-name.ts`
*   **Models:** `model-name.js` or `model-name.ts`
*   **Utility functions:** `utility-name.js` or `utility-name.ts`
*   **Configuration files:** `config-name.js` or `config-name.ts`

### 1.3. Module Organization

Organize your code into logical modules based on functionality.  For `notion-api`, this commonly involves separating:

*   **Notion Client Initialization:**  A module responsible for initializing the Notion client with your API key. This should handle authentication.
*   **Data Access Layer (Services):** Modules that encapsulate all interactions with the Notion API. These modules should provide functions for common operations like creating pages, querying databases, and updating content.  Use separate modules for database and page-related operations.
*   **Data Models:**  Define TypeScript interfaces or JavaScript classes to represent Notion objects (e.g., Page, Database, Block). This improves type safety and code readability.
*   **Utility Functions:**  Create utility functions for tasks such as formatting data for the Notion API, handling pagination, and error handling.

### 1.4. Component Architecture (If Applicable)

If you're building a UI that interacts with the Notion API, consider using a component-based architecture (e.g., React, Vue, Angular).  Separate concerns by creating reusable components for displaying and interacting with Notion data.

### 1.5. Code Splitting

For larger applications, use code splitting to improve initial load times.  Lazy-load components and modules that are not immediately needed. This is particularly useful for components that display large amounts of Notion data or perform complex API operations.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Repository Pattern:** Use the repository pattern to abstract data access logic.  This makes it easier to switch data sources in the future (e.g., using a mock API for testing).
*   **Adapter Pattern:**  Adapt the data returned by the Notion API to your application's specific needs. This helps decouple your application from the API's data structure.
*   **Singleton Pattern (for Notion Client):** Use the singleton pattern to ensure only one instance of the Notion client is created. This can improve performance and avoid rate limiting issues.

### 2.2. Recommended Approaches for Common Tasks

*   **Fetching Data:** Use asynchronous functions (`async/await`) to fetch data from the Notion API. Implement proper error handling (see below).
*   **Creating Pages:**  Use the `pages.create` method to create new pages. Ensure you provide the required properties for the parent (database or page).
*   **Querying Databases:** Use the `databases.query` method to query databases.  Learn how to use filters, sorts, and pagination to retrieve the data you need efficiently.
*   **Updating Content:**  Use the `blocks.update` and `pages.update` method to update content. Understand how to update different types of blocks and page properties.
*   **Handling Pagination:** Implement proper pagination to handle large datasets.  Use the `has_more` and `next_cursor` properties returned by the API to iterate through all pages of data.

### 2.3. Anti-patterns and Code Smells

*   **Directly calling the Notion API in UI components:** This violates separation of concerns and makes testing difficult.  Use a service layer to handle API interactions.
*   **Hardcoding API keys:** Store API keys in environment variables and never commit them to your repository.
*   **Ignoring errors:** Always handle errors and log them appropriately.  Do not swallow exceptions.
*   **Over-fetching data:** Only request the data you need. Use filters and projections to reduce the amount of data transferred.
*   **Creating too many Notion client instances:**  Re-use existing client instances to optimize performance and avoid rate limiting.
*   **Lack of Pagination:** Neglecting to handle pagination when fetching large datasets from Notion databases.

### 2.4. State Management (If Applicable)

If you're building a UI, choose a state management solution that fits your application's complexity (e.g., React Context, Redux, Zustand).  Store Notion data in a normalized format to improve performance and avoid unnecessary re-renders.  Use selectors to efficiently retrieve data from the state.

### 2.5. Error Handling Patterns

Implement robust error handling to gracefully handle API errors and unexpected situations.

*   **Try-Catch Blocks:** Use `try-catch` blocks to wrap API calls and handle potential errors.
*   **Error Logging:** Log errors with sufficient detail to facilitate debugging.
*   **User Feedback:** Provide informative error messages to the user.
*   **Retry Mechanism:** Implement a retry mechanism for transient errors (e.g., network errors, rate limiting).
*   **Centralized Error Handling:** Create a centralized error handling function or middleware to handle errors consistently across your application.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Caching:** Implement caching to reduce the number of API calls. Cache frequently accessed data in memory or in a persistent store.
*   **Batching:** Batch multiple API requests into a single request to reduce network overhead (where supported by the API.  Currently, Notion's API offers limited batching capabilities.
*   **Minimize Data Transfer:** Only request the data you need.  Use filters and projections to reduce the amount of data transferred.
*   **Compression:** Enable compression (e.g., gzip) to reduce the size of API responses.
*   **Optimize Database Queries:** Carefully design your database queries to retrieve data efficiently. Use indexes to improve query performance.

### 3.2. Memory Management

*   **Avoid Memory Leaks:**  Be mindful of memory leaks, especially in long-running applications.  Release resources when they are no longer needed.
*   **Use Efficient Data Structures:**  Use efficient data structures to store Notion data. Avoid creating unnecessary copies of data.

### 3.3. Rendering Optimization (If Applicable)

*   **Virtualization:** Use virtualization to efficiently render large lists of Notion blocks.
*   **Memoization:** Use memoization to avoid re-rendering components that haven't changed.
*   **Code Splitting:** As described above.

### 3.4. Bundle Size Optimization

*   **Tree Shaking:** Use tree shaking to remove unused code from your bundles.
*   **Minification:** Minify your code to reduce bundle size.
*   **Code Splitting:** As described above.
*   **Use a CDN:** Serve static assets from a CDN to improve loading times.

### 3.5. Lazy Loading

*   **Lazy Load Images:** Lazy load images to improve initial load times.
*   **Lazy Load Components:** As described above.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

*   **API Key Exposure:** Never commit API keys to your repository. Store them in environment variables and access them securely.
*   **Injection Attacks:**  Sanitize user input to prevent injection attacks (e.g., SQL injection, XSS).
*   **Cross-Site Scripting (XSS):** If your application renders user-generated content from Notion, sanitize it to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Implement CSRF protection to prevent malicious websites from making requests on behalf of authenticated users.
*   **Man-in-the-Middle Attacks:** Ensure that all communication with the Notion API is encrypted using HTTPS.

### 4.2. Input Validation

*   **Validate All User Input:**  Validate all user input before sending it to the Notion API. This includes checking data types, formats, and ranges.
*   **Sanitize User Input:** Sanitize user input to remove potentially harmful characters.

### 4.3. Authentication and Authorization

*   **Use OAuth 2.0:** Use OAuth 2.0 for authentication and authorization. This allows users to grant your application access to their Notion data without sharing their passwords.
*   **Store Tokens Securely:** Store access tokens securely.  Use encryption or a secure storage mechanism (e.g., Keychain).
*   **Follow the Principle of Least Privilege:** Only request the permissions you need.

### 4.4. Data Protection

*   **Encrypt Sensitive Data:**  Encrypt sensitive data at rest and in transit.
*   **Follow Data Privacy Regulations:**  Comply with all applicable data privacy regulations (e.g., GDPR, CCPA).
*   **Regularly Audit Security Practices:** Regularly audit your security practices to identify and address potential vulnerabilities.

### 4.5. Secure API Communication

*   **Use HTTPS:** Ensure that all communication with the Notion API is encrypted using HTTPS.
*   **Validate SSL Certificates:** Validate SSL certificates to prevent man-in-the-middle attacks.
*   **Use a Secure API Client:** Use a secure API client that supports encryption and certificate validation.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Individual Components:** Unit test individual components and modules to ensure they function correctly in isolation.
*   **Mock External Dependencies:** Mock external dependencies (e.g., the Notion API) to isolate your code and make tests more predictable.
*   **Use a Testing Framework:** Use a testing framework like Jest or Mocha to write and run your unit tests.

### 5.2. Integration Testing

*   **Test Interactions Between Components:** Integration test the interactions between different components and modules to ensure they work together correctly.
*   **Use a Test Database:** Use a test database to avoid affecting your production data.
*   **Seed the Test Database:** Seed the test database with sample data to make tests more realistic.

### 5.3. End-to-End Testing

*   **Test the Entire Application Flow:** End-to-end test the entire application flow to ensure that everything works correctly from start to finish.
*   **Use a Testing Framework:** Use a testing framework like Cypress or Puppeteer to write and run your end-to-end tests.
*   **Run Tests in a CI/CD Pipeline:** Run your tests in a CI/CD pipeline to ensure that changes don't break existing functionality.

### 5.4. Test Organization

*   **Create a Dedicated Test Directory:** Create a dedicated test directory to store your tests.
*   **Follow a Consistent Naming Convention:** Follow a consistent naming convention for your test files.
*   **Organize Tests by Feature:** Organize tests by feature to make them easier to find and maintain.

### 5.5. Mocking and Stubbing

*   **Use Mocking Libraries:** Use mocking libraries like Jest or Sinon to create mocks and stubs.
*   **Mock the Notion API:** Mock the Notion API to isolate your code and make tests more predictable.
*   **Stub API Responses:** Stub API responses to control the data returned by the API.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Not Understanding the Notion API Data Model:**  Failing to understand the Notion API data model can lead to incorrect API calls and unexpected results. Study the documentation thoroughly.
*   **Misunderstanding Rate Limits:**  Exceeding rate limits can lead to temporary blocking. Implement proper rate limiting handling and retry mechanisms.
*   **Improperly Handling Errors:**  Ignoring errors can lead to silent failures and difficult debugging. Always handle errors and log them appropriately.
*   **Not Using Pagination:**  Failing to use pagination can lead to incomplete data retrieval. Implement proper pagination to handle large datasets.

### 6.2. Edge Cases

*   **Empty Databases:** Handle the case where a database is empty or doesn't contain the expected properties.
*   **Deleted Pages:** Handle the case where a page has been deleted or moved.
*   **Permission Errors:**  Handle the case where the application doesn't have the required permissions to access a page or database.
*   **API Downtime:** Handle the case where the Notion API is temporarily unavailable.

### 6.3. Version-Specific Issues

*   **Breaking Changes:** Be aware of breaking changes in new versions of the Notion API. Test your application thoroughly after upgrading.
*   **Deprecated Features:** Avoid using deprecated features, as they may be removed in future versions.

### 6.4. Compatibility Concerns

*   **Library Conflicts:** Be aware of potential library conflicts between `notion-api` and other libraries in your project.
*   **Browser Compatibility:** Ensure that your application is compatible with the target browsers.

### 6.5. Debugging Strategies

*   **Use Debugging Tools:** Use debugging tools to inspect your code and identify errors.
*   **Log API Requests and Responses:** Log API requests and responses to help diagnose issues.
*   **Use a Network Monitor:** Use a network monitor to inspect network traffic and identify performance bottlenecks.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **IDE:** Visual Studio Code, IntelliJ IDEA, or WebStorm
*   **Node.js:**  A JavaScript runtime environment
*   **npm or Yarn:** Package managers
*   **Git:** Version control system
*   **Postman or Insomnia:** API testing tools

### 7.2. Build Configuration

*   **Use a Build Tool:** Use a build tool like Webpack or Parcel to bundle your code.
*   **Configure Environment Variables:** Configure environment variables to store API keys and other sensitive information.
*   **Use a .gitignore File:** Use a `.gitignore` file to exclude sensitive files from your repository.

### 7.3. Linting and Formatting

*   **Use a Linter:** Use a linter like ESLint or JSHint to enforce coding standards.
*   **Use a Formatter:** Use a formatter like Prettier to automatically format your code.
*   **Configure CI/CD:** Configure CI/CD to automatically lint and format your code.

### 7.4. Deployment

*   **Choose a Deployment Platform:** Choose a deployment platform that fits your application's needs (e.g., Vercel, Netlify, AWS).
*   **Configure Environment Variables:** Configure environment variables on your deployment platform to store API keys and other sensitive information.
*   **Monitor Your Application:** Monitor your application to identify and address performance issues and errors.

### 7.5. CI/CD Integration

*   **Use a CI/CD Platform:** Use a CI/CD platform like GitHub Actions or GitLab CI/CD to automate your build, test, and deployment processes.
*   **Run Tests in CI/CD:** Run your tests in your CI/CD pipeline to ensure that changes don't break existing functionality.
*   **Automate Deployment:** Automate deployment to reduce the risk of human error.