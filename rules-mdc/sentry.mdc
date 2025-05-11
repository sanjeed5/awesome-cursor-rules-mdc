---
description: This rule provides comprehensive best practices for integrating and utilizing Sentry in your projects. It covers code organization, performance, security, testing, and common pitfalls when using Sentry for error tracking and performance monitoring.
globs: *.js,*.jsx,*.ts,*.tsx,*.py,*.java,*.kt,*.swift,*.go,*.rb,*.php,*.cs,*.cpp,*.c,*.m,*.mm,*.clj,*.cljs,*.scala,*.groovy,*.lua,*.erl,*.hrl,*.fs,*.fsi,*.vb,*.vba,*.sh,*.bash,*.zsh,*.ps1,*.psm1,*.sql,*.html,*.htm,*.css,*.scss,*.less,*.graphql,*.gql
---
- Ensure that your codebase is connected to Sentry, with source code management integrations (e.g., GitHub, GitLab, Bitbucket) and source maps properly configured. This enables Sentry to link errors to specific lines of code, dramatically improving debugging efficiency.
- Implement custom alerts tailored to your needs, avoiding alert fatigue by prioritizing alerts based on severity or user impact. Set up different notification channels for varying alert priorities (e.g., Slack for low-priority, PagerDuty for high-priority).
- Leverage Sentry's performance monitoring capabilities to track application performance metrics and identify slow transactions. Use distributed tracing to visualize request flows and pinpoint performance bottlenecks.
- Implement alerts based on session-affected percentages when applicable to handle dynamic traffic.

## 1. Code Organization and Structure

- **Directory Structure:**
  - Group Sentry-related initialization and configuration files in a dedicated directory (e.g., `sentry/`) for easy management.
  - Consider separate files for Sentry SDK initialization, custom integrations, and alert configurations.
- **File Naming Conventions:**
  - Use descriptive names for Sentry-related files (e.g., `sentry.init.js`, `sentry.config.py`, `sentry.integrations.ts`).
- **Module Organization:**
  - Encapsulate Sentry-related functionality within dedicated modules or classes to promote code reusability and maintainability.
  - Avoid direct Sentry SDK calls throughout the application; instead, use an abstraction layer or helper functions.
- **Component Architecture:**
  - When using Sentry in UI frameworks (e.g., React, Vue), wrap key components with error boundaries to catch and report errors gracefully.
  - Create reusable components or hooks for common Sentry interactions, such as capturing user feedback or breadcrumbs.
- **Code Splitting:**
  - Lazy-load Sentry SDK and related code to minimize initial bundle size and improve application loading times.
  - Use dynamic imports or similar techniques to load Sentry code only when it's needed.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
  - **Singleton:** Use a singleton pattern for the Sentry client instance to ensure consistent configuration and behavior.
  - **Observer:** Implement an observer pattern to capture and report application-specific events to Sentry.
  - **Decorator:** Utilize decorators to automatically capture exceptions or performance metrics for specific functions or methods.
- **Recommended Approaches:**
  - Use breadcrumbs to capture user actions and application state leading up to an error, providing valuable context for debugging.
  - Set user context (ID, email, username) to associate errors with specific users, facilitating troubleshooting and impact assessment.
  - Capture release information (version, environment) to track errors across different deployments and environments.
- **Anti-patterns:**
  - Avoid hardcoding sensitive information (DSN, API keys) directly in the code; use environment variables or configuration files.
  - Do not capture personally identifiable information (PII) without proper anonymization or consent; adhere to privacy regulations.
  - Refrain from disabling Sentry entirely in production; instead, configure appropriate sampling rates or filters to manage data volume.
- **State Management:**
  - In applications using state management libraries (e.g., Redux, Vuex), integrate Sentry middleware to capture state changes and actions leading up to an error.
  - Serialize relevant state information and include it in the error context for debugging purposes.
- **Error Handling:**
  - Use try-catch blocks to handle potential errors gracefully and report them to Sentry.
  - Implement global error handlers to catch uncaught exceptions and unhandled rejections.
  - Provide informative error messages to users and avoid exposing sensitive implementation details.

## 3. Performance Considerations

- **Optimization Techniques:**
  - Enable compression for Sentry payloads to reduce network bandwidth and improve transmission speed.
  - Use Sentry's sampling options to control the number of events captured, balancing data accuracy with performance impact.
  - Configure Sentry's throttling options to prevent excessive event submission and potential performance bottlenecks.
- **Memory Management:**
  - Be mindful of memory usage when capturing large payloads or attachments; avoid unnecessary data serialization.
  - Periodically flush the Sentry event queue to release memory and prevent out-of-memory errors.
- **Rendering Optimization:**
  - For UI frameworks, defer Sentry initialization until after the initial render to avoid blocking the user interface.
  - Optimize breadcrumb and event data collection to minimize performance overhead during rendering.
- **Bundle Size Optimization:**
  - Tree-shake the Sentry SDK to remove unused modules and reduce bundle size.
  - Use code splitting to load Sentry code only when it's needed.
- **Lazy Loading:**
  - Implement lazy loading for Sentry components and modules to improve initial page load times.
  - Use dynamic imports or similar techniques to load Sentry code on demand.

## 4. Security Best Practices

- **Common Vulnerabilities:**
  - **Data Leakage:** Ensure that sensitive information (PII, API keys) is not inadvertently exposed in Sentry events or breadcrumbs.
  - **Cross-Site Scripting (XSS):** Sanitize user-generated content before capturing it in Sentry to prevent XSS attacks.
  - **Denial of Service (DoS):** Implement rate limiting and throttling to prevent excessive event submission from malicious actors.
- **Input Validation:**
  - Validate user input on both the client and server sides to prevent malicious data from reaching Sentry.
  - Sanitize and escape user input before capturing it in Sentry events or breadcrumbs.
- **Authentication and Authorization:**
  - Implement strong authentication and authorization mechanisms to protect Sentry API endpoints and data.
  - Restrict access to Sentry projects and events based on user roles and permissions.
- **Data Protection:**
  - Implement data anonymization and masking techniques to protect sensitive user information in Sentry events.
  - Comply with data privacy regulations (e.g., GDPR, CCPA) when collecting and processing user data with Sentry.
- **Secure API Communication:**
  - Use HTTPS for all communication with Sentry API endpoints to encrypt data in transit.
  - Verify the SSL/TLS certificates of Sentry API endpoints to prevent man-in-the-middle attacks.

## 5. Testing Approaches

- **Unit Testing:**
  - Write unit tests to verify the functionality of Sentry-related modules and components.
  - Mock or stub Sentry SDK functions to isolate the code under test.
- **Integration Testing:**
  - Perform integration tests to ensure that Sentry is properly integrated with the application and that events are captured correctly.
  - Use a test Sentry project to capture and verify events during integration testing.
- **End-to-End Testing:**
  - Conduct end-to-end tests to simulate real-world user scenarios and verify that errors are captured and reported to Sentry.
  - Use a staging environment or test Sentry project for end-to-end testing.
- **Test Organization:**
  - Organize Sentry-related tests in a dedicated directory or module.
  - Follow a consistent naming convention for Sentry-related test files and functions.
- **Mocking and Stubbing:**
  - Use mocking libraries (e.g., Jest, Sinon) to create mock Sentry SDK objects for testing purposes.
  - Stub Sentry SDK functions to control their behavior and verify that they are called correctly.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
  - Forgetting to initialize the Sentry SDK properly.
  - Capturing sensitive information without proper anonymization.
  - Overloading Sentry with too many events or attachments.
  - Ignoring Sentry alerts or dashboards.
- **Edge Cases:**
  - Handling errors during Sentry SDK initialization.
  - Dealing with network connectivity issues when sending events.
  - Managing Sentry SDK configuration in different environments.
- **Version-Specific Issues:**
  - Review Sentry SDK release notes for breaking changes and compatibility issues.
  - Test Sentry SDK upgrades in a staging environment before deploying to production.
- **Compatibility Concerns:**
  - Ensure compatibility between the Sentry SDK and other libraries or frameworks used in the application.
  - Resolve any conflicts or version mismatches that may arise.
- **Debugging Strategies:**
  - Use Sentry's debug mode to log SDK activity and troubleshoot integration issues.
  - Inspect Sentry events in the Sentry UI to verify data accuracy and identify potential problems.
  - Utilize Sentry's breadcrumbs and user context to gain insights into the cause of errors.

## 7. Tooling and Environment

- **Recommended Tools:**
  - Sentry CLI for managing Sentry projects and releases.
  - Source code management system (e.g., Git) for version control.
  - Package manager (e.g., npm, pip) for managing Sentry SDK dependencies.
- **Build Configuration:**
  - Integrate Sentry SDK into the build process to capture build errors and performance metrics.
  - Use environment variables or configuration files to manage Sentry SDK settings in different environments.
- **Linting and Formatting:**
  - Configure linters (e.g., ESLint, PyLint) to enforce code style and best practices for Sentry code.
  - Use code formatters (e.g., Prettier, Black) to automatically format Sentry code and ensure consistency.
- **Deployment:**
  - Deploy Sentry SDK and related code as part of the application deployment process.
  - Configure Sentry SDK to capture deployment information (version, environment) for tracking errors across releases.
- **CI/CD Integration:**
  - Integrate Sentry SDK into the CI/CD pipeline to automate error capture and reporting during testing and deployment.
  - Use Sentry's release tracking features to associate errors with specific deployments and releases.

By adhering to these best practices, you can effectively leverage Sentry to improve the reliability, performance, and security of your applications.