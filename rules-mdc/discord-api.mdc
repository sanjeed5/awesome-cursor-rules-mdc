---
description: This rule provides best practices and coding standards for developing applications with the discord-api library. It covers code organization, performance, security, testing, and common pitfalls to ensure robust and maintainable Discord integrations.
globs: *.js,*.ts,*.jsx,*.tsx,*.py,*.cs
---
## 1. Code Organization and Structure

-   **Directory Structure:**
    -   `src/`: Contains the main source code of your bot or application.
        -   `commands/`:  Houses individual command modules, each in its own file or subdirectory.
        -   `events/`: Stores event handlers for Discord events (e.g., message creation, member join).
        -   `models/`: Defines data models and schemas (if applicable).
        -   `services/`: Contains reusable services or utilities (e.g., database connections, API wrappers).
        -   `config/`: Configuration files (e.g., API keys, bot token).
        -   `utils/`: Utility functions and helper modules.
    -   `tests/`: Unit and integration tests.
    -   `docs/`: Documentation for your bot or application.
    -   `scripts/`:  Automation scripts (e.g., deployment, database setup).
-   **File Naming Conventions:**
    -   Use descriptive and consistent file names.
    -   Commands: `command_name.js`, `command_name.ts` or `command_name/index.js` for command directories.
    -   Events: `event_name.js`, `event_name.ts`.
    -   Models: `model_name.js`, `model_name.ts`.
    -   Services: `service_name.js`, `service_name.ts`.
    -   Configuration files: `config.json`, `config.js`, `config.ts`.
-   **Module Organization:**
    -   Break down your bot logic into smaller, reusable modules.
    -   Use ES modules ( `import/export` in JavaScript/TypeScript) or equivalent module systems in other languages to manage dependencies.
    -   Avoid circular dependencies between modules.
-   **Component Architecture:**
    -   **Command Handlers:** Create a dedicated module to handle command parsing, validation, and execution.
    -   **Event Emitters:** Use event emitters to decouple event handling logic from the core bot logic.
    -   **Service Layer:**  Abstract external services (e.g., databases, APIs) behind a service layer.
    -   **Configuration Management:** Use a configuration management library to handle application settings.
-   **Code Splitting:**
    -   Use dynamic imports to load command modules or event handlers on demand, reducing startup time.
    -   Bundle your code using tools like Webpack, Parcel, or Rollup to optimize bundle size.
    -   If applicable, use lazy loading for components or modules that are not immediately needed.

## 2. Common Patterns and Anti-patterns

-   **Design Patterns:**
    -   **Command Pattern:** Encapsulate commands as objects, allowing for flexible command management.
    -   **Observer Pattern:** Use event emitters to decouple event sources from event listeners.
    -   **Singleton Pattern:** Implement singleton instances for database connections or other shared resources.
    -   **Factory Pattern:** Create factories for creating Discord API objects (e.g., messages, embeds).
-   **Recommended Approaches:**
    -   Use a command framework (e.g., discord.js commands) to streamline command creation and handling.
    -   Utilize Discord's rate limiting mechanisms effectively.
    -   Implement robust error handling and logging.
    -   Store sensitive information (e.g., API keys, bot token) in environment variables.
-   **Anti-patterns and Code Smells:**
    -   **Global State:** Avoid using global variables to store bot state, as it can lead to unexpected behavior.
    -   **Hardcoding:** Do not hardcode configuration values or API keys in your code.
    -   **Nested Callbacks:** Avoid deeply nested callbacks, which can make your code difficult to read and maintain. Use async/await or promises instead.
    -   **Ignoring Errors:** Always handle errors properly and log them for debugging.
    -   **Over-Complicating:** Keep your code as simple as possible while still meeting the requirements.
-   **State Management:**
    -   Use a dedicated state management library (e.g., Redux, Zustand) for complex bot state.
    -   Store persistent data in a database (e.g., MongoDB, PostgreSQL).
    -   Implement caching to improve performance.
-   **Error Handling:**
    -   Use try-catch blocks to handle potential errors.
    -   Log errors to a file or service for debugging.
    -   Implement retry mechanisms for transient errors.
    -   Use Discord's error events to catch API errors.
    -   Send user-friendly error messages to users in the Discord channel.

## 3. Performance Considerations

-   **Optimization Techniques:**
    -   Use efficient data structures and algorithms.
    -   Cache frequently accessed data.
    -   Optimize database queries.
    -   Use asynchronous operations to avoid blocking the main thread.
    -   Shard your bot to distribute the load across multiple processes.
-   **Memory Management:**
    -   Avoid memory leaks by properly releasing resources.
    -   Use garbage collection to reclaim unused memory.
    -   Be mindful of large data structures that can consume a lot of memory.
-   **Rendering Optimization:**
    -   Optimize image sizes and formats.
    -   Use lazy loading for images and other media.
    -   Consider using optimized libraries for generating images or videos.
-   **Bundle Size Optimization:**
    -   Use tree shaking to remove unused code from your bundle.
    -   Minify your code to reduce bundle size.
    -   Compress your bundle using Gzip or Brotli.
-   **Lazy Loading:**
    -   Load command modules and event handlers on demand.
    -   Load images and other media only when they are needed.

## 4. Security Best Practices

-   **Common Vulnerabilities:**
    -   **Code Injection:** Prevent code injection vulnerabilities by validating user input and avoiding the use of `eval()` or similar functions.
    -   **Cross-Site Scripting (XSS):**  If your bot interacts with websites or web applications, prevent XSS vulnerabilities by sanitizing user input and escaping output.
    -   **SQL Injection:**  If your bot interacts with a database, prevent SQL injection vulnerabilities by using parameterized queries or an ORM.
    -   **Denial of Service (DoS):** Protect your bot from DoS attacks by implementing rate limiting and input validation.
    -   **Unauthorized Access:** Secure your bot's API endpoints and data by implementing authentication and authorization.
-   **Input Validation:**
    -   Validate all user input to prevent malicious or invalid data from being processed.
    -   Use regular expressions or other validation techniques to enforce data types and formats.
    -   Sanitize user input to remove potentially harmful characters or code.
-   **Authentication and Authorization:**
    -   Use a secure authentication mechanism to verify the identity of users.
    -   Implement authorization policies to control access to resources and functionality.
    -   Use role-based access control (RBAC) to manage user permissions.
-   **Data Protection:**
    -   Encrypt sensitive data at rest and in transit.
    -   Use secure storage mechanisms to protect API keys, bot tokens, and other credentials.
    -   Implement data loss prevention (DLP) measures to prevent sensitive data from being leaked.
-   **Secure API Communication:**
    -   Use HTTPS to encrypt communication with the Discord API.
    -   Verify the server certificate to prevent man-in-the-middle attacks.
    -   Use secure API keys or tokens to authenticate with the Discord API.

## 5. Testing Approaches

-   **Unit Testing:**
    -   Write unit tests for individual components and modules.
    -   Use mocking or stubbing to isolate components from external dependencies.
    -   Test edge cases and error conditions.
-   **Integration Testing:**
    -   Write integration tests to verify the interaction between different components.
    -   Test the bot's integration with the Discord API.
    -   Use a test Discord server to run integration tests.
-   **End-to-End Testing:**
    -   Write end-to-end tests to verify the entire bot workflow.
    -   Simulate user interactions to test the bot's functionality.
    -   Use a testing framework to automate end-to-end tests.
-   **Test Organization:**
    -   Organize your tests into separate directories for unit tests, integration tests, and end-to-end tests.
    -   Use descriptive test names to indicate the purpose of each test.
    -   Follow a consistent naming convention for test files.
-   **Mocking and Stubbing:**
    -   Use mocking libraries to create mock objects that simulate the behavior of external dependencies.
    -   Use stubbing to replace real dependencies with simplified versions for testing purposes.

## 6. Common Pitfalls and Gotchas

-   **Frequent Mistakes:**
    -   **Rate Limiting:**  Failing to handle Discord API rate limits properly.
    -   **Incorrect Intents:**  Not specifying the correct gateway intents for your bot.
    -   **Asynchronous Operations:**  Not properly handling asynchronous operations, leading to race conditions or unexpected behavior.
    -   **Data Serialization:** Incorrectly serializing or deserializing data when interacting with the API.
    -   **Privileged Intents:** Not understanding the requirements for using privileged intents (e.g., presence, guild members, message content).
-   **Edge Cases:**
    -   Handling large guilds with many members or channels.
    -   Dealing with unexpected API errors or outages.
    -   Handling different user locales and languages.
    -   Managing concurrency and race conditions.
-   **Version-Specific Issues:**
    -   Being aware of breaking changes between different versions of the discord-api library.
    -   Using deprecated features or APIs.
    -   Ensuring compatibility with different versions of Node.js or other dependencies.
-   **Compatibility Concerns:**
    -   Ensuring compatibility with different operating systems and platforms.
    -   Avoiding conflicts with other libraries or dependencies.
    -   Testing the bot on different Discord clients (e.g., web, desktop, mobile).
-   **Debugging Strategies:**
    -   Use logging to track the bot's execution flow and identify errors.
    -   Use a debugger to step through the code and inspect variables.
    -   Use Discord's developer tools to inspect API requests and responses.
    -   Test your code thoroughly and write unit tests to catch errors early.

## 7. Tooling and Environment

-   **Recommended Development Tools:**
    -   **IDE:** Visual Studio Code, IntelliJ IDEA, or other IDE with support for JavaScript/TypeScript.
    -   **Package Manager:** npm or yarn for managing dependencies.
    -   **Debugging Tools:** Node.js debugger or Chrome DevTools.
    -   **Linting Tools:** ESLint or JSHint for enforcing coding standards.
    -   **Formatting Tools:** Prettier for automatically formatting code.
-   **Build Configuration:**
    -   Use a build tool like Webpack, Parcel, or Rollup to bundle your code.
    -   Configure your build process to minify and compress your code.
    -   Use environment variables to configure your build process.
-   **Linting and Formatting:**
    -   Use ESLint or JSHint to enforce coding standards.
    -   Use Prettier to automatically format your code.
    -   Configure your editor to automatically run linters and formatters on save.
-   **Deployment Best Practices:**
    -   Use a process manager like PM2 or systemd to keep your bot running.
    -   Deploy your bot to a cloud platform like Heroku, AWS, or Google Cloud.
    -   Use a reverse proxy like Nginx or Apache to handle incoming requests.
    -   Monitor your bot's performance and health.
-   **CI/CD Integration:**
    -   Use a CI/CD platform like GitHub Actions, GitLab CI, or CircleCI to automate your build, test, and deployment process.
    -   Run unit tests and integration tests as part of your CI/CD pipeline.
    -   Automate the deployment process to reduce manual effort and errors.