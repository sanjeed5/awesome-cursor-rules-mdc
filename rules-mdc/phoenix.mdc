---
description: This rule outlines the best practices and coding standards for developing Elixir applications with the Phoenix framework, covering code organization, performance, security, testing, and common pitfalls.
globs: *.ex,*.exs,*.eex,*.leex,*.sface
---
- # Phoenix Framework Best Practices

  This document outlines general best practices for developing Elixir applications with the Phoenix Framework. It aims to promote maintainable, scalable, and performant codebases.

- ## 1. Code Organization and Structure

  - ### 1.1. Directory Structure Best Practices

    - **`lib/`**: Contains the core application logic, including contexts, schemas, and supporting modules.
    - **`lib/<app_name>_web/`**: Holds the web-related components, such as controllers, views, templates, channels, and live views.
    - **`lib/<app_name>/`**: Contains your application’s core domain logic. Favor placing most of your code here.
    - **`test/`**: Contains all test-related files, including unit, integration, and acceptance tests.
    - **`priv/repo/migrations/`**: Stores database migration files.
    - **`config/`**: Contains configuration files for different environments (dev, test, prod).
    - **`assets/`**: Contains static assets like JavaScript, CSS, and images. Managed by tools like esbuild or Webpack.
    - **`deps/`**: Automatically generated folder containing project dependencies.
    - **`rel/`**: Used for building releases with distillery or mix release. Contains configuration for releases.

  - ### 1.2. File Naming Conventions

    - Use `snake_case` for file names (e.g., `user_controller.ex`, `user_schema.ex`).
    - Match the file name to the module name (e.g., `UserController` module should be in `user_controller.ex`).
    - For LiveView components, use `_live` suffix (e.g., `user_live.ex`).
    - For schema files, use `_schema` suffix(e.g. `user_schema.ex`).

  - ### 1.3. Module Organization

    - Organize modules into contexts, representing logical domains within the application (e.g., `Accounts`, `Blog`, `Payments`).
    - Keep modules small and focused, adhering to the Single Responsibility Principle (SRP).
    - Use namespaces to group related modules (e.g., `MyApp.Accounts.User`, `MyApp.Blog.Post`).
    - Favor explicit dependencies between modules to promote clarity and reduce coupling.

  - ### 1.4. Component Architecture (LiveView)

    - Break down complex LiveView pages into smaller, reusable components.
    - Use functional components (HEEx templates with function definitions) for simple UI elements.
    - Use LiveComponent modules for more complex, stateful components with event handling.
    - Organize components into directories based on their purpose or domain.
    - Define attributes and slots for LiveView components to ensure type safety.
    - Utilize PubSub for communication between LiveView components.

  - ### 1.5. Code Splitting Strategies

    - Separate concerns by using contexts. Controllers delegate to contexts, which handle the business logic, and contexts rely on schemas, which handle the data validations and structure.
    - Utilize umbrella projects to divide a large application into smaller, independent applications.
    - Consider code splitting at the LiveView level, using separate LiveView modules for different sections of a page.
    - Use dynamic imports for JavaScript modules to load code on demand.

- ## 2. Common Patterns and Anti-patterns

  - ### 2.1. Design Patterns

    - **MVC (Model-View-Controller)**: The core architectural pattern in Phoenix. Ensure proper separation of concerns between models (schemas), views (templates), and controllers.
    - **Context Pattern**: Encapsulate business logic and data access within contexts to provide a clear API for controllers and other parts of the application.
    - **Repository Pattern**: Abstract away database interactions behind a repository interface to allow for easier testing and potential database changes.  While contexts often serve this purpose in Phoenix, a dedicated repository layer can be useful for very complex data access logic.
    - **PubSub**: Use `Phoenix.PubSub` for real-time communication between different parts of the application (e.g., LiveView components, channels).
    - **Ecto Changesets**: Employ Ecto changesets for data validation and sanitization before persisting to the database.

  - ### 2.2. Recommended Approaches for Common Tasks

    - **Authentication**: Use a library like `Pow` or `Ueberauth` for handling user authentication and authorization.
    - **Authorization**: Implement role-based access control (RBAC) or attribute-based access control (ABAC) using libraries like `Pomegranate` or custom logic.
    - **Form Handling**: Use `Phoenix.HTML.Form` helpers and Ecto changesets for building and validating forms.
    - **Real-time Updates**: Leverage Phoenix Channels or LiveView for real-time updates and interactive features.
    - **Background Jobs**: Use `Oban` for reliable background job processing.

  - ### 2.3. Anti-patterns and Code Smells

    - **Fat Controllers**: Avoid putting too much logic in controllers. Delegate to contexts or services.
    - **Direct Repo Access in Controllers**: Do not directly call `Repo` functions in controllers. Use contexts.
    - **Ignoring Changeset Errors**: Always handle changeset errors and display appropriate messages to the user.
    - **Over-Complicated Queries**: Keep Ecto queries simple and composable. Use views or functions in schema modules to build more complex queries.
    - **Unnecessary Global State**: Minimize the use of global state. Prefer passing data explicitly between functions and components.

  - ### 2.4. State Management Best Practices (LiveView)

    - Store only the minimal required state in LiveView's `assigns`.
    - Use `handle_params` to load data based on URL parameters.
    - Employ `handle_info` for handling asynchronous events and updates.
    - Implement proper state cleanup in `mount` and `terminate` callbacks.
    - Consider using a global state management library like `global` only when truly necessary for cross-component communication.

  - ### 2.5. Error Handling Patterns

    - Use pattern matching to handle different error scenarios.
    - Raise exceptions only for truly exceptional cases.  For expected errors, return `{:error, reason}` tuples.
    - Implement error logging and monitoring using tools like `Sentry` or `Bugsnag`.
    - Use `try...rescue` blocks for handling potential exceptions during external API calls or file operations.
    - Define custom error types using atoms or structs to provide more context for error handling.

- ## 3. Performance Considerations

  - ### 3.1. Optimization Techniques

    - **Database Queries**: Optimize Ecto queries by using indexes, preloading associations, and avoiding N+1 queries.
    - **Caching**: Implement caching for frequently accessed data using `ETS` tables or a dedicated caching library like `Cachex`.
    - **Concurrency**: Leverage Elixir's concurrency features (e.g., `Task`, `GenServer`) to handle concurrent requests and background tasks efficiently.
    - **Code Profiling**: Use tools like `Erlang's observer` or `fprof` to identify performance bottlenecks in your code.
    - **Connection Pooling**: Ensure proper database connection pooling to minimize connection overhead.

  - ### 3.2. Memory Management

    - **Avoid Memory Leaks**: Be mindful of memory leaks, especially in long-running processes like `GenServers`.
    - **Use Streams**: Employ Elixir streams for processing large datasets in a memory-efficient manner.
    - **Garbage Collection**: Understand Erlang's garbage collection behavior and optimize code to minimize garbage collection overhead.

  - ### 3.3. Rendering Optimization (LiveView)

    - **Minimize DOM Updates**: Reduce the number of DOM updates in LiveView by using `phx-update` attributes and smart diffing.
    - **Use Static Content**: Serve static content (e.g., images, CSS, JavaScript) directly from the web server without involving LiveView.
    - **Optimize Templates**: Optimize HEEx templates for efficient rendering.

  - ### 3.4. Bundle Size Optimization

    - **Code Splitting**: Use code splitting to reduce the initial bundle size and load code on demand.
    - **Tree Shaking**: Enable tree shaking in esbuild or Webpack to remove unused code from the bundle.
    - **Minification**: Minify CSS and JavaScript files to reduce their size.
    - **Image Optimization**: Optimize images for web use by compressing them and using appropriate formats.

  - ### 3.5. Lazy Loading

    - **Lazy Load Images**: Implement lazy loading for images to improve initial page load time.
    - **Lazy Load Components**: Use dynamic imports or conditional rendering to lazy load LiveView components.

- ## 4. Security Best Practices

  - ### 4.1. Common Vulnerabilities and Prevention

    - **Cross-Site Scripting (XSS)**: Sanitize user input and use proper escaping in templates to prevent XSS attacks. Phoenix's HEEx templates automatically escape variables, but be careful when using `raw/1` or `safe/1`.
    - **Cross-Site Request Forgery (CSRF)**: Protect against CSRF attacks by using Phoenix's built-in CSRF protection mechanisms. Include the CSRF token in forms and AJAX requests.
    - **SQL Injection**: Use Ecto's parameterized queries to prevent SQL injection vulnerabilities.
    - **Authentication and Authorization Issues**: Implement robust authentication and authorization mechanisms to protect sensitive data and functionality.

  - ### 4.2. Input Validation

    - **Use Ecto Changesets**: Always validate user input using Ecto changesets to ensure data integrity and prevent malicious input.
    - **Whitelist Input**: Define a whitelist of allowed input values instead of a blacklist of disallowed values.
    - **Sanitize Input**: Sanitize user input to remove potentially harmful characters or code.

  - ### 4.3. Authentication and Authorization

    - **Use Strong Passwords**: Enforce strong password policies and use proper hashing algorithms (e.g., `bcrypt`) to store passwords securely.
    - **Implement Multi-Factor Authentication (MFA)**: Add an extra layer of security by requiring users to authenticate using multiple factors.
    - **Use JWT (JSON Web Tokens)**: Use JWT for secure API authentication and authorization.
    - **Implement Role-Based Access Control (RBAC)**: Define roles and permissions to control access to different parts of the application.

  - ### 4.4. Data Protection

    - **Encrypt Sensitive Data**: Encrypt sensitive data at rest and in transit using appropriate encryption algorithms.
    - **Use HTTPS**: Always use HTTPS to encrypt communication between the client and the server.
    - **Protect API Keys**: Store API keys securely and restrict access to them.

  - ### 4.5. Secure API Communication

    - **Use HTTPS**: Enforce HTTPS for all API communication.
    - **Validate API Requests**: Validate API requests to prevent malicious input and unauthorized access.
    - **Rate Limiting**: Implement rate limiting to prevent abuse and denial-of-service attacks.
    - **API Versioning**: Use API versioning to ensure backward compatibility and allow for future API changes.

- ## 5. Testing Approaches

  - ### 5.1. Unit Testing

    - **Test Contexts and Schemas**: Write unit tests for contexts and schemas to ensure that business logic and data validation work as expected.
    - **Mock External Dependencies**: Use mocking libraries like `Mock` or `Meck` to isolate units under test and avoid dependencies on external services.
    - **Test Edge Cases**: Test edge cases and boundary conditions to ensure that code handles unexpected input correctly.

  - ### 5.2. Integration Testing

    - **Test Controllers and Channels**: Write integration tests for controllers and channels to ensure that they interact correctly with contexts and other parts of the application.
    - **Use a Test Database**: Use a dedicated test database to avoid affecting the production database.
    - **Verify Database Interactions**: Verify that database interactions are performed correctly by inspecting the database state after running tests.

  - ### 5.3. End-to-End Testing

    - **Use Wallaby or Cypress**: Use end-to-end testing frameworks like Wallaby or Cypress to simulate user interactions and verify that the application works as expected from the user's perspective.
    - **Test Critical User Flows**: Test critical user flows to ensure that the most important features of the application are working correctly.

  - ### 5.4. Test Organization

    - **Organize Tests by Module**: Organize tests into directories that mirror the application's module structure.
    - **Use Descriptive Test Names**: Use descriptive test names to clearly indicate what each test is verifying.
    - **Keep Tests Small and Focused**: Keep tests small and focused to make them easier to understand and maintain.

  - ### 5.5. Mocking and Stubbing

    - **Use Mock Libraries**: Use mocking libraries like `Mock` or `Meck` to create mock objects and stub function calls.
    - **Avoid Over-Mocking**: Avoid over-mocking, as it can make tests less realistic and harder to maintain.
    - **Use Stubs for External Dependencies**: Use stubs to replace external dependencies with simplified versions that are easier to control during testing.

- ## 6. Common Pitfalls and Gotchas

  - ### 6.1. Frequent Mistakes

    - **Not Using Contexts**: Directly accessing the Repo in controllers or other modules, bypassing the business logic encapsulation provided by contexts.
    - **Ignoring Changeset Validation Errors**: Neglecting to handle and display changeset validation errors to the user, leading to confusing behavior.
    - **N+1 Queries**: Failing to preload associations in Ecto queries, resulting in performance bottlenecks.
    - **Over-reliance on Global State**: Using global state when it's not necessary, making code harder to reason about and test.

  - ### 6.2. Edge Cases

    - **Handling Timezones**: Dealing with timezones correctly, especially when storing and displaying dates and times to users in different locations.
    - **Concurrency Issues**: Avoiding race conditions and other concurrency issues when using Elixir's concurrency features.
    - **Large File Uploads**: Handling large file uploads efficiently and securely.

  - ### 6.3. Version-Specific Issues

    - **LiveView Updates**: Staying up-to-date with LiveView updates and understanding potential breaking changes.
    - **Ecto Version Compatibility**: Ensuring compatibility between Ecto and other dependencies.

  - ### 6.4. Compatibility Concerns

    - **JavaScript Framework Integration**: Integrating Phoenix with JavaScript frameworks like React or Vue.js can introduce compatibility issues.
    - **Database Driver Compatibility**: Ensuring compatibility between Ecto and the chosen database driver.

  - ### 6.5. Debugging Strategies

    - **Use IEx.pry**: Use `IEx.pry` to pause execution and inspect variables.
    - **Enable Debug Logging**: Enable debug logging to get more information about what's happening in the application.
    - **Use Remote Debugging**: Use remote debugging tools to debug applications running in production environments.
    - **Analyze Logs**: Analyze logs to identify errors and performance issues.

- ## 7. Tooling and Environment

  - ### 7.1. Recommended Development Tools

    - **Visual Studio Code**: VS Code with the ElixirLS extension provides excellent Elixir support, including code completion, linting, and debugging.
    - **IntelliJ IDEA**: IntelliJ IDEA with the Elixir plugin also provides excellent Elixir support.
    - **Mix**: Elixir's build tool, used for creating, compiling, testing, and managing dependencies.
    - **IEx**: Elixir's interactive shell, used for exploring code and debugging.
    - **Erlang Observer**: Erlang's GUI tool for monitoring and debugging Erlang/Elixir applications.

  - ### 7.2. Build Configuration

    - **Use `mix.exs`**: Use `mix.exs` to manage project dependencies, compilation settings, and other build configurations.
    - **Define Environments**: Define separate environments for development, testing, and production.
    - **Use Configuration Files**: Use configuration files (`config/config.exs`, `config/dev.exs`, `config/test.exs`, `config/prod.exs`) to configure the application for different environments.
    - **Secrets Management**: Use environment variables or dedicated secrets management tools to store sensitive information like API keys and database passwords.

  - ### 7.3. Linting and Formatting

    - **Use `mix format`**: Use `mix format` to automatically format Elixir code according to a consistent style.
    - **Use Credo**: Use Credo to analyze Elixir code for style and potential issues.
    - **Configure Editor**: Configure the editor to automatically format code on save.

  - ### 7.4. Deployment

    - **Use Mix Release**: Use `mix release` to build and deploy Elixir applications. Distillery is deprecated.
    - **Use Docker**: Use Docker to containerize Elixir applications for easy deployment and scaling.
    - **Deploy to Cloud Platforms**: Deploy Elixir applications to cloud platforms like Heroku, AWS, Google Cloud, or Azure.
    - **Use a Process Manager**: Use a process manager like `systemd` or `pm2` to manage Elixir application processes.

  - ### 7.5. CI/CD Integration

    - **Use GitHub Actions, GitLab CI, or CircleCI**: Use CI/CD platforms to automate the build, test, and deployment processes.
    - **Run Tests on CI**: Run tests on CI to ensure that code changes don't introduce regressions.
    - **Automate Deployments**: Automate deployments to production environments after successful tests.

- ## Additional Resources

  - [Phoenix Framework Documentation](https://www.phoenixframework.org/docs)
  - [Elixir Language Documentation](https://elixir-lang.org/docs.html)
  - [Ecto Documentation](https://hexdocs.pm/ecto/Ecto.html)
  - [Phoenix LiveView Documentation](https://hexdocs.pm/phoenix_live_view/Phoenix.LiveView.html)
  - [Oban Documentation](https://github.com/sorentwo/oban)
  - [Credo Documentation](https://hexdocs.pm/credo/Credo.html)