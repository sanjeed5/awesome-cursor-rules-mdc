---
description: This rule file provides best practices and coding standards for ASP.NET Core development, covering various aspects from code organization to security and performance optimization.
globs: *.cs
---
# ASP.NET Core Best Practices and Coding Standards

This document outlines recommended best practices and coding standards for developing ASP.NET Core applications. Following these guidelines will help ensure code clarity, maintainability, performance, and security.

## 1. Code Organization and Structure

### Directory Structure Best Practices:

A well-defined directory structure is crucial for maintaining a scalable and organized ASP.NET Core project. Here's a suggested structure:


ProjectName/
├── src/
│   ├── ProjectName.Web/
│   │   ├── Controllers/
│   │   ├── Models/
│   │   ├── Views/
│   │   ├── Pages/ (for Razor Pages)
│   │   ├── wwwroot/
│   │   ├── appsettings.json
│   │   ├── Program.cs
│   │   ├── Startup.cs
│   ├── ProjectName.Data/
│   │   ├── Models/
│   │   ├── Contexts/
│   │   ├── Repositories/
│   │   ├── Migrations/
│   ├── ProjectName.Services/
│   │   ├── Interfaces/
│   │   ├── Implementations/
│   ├── ProjectName.Models/ (if different from Data Models)
├── tests/
│   ├── ProjectName.UnitTests/
│   ├── ProjectName.IntegrationTests/
├── artifacts/
├── .editorconfig
├── .gitignore
├── ProjectName.sln


*   **src:** Contains the source code of the application.
*   **ProjectName.Web:** The main web application project.  This can also be named `API` for web API projects.
*   **Controllers:** Contains ASP.NET Core controllers.
*   **Models:** Contains view models and other data transfer objects (DTOs) specific to the web layer.
*   **Views:** Contains Razor views (.cshtml files).
*   **Pages:** Contains Razor Pages (if using Razor Pages model).
*   **wwwroot:** Contains static files like CSS, JavaScript, and images.
*   **ProjectName.Data:** Contains data access logic, Entity Framework Core contexts, and models representing database entities.
*   **Models (Data):** Defines entity models that correspond to database tables.
*   **Contexts:**  Contains the DbContext class.
*   **Repositories:** Contains repositories for data access.
*   **Migrations:** Contains Entity Framework Core migrations.
*   **ProjectName.Services:** Contains business logic and application services.
*   **Interfaces:** Defines interfaces for services.
*   **Implementations:** Contains concrete implementations of service interfaces.
*   **ProjectName.Models:** (Optional) Contains domain models that are shared across multiple layers. This layer promotes separation of concerns if your data layer models should not be exposed at the API or Web layer.
*   **tests:** Contains unit and integration tests.
*   **ProjectName.UnitTests:** Contains unit tests.
*   **ProjectName.IntegrationTests:** Contains integration tests.
*   **artifacts:**  Contains build artifacts.
*   **.editorconfig:** Defines code style rules.
*   **.gitignore:** Specifies intentionally untracked files that Git should ignore.
*   **ProjectName.sln:** The solution file.

### File Naming Conventions:

*   **Classes:** PascalCase (e.g., `UserController`, `ProductService`)
*   **Interfaces:** IPascalCase (e.g., `IProductService`, `IUserRepository`)
*   **Methods:** PascalCase (e.g., `GetProduct`, `CreateUser`)
*   **Variables (local):** camelCase (e.g., `productName`, `userId`)
*   **Parameters:** camelCase (e.g., `productId`, `userName`)
*   **Constants:** PascalCase (e.g., `DefaultPageSize`, `MaxRetries`)
*   **Enums:** PascalCase (e.g., `OrderStatus`, `UserRole`)
*   **Namespaces:** PascalCase, mirroring the directory structure (e.g., `ProjectName.Web.Controllers`, `ProjectName.Services.Implementations`)
*   **Files:** Match the class/interface name (e.g., `UserController.cs`, `IProductService.cs`)

### Module Organization:

*   **Feature-based Modules:** Group related functionality into modules based on features (e.g., `Authentication`, `Products`, `Orders`).  Each feature can have its own folder within the `src` directory, containing its own controllers, models, services, and data access components.  This enhances cohesion and reduces coupling.
*   **Vertical Slices:** Organize modules around specific use cases or user stories.  Each slice contains all the code necessary to implement a particular feature, from the UI to the data access layer. This promotes faster development and easier maintenance.

### Component Architecture:

*   **Layered Architecture:** Divide the application into layers (e.g., Presentation, Application, Domain, Infrastructure) to separate concerns and promote testability. Each layer has a specific responsibility and interacts with other layers through well-defined interfaces.
*   **Microservices:**  For larger applications, consider breaking down the application into smaller, independent microservices. Each microservice handles a specific business capability and can be deployed and scaled independently.

### Code Splitting Strategies:

*   **Feature-based Splitting:** Split code based on features, loading modules only when they are needed. This can improve initial load time and reduce memory consumption.
*   **Route-based Splitting:** Load code only when a specific route is accessed. This is especially useful for large applications with many routes.
*   **On-Demand Loading:** Load code dynamically when a user interacts with a specific component. This can improve perceived performance and reduce the initial download size.

## 2. Common Patterns and Anti-patterns

### Design Patterns:

*   **Dependency Injection (DI):** Use the built-in DI container to manage dependencies and promote loose coupling.
*   **Repository Pattern:** Abstract data access logic behind repositories to improve testability and maintainability. Avoid exposing EF Core `IQueryable` outside of the repository.
*   **Unit of Work Pattern:** Encapsulate multiple operations into a single transaction to ensure data consistency.
*   **CQRS (Command Query Responsibility Segregation):** Separate read and write operations into separate models to optimize performance and scalability.
*   **Mediator Pattern:** Decouple components by routing messages through a central mediator.  Libraries like MediatR are excellent for this.
*   **Specification Pattern:**  Encapsulate complex query logic into reusable specification objects.
*   **Strategy Pattern:** Define a family of algorithms, encapsulate each one, and make them interchangeable.

### Recommended Approaches for Common Tasks:

*   **Configuration:** Use the `IOptions<T>` pattern to access configuration settings in a strongly-typed manner.
*   **Logging:** Use the `ILogger<T>` interface for logging. Configure logging providers in `appsettings.json` and leverage structured logging.
*   **Exception Handling:** Implement global exception handling middleware to handle unhandled exceptions and return appropriate error responses.
*   **Data Validation:** Use data annotations and the `ModelState` property for validating user input. Implement custom validation attributes for complex validation rules.
*   **Asynchronous Programming:** Use `async` and `await` for I/O-bound operations to avoid blocking threads.
*   **HTTP Requests:** Utilize `HttpClientFactory` for creating and managing `HttpClient` instances.
*   **Background Tasks:** Use hosted services or background worker services for running long-running tasks asynchronously.

### Anti-patterns and Code Smells:

*   **Tight Coupling:** Avoid tight coupling between components. Use interfaces and dependency injection to promote loose coupling.
*   **God Classes:** Avoid creating large classes with too many responsibilities. Break down large classes into smaller, more manageable classes.
*   **Shotgun Surgery:** Avoid making changes to multiple classes when a single responsibility changes. This indicates poor separation of concerns.
*   **Primitive Obsession:** Avoid using primitive types to represent domain concepts. Create value objects to encapsulate domain logic and validation rules.
*   **Feature Envy:** Avoid classes that access the data of another class more than their own. Move the method to the class where the data resides.
*   **Sync-over-Async:**  Avoid using `Task.Wait()` or `Task.Result` in asynchronous methods, as this can lead to thread pool starvation. Always use `await`.
*   **Ignoring Exceptions:** Avoid catching exceptions and not logging or handling them appropriately.
*   **Magic Strings/Numbers:** Avoid hardcoding values directly in the code.  Use constants or configuration settings instead.
*   **Returning raw Entities:**  Do not expose EF Core entities directly to the client.  Create DTOs or View Models to represent the data that needs to be returned.

### State Management Best Practices:

*   **Stateless Controllers:** Design controllers to be stateless. Avoid storing state in controller fields.
*   **Session State:** Use session state sparingly. Session state can impact scalability. Consider using a distributed cache for session state.
*   **Cookies:** Use cookies for storing small amounts of non-sensitive data. Be mindful of cookie size limits and security concerns.
*   **TempData:** Use TempData for passing data between redirects.  TempData is stored in session or cookies.
*   **Caching:** Use caching (memory cache, distributed cache, response caching) to improve performance and reduce database load.  Cache aggressively, but invalidate appropriately.

### Error Handling Patterns:

*   **Global Exception Handling:** Implement global exception handling middleware to catch unhandled exceptions and return consistent error responses.
*   **Exception Filters:** Use exception filters to handle exceptions at the controller level.
*   **Try-Catch Blocks:** Use try-catch blocks for handling expected exceptions.
*   **Problem Details:** Return problem details (RFC 7807) in error responses to provide more detailed information about the error.
*   **Idempotency:** Design APIs to be idempotent where possible, so that retries are safe.

## 3. Performance Considerations

### Optimization Techniques:

*   **Asynchronous Programming:** Use `async` and `await` for I/O-bound operations.
*   **Caching:** Implement caching strategies to reduce database load and improve response times.
*   **Response Compression:** Enable response compression to reduce the size of HTTP responses.
*   **Minification and Bundling:** Minify and bundle CSS and JavaScript files to reduce the number of HTTP requests and improve load times.
*   **Image Optimization:** Optimize images to reduce their file size without sacrificing quality.
*   **Database Optimization:** Optimize database queries and indexing to improve data access performance.
*   **Connection Pooling:** Use connection pooling to reuse database connections and reduce connection overhead.
*   **HTTP/2:** Use HTTP/2 for improved performance and reduced latency.
*   **Precompiled Views:**  Precompile Razor views to improve startup time. 
*   **Use Span<T> and Memory<T>:** Utilize these types for efficient memory manipulation, reducing allocations and copies.
*   **Object Pooling:** Employ object pooling for frequently created and destroyed objects to reduce garbage collection overhead. `ArrayPool<T>` is especially effective for large arrays.
*   **Optimize LINQ Usage:** Use LINQ effectively, avoiding unnecessary iterations and allocations. Utilize `AsNoTracking()` for read-only queries and be mindful of client-side evaluation.
*   **Use System.Text.Json:** Prefer `System.Text.Json` over `Newtonsoft.Json` where possible, as it offers better performance and is designed for UTF-8 text.

### Memory Management:

*   **Minimize Object Allocations:** Reduce object allocations, especially in hot code paths. Consider using object pooling and value types.
*   **Avoid Large Objects:** Avoid allocating large objects (>= 85,000 bytes) as they are stored on the large object heap and can cause performance issues.
*   **Dispose of Resources:** Dispose of disposable resources (e.g., database connections, file streams) properly to prevent memory leaks.
*   **Garbage Collection:** Understand how the garbage collector works and avoid patterns that trigger frequent garbage collections.

### Rendering Optimization:

*   **Partial Views:** Use partial views to reuse code and reduce code duplication.
*   **View Components:** Use view components to encapsulate complex rendering logic.
*   **Tag Helpers:** Use tag helpers to create reusable UI components.
*   **Client-Side Rendering:** Consider using client-side rendering frameworks (e.g., React, Angular, Vue.js) for complex UI interactions.
*   **Lazy Loading:** Implement lazy loading for images and other assets to improve initial load time.

### Bundle Size Optimization:

*   **Tree Shaking:** Use tree shaking to remove unused code from JavaScript bundles.
*   **Code Splitting:** Split code into smaller chunks that can be loaded on demand.
*   **Compression:** Compress JavaScript and CSS files using gzip or Brotli.
*   **Dead Code Elimination:** Remove unused code from the application.

### Lazy Loading Strategies:

*   **Lazy Loading of Images:** Load images only when they are visible in the viewport.
*   **Lazy Loading of Modules:** Load modules only when they are needed.
*   **Lazy Loading of Data:** Load data only when it is requested.

## 4. Security Best Practices

### Common Vulnerabilities and Prevention:

*   **SQL Injection:** Use parameterized queries or ORMs to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Sanitize user input and encode output to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Use anti-forgery tokens to prevent CSRF attacks.
*   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms to protect sensitive data and functionality.
*   **Clickjacking:** Protect against clickjacking attacks by using the X-Frame-Options header.
*   **Denial of Service (DoS):** Implement rate limiting and other measures to prevent DoS attacks.
*   **Open Redirects:**  Validate redirect URLs to prevent open redirect vulnerabilities.
*   **Insecure Direct Object References (IDOR):** Implement proper authorization checks to prevent unauthorized access to resources.

### Input Validation:

*   **Server-Side Validation:** Always validate user input on the server-side.
*   **Whitelisting:** Use whitelisting to allow only valid characters and data formats.
*   **Data Annotations:** Use data annotations to validate model properties.
*   **Custom Validation:** Implement custom validation attributes for complex validation rules.
*   **Sanitization:** Sanitize user input to remove potentially harmful characters.

### Authentication and Authorization:

*   **Authentication Schemes:** Choose an appropriate authentication scheme (e.g., cookies, JWT, OAuth2) based on the application requirements.
*   **Authorization Policies:** Use authorization policies to define access control rules.
*   **Role-Based Access Control (RBAC):** Implement RBAC to grant users access to specific resources based on their roles.
*   **Claims-Based Authorization:** Use claims-based authorization to grant users access to specific resources based on their claims.
*   **Secure Password Storage:** Use strong password hashing algorithms (e.g., bcrypt, Argon2) to store passwords securely.
*   **Multi-Factor Authentication (MFA):** Implement MFA to enhance security.
*   **Implement OWASP Recommendations:** Follow the OWASP (Open Web Application Security Project) guidelines for secure authentication and authorization.

### Data Protection:

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Protection API:** Use the Data Protection API to protect sensitive data.
*   **Key Management:** Implement proper key management practices to protect encryption keys.
*   **HTTPS:** Use HTTPS to encrypt communication between the client and the server.

### Secure API Communication:

*   **API Keys:** Use API keys to authenticate API clients.
*   **OAuth 2.0:** Use OAuth 2.0 to authorize API clients.
*   **JWT (JSON Web Tokens):** Use JWT for secure API authentication and authorization.
*   **Rate Limiting:** Implement rate limiting to prevent API abuse.
*   **Input Validation:** Implement proper input validation on the server-side to prevent injection attacks.
*   **Output Encoding:** Encode API responses to prevent XSS attacks.
*   **CORS (Cross-Origin Resource Sharing):** Configure CORS to allow only authorized domains to access the API.

## 5. Testing Approaches

### Unit Testing:

*   **Test-Driven Development (TDD):** Write unit tests before writing the code.
*   **Arrange, Act, Assert (AAA):** Follow the AAA pattern in unit tests.
*   **Mocking:** Use mocking frameworks (e.g., Moq, NSubstitute) to isolate the code being tested.
*   **Code Coverage:** Aim for high code coverage.
*   **Focus on Business Logic:** Unit test the business logic of the application.

### Integration Testing:

*   **Test Data Access:** Test the interaction between the application and the database.
*   **Test External Services:** Test the interaction between the application and external services.
*   **Use a Test Database:** Use a separate test database for integration tests.
*   **Seed the Database:** Seed the test database with test data before running integration tests.

### End-to-End Testing:

*   **Test the Entire Application Flow:** Test the entire application flow from the user interface to the database.
*   **Use Automation Tools:** Use automation tools (e.g., Selenium, Cypress) to automate end-to-end tests.
*   **Test in a Production-Like Environment:** Test in a production-like environment to ensure that the application works as expected in production.

### Test Organization:

*   **Separate Test Projects:** Create separate test projects for unit tests, integration tests, and end-to-end tests.
*   **Organize Tests by Feature:** Organize tests by feature or module.
*   **Use Naming Conventions:** Use consistent naming conventions for test classes and methods.

### Mocking and Stubbing:

*   **Use Mocking Frameworks:** Use mocking frameworks to create mock objects and stubs.
*   **Mock Dependencies:** Mock external dependencies to isolate the code being tested.
*   **Verify Interactions:** Verify that the code being tested interacts with its dependencies as expected.
*   **Avoid Over-Mocking:** Avoid mocking too many dependencies. Focus on mocking external dependencies and complex objects.

## 6. Common Pitfalls and Gotchas

### Frequent Mistakes:

*   **Incorrect Dependency Injection Configuration:** Incorrectly configuring dependency injection can lead to runtime errors.
*   **Not Handling Exceptions Properly:** Not handling exceptions properly can lead to application crashes.
*   **Blocking Calls:** Making blocking calls in asynchronous methods can lead to thread pool starvation.
*   **Incorrect Data Validation:** Incorrect data validation can lead to security vulnerabilities and data corruption.
*   **Over-Engineering:** Over-engineering the application can lead to unnecessary complexity.
*   **Premature Optimization:** Optimizing the application too early can lead to wasted effort and code that is difficult to maintain.
*   **Ignoring Security Best Practices:** Ignoring security best practices can lead to security vulnerabilities.
*   **Not Writing Tests:** Not writing tests can lead to code that is difficult to maintain and prone to errors.
*   **Leaking Database Connections:** Failing to properly dispose of database connections can exhaust connection resources and cause performance issues.
*   **Using Sync over Async:** Blocking asynchronous operations can lead to thread pool starvation and performance degradation.
*   **Not using HttpClientFactory correctly:** Improper disposal of `HttpClient` instances leads to socket exhaustion. Always use `HttpClientFactory`.

### Edge Cases:

*   **Handling Null Values:** Handle null values properly to prevent null reference exceptions.
*   **Handling Empty Collections:** Handle empty collections properly to prevent unexpected behavior.
*   **Handling Large Data Sets:** Handle large data sets efficiently to prevent performance issues.
*   **Handling Time Zones:** Handle time zones correctly to prevent data inconsistencies.

### Version-Specific Issues:

*   **Breaking Changes:** Be aware of breaking changes between different versions of ASP.NET Core.
*   **Deprecated Features:** Be aware of deprecated features and plan to migrate to newer features.
*   **Performance Improvements:** Take advantage of performance improvements in newer versions of ASP.NET Core.
*   **Security Patches:** Stay up-to-date with security patches to protect against vulnerabilities.

### Compatibility Concerns:

*   **.NET Framework vs .NET Core:** Be aware of the differences between .NET Framework and .NET Core and choose the appropriate framework for the application.
*   **Operating System Compatibility:** Ensure that the application is compatible with the target operating system.
*   **Browser Compatibility:** Ensure that the application is compatible with the target browsers.

### Debugging Strategies:

*   **Logging:** Use logging to track down errors and unexpected behavior.
*   **Debugging Tools:** Use debugging tools (e.g., Visual Studio debugger) to step through the code and examine variables.
*   **Profiling Tools:** Use profiling tools to identify performance bottlenecks.
*   **Remote Debugging:** Use remote debugging to debug applications running on remote servers.
*   **Diagnostic Tools:** Use diagnostic tools (e.g., Application Insights) to monitor the health and performance of the application.

## 7. Tooling and Environment

### Recommended Development Tools:

*   **Visual Studio:** A comprehensive IDE for developing ASP.NET Core applications.
*   **Visual Studio Code:** A lightweight and versatile code editor for developing ASP.NET Core applications.
*   **.NET CLI:** A command-line tool for creating, building, and running ASP.NET Core applications.
*   **NuGet Package Manager:** A package manager for managing dependencies in ASP.NET Core applications.
*   **Postman:** A tool for testing APIs.
*   **Fiddler:** A tool for debugging HTTP traffic.

### Build Configuration:

*   **Use a Build Server:** Use a build server (e.g., Azure DevOps, Jenkins) to automate the build process.
*   **Configure Build Configurations:** Configure build configurations (e.g., Debug, Release) to optimize the application for different environments.
*   **Use NuGet Package Restore:** Use NuGet package restore to automatically download dependencies during the build process.
*   **Version Control:** Use version control (e.g., Git) to track changes to the code.

### Linting and Formatting:

*   **.editorconfig:** Use an .editorconfig file to define code style rules.
*   **Code Analyzers:** Use code analyzers to enforce code style rules and detect potential errors.
*   **StyleCop:** Use StyleCop to enforce coding style guidelines.
*   **ReSharper:** Use ReSharper to improve code quality and productivity.
*   **Format on Save:** Configure the IDE to format code on save.

### Deployment Best Practices:

*   **Choose a Hosting Environment:** Choose an appropriate hosting environment (e.g., Azure, AWS, on-premises server) based on the application requirements.
*   **Configure Application Settings:** Configure application settings (e.g., connection strings, API keys) for the production environment.
*   **Use HTTPS:** Use HTTPS to encrypt communication between the client and the server.
*   **Enable Logging:** Enable logging to track errors and unexpected behavior in the production environment.
*   **Monitor the Application:** Monitor the health and performance of the application in the production environment.
*   **Use a Deployment Pipeline:** Automate the deployment process with a CI/CD pipeline.
*   **Consider Containerization:** Deploy your application using containers (e.g., Docker) to ensure consistency across environments.
*   **Implement Health Checks:** Use ASP.NET Core's built-in health checks endpoint to monitor the health of your application.

### CI/CD Integration:

*   **Use a CI/CD Pipeline:** Use a CI/CD pipeline (e.g., Azure DevOps, GitHub Actions, GitLab CI) to automate the build, test, and deployment process.
*   **Automate Testing:** Automate unit tests, integration tests, and end-to-end tests in the CI/CD pipeline.
*   **Automate Deployment:** Automate the deployment process in the CI/CD pipeline.
*   **Implement Rollbacks:** Implement rollbacks to quickly revert to a previous version of the application if there are problems with the new deployment.