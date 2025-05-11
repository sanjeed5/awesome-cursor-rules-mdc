---
description: This rule file provides a comprehensive guide to C# best practices, coding standards, and common patterns for writing maintainable, performant, and secure code.
globs: *.cs
---
# C# Best Practices and Coding Standards

This document provides a comprehensive guide to C# best practices, coding standards, and common patterns for writing maintainable, performant, and secure code. It covers various aspects of C# development, including code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and tooling.

**Library Information:**
- Name: c-sharp
- Tags: language, microsoft, dotnet, backend

## 1. Code Organization and Structure

A well-organized codebase is crucial for maintainability, scalability, and collaboration. Here are some best practices for organizing your C# code:

### 1.1. Directory Structure Best Practices

*   **Project Root:** Contains the solution file (.sln) and project directories.
*   **Project Directory:** Contains the project file (.csproj), source code, and other project-related files.
    *   `src/`: Contains the main source code.
        *   `Models/`: Data models and DTOs (Data Transfer Objects).
        *   `Services/`: Business logic and service classes.
        *   `Controllers/`: API controllers (if applicable).
        *   `Repositories/`: Data access logic.
        *   `Utilities/`: Helper classes and utility functions.
        *   `Exceptions/`: Custom exception definitions.
        *   `Interfaces/`: Interface definitions for abstraction.
        *   `Configuration/`: Configuration-related classes.
    *   `tests/`: Contains unit tests, integration tests, and end-to-end tests.
    *   `docs/`: Documentation for the project.
    *   `build/`: Build scripts and configuration files.
    *   `Properties/`: Assembly information and project settings.

Example:


MyProject/
├── MyProject.sln
├── MyProject/
│   ├── MyProject.csproj
│   ├── src/
│   │   ├── Models/
│   │   ├── Services/
│   │   ├── Controllers/
│   │   ├── Repositories/
│   │   ├── Utilities/
│   │   ├── Exceptions/
│   │   ├── Interfaces/
│   │   └── Configuration/
│   ├── Properties/
│   │   └── AssemblyInfo.cs
│   └── appsettings.json
├── MyProject.Tests/
│   ├── MyProject.Tests.csproj
│   └── UnitTests/
└── README.md


### 1.2. File Naming Conventions

*   **Classes:** PascalCase (e.g., `MyClass.cs`)
*   **Interfaces:** IPascalCase (e.g., `IMyInterface.cs`)
*   **Enums:** PascalCase (e.g., `MyEnum.cs`)
*   **Structs:** PascalCase (e.g., `MyStruct.cs`)
*   **Delegates:** PascalCase (e.g., `MyDelegate.cs`)
*   **Configuration Files:** appsettings.json, config.xml
*   **Test Files:** `MyClassTests.cs`

### 1.3. Module Organization

*   **Namespaces:** Use namespaces to group related classes and interfaces.  Follow a consistent naming convention for namespaces (e.g., `CompanyName.ProjectName.ModuleName`).
*   **Assemblies:**  Divide large projects into multiple assemblies to improve build times, reduce dependencies, and enable code reuse. Consider functional or domain boundaries when creating assemblies.
*   **NuGet Packages:** Use NuGet packages to manage dependencies and share code across projects.

### 1.4. Component Architecture

*   **Layered Architecture:** Separate the application into distinct layers (e.g., presentation, business logic, data access) to promote separation of concerns and testability.
*   **Microservices Architecture:** For large and complex applications, consider a microservices architecture where the application is composed of small, independent services.
*   **Dependency Injection (DI):** Use DI to manage dependencies between components and improve testability and maintainability. Popular DI containers include Autofac, Ninject, and Microsoft.Extensions.DependencyInjection.

### 1.5. Code Splitting Strategies

*   **By Feature:** Group code related to a specific feature into a separate module or assembly.
*   **By Layer:** Separate code based on the architectural layer (e.g., presentation, business logic, data access).
*   **By Responsibility:**  Split classes and methods into smaller, more focused units of work.
*   **Partial Classes:** Use partial classes to split a large class into multiple files for better organization (use sparingly).

## 2. Common Patterns and Anti-patterns

Understanding common design patterns and anti-patterns is essential for writing effective and maintainable C# code.

### 2.1. Design Patterns

*   **Singleton:** Ensures that a class has only one instance and provides a global point of access to it.
*   **Factory:** Provides an interface for creating objects without specifying their concrete classes.
*   **Abstract Factory:** Provides an interface for creating families of related objects without specifying their concrete classes.
*   **Builder:** Separates the construction of a complex object from its representation, allowing the same construction process to create different representations.
*   **Observer:** Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
*   **Strategy:** Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
*   **Dependency Injection (DI):**  A technique whereby one object (or static method) supplies the dependencies of another object. This helps to decouple components.
*   **Repository:** Mediates between the domain and data mapping layers, acting like an in-memory domain object collection.
*   **Unit of Work:**  Maintains a list of objects affected by a business transaction and coordinates the writing out of changes.
*   **Asynchronous Programming Patterns (TAP, EAP, APM):**  Handle asynchronous operations efficiently using async/await.

### 2.2. Recommended Approaches for Common Tasks

*   **String Manipulation:** Use `StringBuilder` for efficient string concatenation in loops.
*   **File I/O:** Use `using` statements or `try-finally` blocks to ensure proper disposal of file resources.
*   **Data Access:** Use ORMs like Entity Framework Core or Dapper for simplified data access.
*   **Asynchronous Operations:** Use `async` and `await` for non-blocking asynchronous operations.
*   **Configuration Management:** Use `Microsoft.Extensions.Configuration` for managing application configuration.
*   **Logging:** Use logging frameworks like Serilog or NLog for structured logging.

### 2.3. Anti-patterns and Code Smells

*   **God Class:** A class that does too much and has too many responsibilities.
*   **Long Method:** A method that is too long and complex.
*   **Feature Envy:** A method that accesses data from another object more than its own.
*   **Shotgun Surgery:** Changes to one part of the code require changes to many other parts.
*   **Data Clump:** Groups of data that appear together in multiple places.
*   **Primitive Obsession:** Using primitive types instead of creating custom classes for domain concepts.
*   **Switch Statements (over Polymorphism):** Using large switch statements instead of leveraging polymorphism.
*   **Magic Numbers/Strings:** Hardcoding values directly in the code instead of using constants or configuration settings.
*   **Ignoring Exceptions:** Catching exceptions without handling them properly.
*   **Empty Catch Blocks:** Catching exceptions and doing nothing.
*   **Over-commenting:** Writing excessive comments that are obvious from the code itself.
*   **Dead Code:** Code that is never executed.

### 2.4. State Management Best Practices

*   **Stateless Services:** Design services to be stateless whenever possible to improve scalability and reliability.
*   **Session State:** Use session state sparingly and only when necessary. Consider using distributed caching for session state in web applications.
*   **Caching:** Use caching to improve performance by storing frequently accessed data in memory.
*   **Redux/Flux:** For complex UI applications, consider using a state management library like Redux or Flux.
*   **Immutable Data Structures:** Use immutable data structures to simplify state management and prevent unintended side effects.

### 2.5. Error Handling Patterns

*   **Try-Catch-Finally:** Use `try-catch-finally` blocks to handle exceptions and ensure proper resource cleanup.
*   **Exception Filters:** Use exception filters to catch specific exceptions based on certain conditions.
*   **Custom Exceptions:** Create custom exception types for specific error conditions in your application.
*   **Logging Exceptions:** Log exceptions with sufficient context to aid in debugging.
*   **Graceful Degradation:** Handle errors gracefully and provide informative error messages to the user.
*   **Throw Early, Catch Late:** Detect errors as early as possible and handle exceptions at a higher level.

## 3. Performance Considerations

Optimizing C# code for performance is crucial for creating responsive and efficient applications.

### 3.1. Optimization Techniques

*   **Avoid Boxing and Unboxing:** Boxing and unboxing value types can be expensive. Use generics to avoid boxing and unboxing.
*   **Use Value Types When Appropriate:** Value types (structs) can be more efficient than reference types (classes) for small, immutable data structures.
*   **Minimize Object Allocation:** Object allocation can be expensive. Reuse objects whenever possible.
*   **Use `StringBuilder` for String Concatenation:** `StringBuilder` is more efficient than string concatenation using the `+` operator, especially in loops.
*   **Optimize LINQ Queries:** Use LINQ carefully and avoid unnecessary iterations or computations.  Consider using `AsParallel()` for parallel processing of large collections (with caution, as parallelism introduces complexity).
*   **Use Asynchronous Programming:** Use `async` and `await` to avoid blocking the main thread and improve responsiveness.
*   **Avoid Excessive Locking:** Minimize the use of locks to prevent contention and improve concurrency.
*   **Use Lazy Initialization:** Initialize objects only when they are needed to avoid unnecessary initialization overhead.
*   **Profile Your Code:** Use profiling tools to identify performance bottlenecks and optimize accordingly.  Visual Studio Profiler, dotTrace, and PerfView are good options.

### 3.2. Memory Management Considerations

*   **Garbage Collection:** Understand how the garbage collector works and avoid creating excessive garbage.
*   **Dispose of Resources:** Implement the `IDisposable` interface and use `using` statements to ensure proper disposal of resources (e.g., file streams, database connections).
*   **Weak References:** Use weak references to hold references to objects without preventing them from being garbage collected.
*   **Object Pooling:** Use object pooling to reuse objects and reduce allocation overhead.
*   **Large Object Heap (LOH):** Be aware of the Large Object Heap and avoid allocating large objects unnecessarily.

### 3.3. Rendering Optimization (If Applicable)

*   **UI Virtualization:** Use UI virtualization to render only the visible items in a large list or grid.
*   **Reduce Overdraw:** Minimize the number of times pixels are drawn on top of each other.
*   **Batch Rendering:** Batch rendering operations to reduce the number of draw calls.
*   **Use Hardware Acceleration:** Use hardware acceleration to offload rendering tasks to the GPU.

### 3.4. Bundle Size Optimization (If Applicable)

*   **Tree Shaking:** Remove unused code from the bundle.
*   **Code Minification:** Minify code to reduce its size.
*   **Code Compression:** Compress code using Gzip or Brotli.
*   **Image Optimization:** Optimize images to reduce their size without sacrificing quality.

### 3.5. Lazy Loading Strategies

*   **Lazy Initialization:** Use `Lazy<T>` to initialize objects only when they are accessed.
*   **Virtual Proxy:** Use a virtual proxy to load related data only when it is needed.
*   **Explicit Loading:** Load related data explicitly using methods like `Include` in Entity Framework Core.

## 4. Security Best Practices

Security should be a primary concern in C# development to protect against vulnerabilities and attacks.

### 4.1. Common Vulnerabilities and Prevention

*   **SQL Injection:** Parameterize database queries to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Encode user input to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Use anti-forgery tokens to prevent CSRF attacks.
*   **Authentication and Authorization Vulnerabilities:** Implement secure authentication and authorization mechanisms.
*   **Insecure Direct Object References (IDOR):** Validate user access to objects to prevent IDOR attacks.
*   **File Upload Vulnerabilities:** Validate file types and sizes to prevent malicious file uploads.
*   **Denial-of-Service (DoS) Attacks:** Implement rate limiting and other measures to prevent DoS attacks.
*   **Deserialization Vulnerabilities:** Avoid deserializing untrusted data.
*   **Dependency Vulnerabilities:** Regularly update dependencies to patch security vulnerabilities.

### 4.2. Input Validation

*   **Validate All User Input:** Validate all user input on both the client and server side.
*   **Use Regular Expressions:** Use regular expressions to validate input formats.
*   **Sanitize Input:** Sanitize input to remove or escape potentially malicious characters.
*   **Use Whitelisting:** Use whitelisting to allow only known good input values.
*   **Limit Input Length:** Limit the length of input fields to prevent buffer overflows.

### 4.3. Authentication and Authorization

*   **Use Strong Passwords:** Enforce strong password policies and use password hashing algorithms like bcrypt.
*   **Implement Multi-Factor Authentication (MFA):** Use MFA to add an extra layer of security to the authentication process.
*   **Use Role-Based Access Control (RBAC):** Use RBAC to control user access to resources based on their roles.
*   **Use Claims-Based Authentication:** Use claims-based authentication to represent user identities and permissions.
*   **Implement Proper Session Management:** Implement secure session management practices, including session timeouts and secure cookies.
*   **OAuth 2.0 and OpenID Connect:** Leverage industry-standard protocols for authentication and authorization.

### 4.4. Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data at rest and in transit.
*   **Use Key Management:** Use a secure key management system to store and manage encryption keys.
*   **Protect Connection Strings:** Protect connection strings and other sensitive configuration data.
*   **Implement Auditing:** Implement auditing to track user actions and detect security breaches.

### 4.5. Secure API Communication

*   **Use HTTPS:** Use HTTPS to encrypt communication between the client and server.
*   **Implement API Authentication:** Use API keys, JWTs, or other authentication mechanisms to secure API endpoints.
*   **Rate Limiting:** Implement rate limiting to prevent API abuse.
*   **Input Validation:** Validate all API input to prevent injection attacks.
*   **Output Encoding:** Encode API output to prevent XSS attacks.

## 5. Testing Approaches

Testing is a critical part of the software development process and helps ensure the quality and reliability of C# applications.

### 5.1. Unit Testing

*   **Test Individual Units of Code:** Unit tests should focus on testing individual classes, methods, or functions in isolation.
*   **Use Mocking Frameworks:** Use mocking frameworks like Moq or NSubstitute to isolate units of code and simulate dependencies.
*   **Follow the Arrange-Act-Assert Pattern:** Arrange the test data, act on the code under test, and assert the expected results.
*   **Write Clear and Concise Tests:** Write tests that are easy to read and understand.
*   **Test Edge Cases and Error Conditions:** Test edge cases and error conditions to ensure that the code handles them properly.
*   **Aim for High Code Coverage:** Aim for high code coverage to ensure that most of the code is tested.

### 5.2. Integration Testing

*   **Test Interactions Between Components:** Integration tests should focus on testing the interactions between different components or modules of the application.
*   **Use Real Dependencies or Test Doubles:** Use real dependencies or test doubles to simulate the environment in which the components will operate.
*   **Test Data Access Logic:** Test data access logic to ensure that it interacts correctly with the database.
*   **Test API Endpoints:** Test API endpoints to ensure that they handle requests and responses correctly.

### 5.3. End-to-End Testing

*   **Test the Entire Application Flow:** End-to-end tests should focus on testing the entire application flow from start to finish.
*   **Use Automation Frameworks:** Use automation frameworks like Selenium or Playwright to automate end-to-end tests.
*   **Test User Interfaces:** Test user interfaces to ensure that they are functional and user-friendly.
*   **Test Performance and Scalability:** Test performance and scalability to ensure that the application can handle the expected load.

### 5.4. Test Organization

*   **Create Separate Test Projects:** Create separate test projects for unit tests, integration tests, and end-to-end tests.
*   **Organize Tests by Feature or Module:** Organize tests by feature or module to improve maintainability.
*   **Use Descriptive Test Names:** Use descriptive test names that clearly indicate what the test is verifying.
*   **Use Test Categories:** Use test categories to group related tests together.

### 5.5. Mocking and Stubbing

*   **Use Mocking Frameworks:** Use mocking frameworks like Moq or NSubstitute to create mock objects and stub dependencies.
*   **Create Mock Objects for Dependencies:** Create mock objects for dependencies that are difficult or time-consuming to set up.
*   **Stub Method Calls:** Stub method calls to return specific values or throw exceptions.
*   **Verify Method Calls:** Verify that methods are called with the expected arguments.

## 6. Common Pitfalls and Gotchas

Being aware of common pitfalls and gotchas can help developers avoid mistakes and write more robust C# code.

### 6.1. Frequent Mistakes

*   **NullReferenceException:** Not handling null values properly.
*   **Incorrect Use of Asynchronous Programming:** Blocking the main thread with synchronous operations.
*   **Memory Leaks:** Not disposing of resources properly.
*   **Concurrency Issues:** Race conditions, deadlocks, and other concurrency issues.
*   **Unhandled Exceptions:** Not catching or logging exceptions properly.
*   **SQL Injection:** Not parameterizing database queries.
*   **XSS Attacks:** Not encoding user input properly.
*   **CSRF Attacks:** Not using anti-forgery tokens.
*   **Ignoring Code Analysis Warnings:** Ignoring warnings from the compiler or code analysis tools.

### 6.2. Edge Cases

*   **Empty Collections:** Handling empty collections properly.
*   **Zero Values:** Handling zero values properly.
*   **Maximum and Minimum Values:** Handling maximum and minimum values properly.
*   **Date and Time Zones:** Handling date and time zones correctly.
*   **Unicode Characters:** Handling Unicode characters correctly.

### 6.3. Version-Specific Issues

*   **C# Language Features:** Being aware of new language features and how they affect existing code.
*   **.NET Framework vs. .NET Core vs. .NET:** Understanding the differences between the different .NET platforms.
*   **Breaking Changes:** Being aware of breaking changes in new versions of the .NET framework.

### 6.4. Compatibility Concerns

*   **Cross-Platform Compatibility:** Ensuring that the code works correctly on different operating systems.
*   **Backward Compatibility:** Ensuring that the code is compatible with older versions of the .NET framework.
*   **Interoperability with Other Languages:** Ensuring that the code can interoperate with other languages like C++ or Java.

### 6.5. Debugging Strategies

*   **Use the Visual Studio Debugger:** Use the Visual Studio debugger to step through code, inspect variables, and set breakpoints.
*   **Use Logging:** Use logging to track the execution flow of the code and log important information.
*   **Use Unit Tests:** Use unit tests to isolate and debug individual units of code.
*   **Use Code Analysis Tools:** Use code analysis tools to identify potential problems in the code.
*   **Use Profiling Tools:** Use profiling tools to identify performance bottlenecks.

## 7. Tooling and Environment

Choosing the right tools and configuring the development environment properly can significantly improve productivity and code quality.

### 7.1. Recommended Development Tools

*   **Visual Studio:** The primary IDE for C# development.
*   **Visual Studio Code:** A lightweight code editor with C# support.
*   **JetBrains Rider:** A cross-platform .NET IDE.
*   **NuGet Package Manager:** For managing dependencies.
*   **Git:** For version control.
*   **Resharper:** Powerful Visual Studio extension for code analysis, refactoring, and navigation.

### 7.2. Build Configuration

*   **Use MSBuild or dotnet CLI:** Use MSBuild or the dotnet CLI to build the project.
*   **Configure Build Configurations:** Configure different build configurations for debug and release builds.
*   **Use NuGet Package Restore:** Use NuGet package restore to automatically download dependencies during the build process.
*   **Sign Assemblies:** Sign assemblies to prevent tampering.
*   **Generate Documentation:** Generate documentation from XML comments.
*   **Enable deterministic builds:** Ensure builds are reproducible by enabling deterministic builds in the project file.

### 7.3. Linting and Formatting

*   **Use StyleCop Analyzers:** Use StyleCop Analyzers to enforce coding style rules.
*   **Use Roslyn Analyzers:** Use Roslyn analyzers to detect potential problems in the code.
*   **Configure EditorConfig:** Use EditorConfig to define coding style rules for the project.
*   **Use Code Formatters:** Use code formatters like the Visual Studio code formatter to automatically format code.

### 7.4. Deployment Best Practices

*   **Choose a Deployment Target:** Choose a deployment target based on the application's requirements (e.g., Azure App Service, AWS Elastic Beanstalk, Docker).
*   **Use a Deployment Pipeline:** Use a deployment pipeline to automate the deployment process.
*   **Configure Application Settings:** Configure application settings properly for the deployment environment.
*   **Monitor Application Health:** Monitor application health to detect and resolve issues.

### 7.5. CI/CD Integration

*   **Use a CI/CD Platform:** Use a CI/CD platform like Azure DevOps, GitHub Actions, or Jenkins to automate the build, test, and deployment process.
*   **Configure Build Triggers:** Configure build triggers to automatically start builds when code is committed.
*   **Run Unit Tests and Integration Tests:** Run unit tests and integration tests as part of the build process.
*   **Deploy to Different Environments:** Deploy the application to different environments (e.g., development, staging, production) as part of the deployment pipeline.
*   **Automated Code Reviews:** Integrate with code review tools to automate aspects of code review.

By following these best practices and coding standards, developers can write C# code that is maintainable, performant, secure, and reliable.