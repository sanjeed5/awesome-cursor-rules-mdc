---
description: This rule provides comprehensive best practices and coding standards for developing robust, maintainable, and performant Spring Boot applications, covering code structure, performance, security, testing, and common pitfalls.
globs: *.java
---
# Spring Boot Best Practices and Coding Standards

This document outlines best practices and coding standards for developing applications with Spring Boot. Following these guidelines will help ensure that your applications are robust, maintainable, performant, and secure.

## 1. Code Organization and Structure

### 1.1 Directory Structure

Adopt a layered architecture to separate concerns and improve maintainability. A recommended directory structure is:


src/
 ├── main/
 │   ├── java/
 │   │   └── com/example/app/
 │   │       ├── Application.java (Main entry point)
 │   │       ├── config/          (Configuration classes)
 │   │       ├── controller/      (REST controllers)
 │   │       ├── service/         (Business logic services)
 │   │       ├── repository/      (Data access repositories)
 │   │       ├── model/           (Data transfer objects (DTOs), entities)
 │   │       ├── exception/       (Custom exceptions)
 │   │       ├── util/            (Utility classes)
 │   │       └── security/        (Security-related classes)
 │   └── resources/
 │       ├── application.properties/application.yml (Application configuration)
 │       ├── static/            (Static resources like HTML, CSS, JavaScript)
 │       └── templates/         (View templates, e.g., Thymeleaf)
 └── test/
     ├── java/
     │   └── com/example/app/
     │       ├── controller/      (Controller tests)
     │       ├── service/         (Service tests)
     │       └── repository/      (Repository tests)
     └── resources/
         ├── application.properties/application.yml (Test-specific configuration)


*   **Root Package:** Choose a meaningful root package name (e.g., `com.yourcompany.appname`).
*   **Modularization:** For larger applications, consider breaking down the application into modules (e.g., using Maven or Gradle modules) based on business domains or features.

### 1.2 File Naming Conventions

*   **Classes:** Use PascalCase (e.g., `UserController`, `ProductService`).
*   **Interfaces:** Use PascalCase, often prefixed with `I` (e.g., `ProductRepository`, `IOrderService`). Consider omitting the `I` prefix if it doesn't add value.
*   **Methods:** Use camelCase (e.g., `getUserById`, `calculateTotal`).
*   **Variables:** Use camelCase (e.g., `userName`, `productPrice`).
*   **Constants:** Use UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`).
*   **Configuration Files:** Use lowercase with hyphens (e.g., `application.properties`, `bootstrap.yml`).

### 1.3 Module Organization

For larger projects, break down the application into modules. Each module should represent a distinct business domain or feature.

*   **Maven/Gradle Modules:** Use Maven or Gradle to manage module dependencies and build processes.
*   **Clear Boundaries:** Define clear interfaces between modules to promote loose coupling.
*   **Independent Deployments:** Design modules to be independently deployable, if possible.

### 1.4 Component Architecture

*   **Controllers:** Handle incoming requests and delegate to services. Keep controllers thin.
*   **Services:** Implement business logic. Services should be transactional.
*   **Repositories:** Provide data access abstraction. Use Spring Data JPA or other data access technologies.
*   **Models:** Represent data structures. Use DTOs for transferring data between layers and entities for persistence.

### 1.5 Code Splitting Strategies

*   **Feature-Based Splitting:** Group code related to a specific feature into its own package or module.
*   **Layer-Based Splitting:** Separate code based on layers (e.g., presentation, business logic, data access).
*   **Horizontal vs. Vertical Slicing:** Consider horizontal slicing (grouping similar functionalities across features) or vertical slicing (grouping all functionalities for a specific feature) based on project needs.

## 2. Common Patterns and Anti-Patterns

### 2.1 Design Patterns Specific to Spring Boot

*   **Dependency Injection (DI):** Use constructor injection for required dependencies and setter injection for optional dependencies.
*   **Inversion of Control (IoC):** Let the Spring container manage the lifecycle and dependencies of your beans.
*   **Aspect-Oriented Programming (AOP):** Use AOP for cross-cutting concerns like logging, security, and transaction management.
*   **Repository Pattern:** Use Spring Data repositories for simplified data access.
*   **Service Layer Pattern:** Decouple controllers from business logic by introducing a service layer.
*   **Template Method Pattern:** Use `JdbcTemplate` or `RestTemplate` for consistent data access or external API calls.
*   **Factory Pattern:** Use `@Configuration` classes and `@Bean` methods to define and configure beans.

### 2.2 Recommended Approaches for Common Tasks

*   **Configuration:** Use `application.properties` or `application.yml` for externalized configuration. Use `@ConfigurationProperties` to bind configuration properties to a class.
*   **Logging:** Use SLF4J for logging abstraction and a suitable logging implementation (e.g., Logback or Log4j2).
*   **Exception Handling:** Use `@ControllerAdvice` to handle exceptions globally. Create custom exception classes for specific error scenarios.
*   **Validation:** Use JSR-303 Bean Validation for validating request parameters and request bodies. Use `@Validated` annotation with appropriate groups.
*   **Data Transfer:** Use DTOs to transfer data between layers to avoid exposing internal data structures.
*   **Asynchronous Operations:** Use `@Async` annotation and `TaskExecutor` to perform asynchronous operations.
*   **Caching:** Use Spring's caching abstraction with implementations like Ehcache, Caffeine, or Redis.
*   **Scheduling:** Use `@Scheduled` annotation to schedule tasks.
*   **Transaction Management:** Use `@Transactional` annotation for managing transactions.

### 2.3 Anti-Patterns and Code Smells to Avoid

*   **God Class:** A class that does too much. Break it down into smaller, more focused classes.
*   **Long Method:** A method that is too long and complex. Extract smaller methods.
*   **Feature Envy:** A method that accesses data from another object more than its own. Move the method to the other object.
*   **Data Clumps:** Groups of data that frequently appear together. Create a new class to encapsulate the data clump.
*   **Primitive Obsession:** Using primitive data types instead of creating meaningful domain objects.
*   **Shotgun Surgery:** Making small changes in many different places. Refactor the code to centralize the changes.
*   **Spaghetti Code:** Code that is difficult to understand and maintain due to its tangled structure.
*   **Copy-Paste Programming:** Duplicating code instead of reusing existing code. Create reusable components or methods.
*   **Field Injection:** Use constructor injection instead for required dependencies.
*   **Tight Coupling:** Classes that are highly dependent on each other. Decouple the classes using interfaces or abstract classes.
*   **Ignoring Exceptions:** Catching exceptions but not handling them properly. Log the exception and take appropriate action.
*   **Over-Engineering:** Making the code too complex for the problem it solves. Keep it simple and only add complexity when needed.

### 2.4 State Management Best Practices

*   **Stateless Services:** Design services to be stateless whenever possible. This improves scalability and testability.
*   **Session Management:** Use Spring Session to manage user sessions in a distributed environment. Store session data in a persistent store like Redis or a database.
*   **Caching:** Use caching to store frequently accessed data. Choose a suitable caching strategy (e.g., LRU, FIFO).
*   **Database:** Use a relational database or a NoSQL database to persist data.
*   **Distributed Transactions:** Use distributed transaction management techniques like two-phase commit (2PC) or Saga pattern for transactions spanning multiple services.

### 2.5 Error Handling Patterns

*   **Global Exception Handling:** Use `@ControllerAdvice` and `@ExceptionHandler` to handle exceptions globally.
*   **Custom Exceptions:** Create custom exception classes for specific error scenarios.
*   **Logging:** Log exceptions with sufficient context information (e.g., request parameters, user ID).
*   **Error Responses:** Return meaningful error responses with appropriate HTTP status codes and error messages.
*   **Retry Mechanism:** Implement a retry mechanism for transient errors.
*   **Circuit Breaker:** Use a circuit breaker pattern to prevent cascading failures.
*   **Dead Letter Queue:** Use a dead letter queue to handle messages that cannot be processed.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Database Query Optimization:** Use indexes, optimize queries, and avoid N+1 queries.
*   **Caching:** Use caching to reduce database load and improve response times.
*   **Connection Pooling:** Use connection pooling to reuse database connections.
*   **Asynchronous Operations:** Use asynchronous operations to offload long-running tasks from the main thread.
*   **Load Balancing:** Use load balancing to distribute traffic across multiple instances.
*   **Gzip Compression:** Use Gzip compression to reduce the size of HTTP responses.
*   **Code Profiling:** Use profiling tools to identify performance bottlenecks.

### 3.2 Memory Management

*   **Object Pooling:** Use object pooling to reuse objects and reduce object creation overhead.
*   **Avoid Memory Leaks:** Ensure that objects are properly garbage collected.
*   **Use Appropriate Data Structures:** Choose data structures that are efficient for the operations you perform.
*   **Optimize Collections:** Use appropriate collection types (e.g., `ArrayList` vs. `LinkedList`) based on usage patterns.
*   **Lazy Loading:** Use lazy loading to load data only when it is needed.

### 3.3 Rendering Optimization

*   **Template Caching:** Cache frequently used templates to reduce rendering time.
*   **Minimize DOM Manipulations:** Minimize DOM manipulations in the view layer.
*   **Use CDN:** Use a Content Delivery Network (CDN) to serve static resources.

### 3.4 Bundle Size Optimization

*   **Code Splitting:** Split the code into smaller bundles to reduce the initial load time.
*   **Tree Shaking:** Remove unused code from the bundles.
*   **Minification:** Minify the code to reduce the bundle size.
*   **Compression:** Compress the bundles to reduce the transfer size.

### 3.5 Lazy Loading Strategies

*   **Lazy Initialization:** Initialize objects only when they are first accessed.
*   **Virtual Proxy:** Use a virtual proxy to delay the loading of an object until it is needed.
*   **Database Lazy Loading:** Use lazy loading features provided by JPA or other data access technologies.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **SQL Injection:** Use parameterized queries or ORM frameworks to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Sanitize user input and use output encoding to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Use CSRF tokens to prevent CSRF attacks.
*   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms.
*   **Session Management:** Secure session management to prevent session hijacking.
*   **Denial of Service (DoS):** Implement rate limiting and other measures to prevent DoS attacks.
*   **Insecure Direct Object References (IDOR):** Implement access control checks to prevent unauthorized access to objects.
*   **Security Misconfiguration:** Properly configure security settings to prevent misconfigurations.
*   **Using Components with Known Vulnerabilities:** Keep dependencies up-to-date to address known vulnerabilities.
*   **Insufficient Logging and Monitoring:** Implement sufficient logging and monitoring to detect and respond to security incidents.

### 4.2 Input Validation

*   **Whitelisting:** Validate input against a whitelist of allowed values.
*   **Regular Expressions:** Use regular expressions to validate input format.
*   **Data Type Validation:** Validate that input is of the expected data type.
*   **Length Validation:** Validate that input is within the allowed length limits.
*   **Encoding Validation:** Validate that input is properly encoded.

### 4.3 Authentication and Authorization Patterns

*   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization.
*   **JWT (JSON Web Token):** Use JWT for stateless authentication.
*   **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
*   **Attribute-Based Access Control (ABAC):** Implement ABAC for fine-grained access control based on attributes.
*   **Spring Security:** Leverage Spring Security for authentication and authorization.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Hashing:** Hash passwords and other sensitive data using strong hashing algorithms.
*   **Salting:** Use salting to protect against rainbow table attacks.
*   **Data Masking:** Mask sensitive data when it is displayed or used for non-production purposes.
*   **Tokenization:** Tokenize sensitive data to replace it with non-sensitive tokens.

### 4.5 Secure API Communication

*   **HTTPS:** Use HTTPS for secure communication.
*   **TLS/SSL:** Use TLS/SSL to encrypt data in transit.
*   **API Keys:** Use API keys to authenticate API clients.
*   **Rate Limiting:** Implement rate limiting to prevent abuse.
*   **Input Validation:** Validate all input to prevent injection attacks.
*   **Output Encoding:** Encode output to prevent XSS attacks.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test-Driven Development (TDD):** Write tests before writing the code.
*   **Mocking:** Use mocking frameworks (e.g., Mockito) to isolate the unit under test.
*   **Assertion Libraries:** Use assertion libraries (e.g., AssertJ) for expressive assertions.
*   **Code Coverage:** Aim for high code coverage.
*   **Test Naming:** Use clear and descriptive test names.
*   **Arrange-Act-Assert:** Structure tests using the Arrange-Act-Assert pattern.

### 5.2 Integration Testing

*   **Test Slices:** Use Spring Boot's test slices (e.g., `@WebMvcTest`, `@DataJpaTest`) to test specific parts of the application.
*   **TestContainers:** Use Testcontainers to run integration tests with real dependencies (e.g., databases, message queues).
*   **Spring Test:** Use Spring's testing support for integration tests.
*   **Database Testing:** Use an in-memory database or a test database for database integration tests.

### 5.3 End-to-End Testing

*   **Selenium:** Use Selenium to automate browser-based end-to-end tests.
*   **REST Assured:** Use REST Assured to test REST APIs.
*   **Headless Browser:** Use a headless browser for faster end-to-end tests.

### 5.4 Test Organization

*   **Test Packages:** Create separate packages for unit tests, integration tests, and end-to-end tests.
*   **Test Classes:** Create test classes that correspond to the classes under test.
*   **Test Suites:** Use test suites to group related tests.

### 5.5 Mocking and Stubbing

*   **Mockito:** Use Mockito for mocking dependencies.
*   **Spring MockMvc:** Use Spring MockMvc for testing controllers.
*   **WireMock:** Use WireMock for stubbing external services.
*   **Avoid Over-Mocking:** Mock only the dependencies that are necessary to isolate the unit under test.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Not Understanding Spring Boot Concepts:** Jumping into Spring Boot without a solid understanding of Spring and Dependency Injection.
*   **Overusing `@Autowired`:** Using `@Autowired` for field injection instead of constructor injection.
*   **Not Using Spring Boot Starters:** Manually adding dependencies instead of using Spring Boot Starters.
*   **Not Externalizing Configuration:** Hardcoding configuration values instead of using `application.properties` or `application.yml`.
*   **Not Handling Exceptions Properly:** Ignoring exceptions or not providing meaningful error responses.
*   **Not Writing Tests:** Neglecting to write unit tests and integration tests.
*   **Using `System.out.println` for Logging:** Using `System.out.println` instead of a proper logging framework.
*   **Not Securing the Application:** Failing to implement proper security measures.
*   **Not Monitoring the Application:** Not setting up proper monitoring and alerting.

### 6.2 Edge Cases to Be Aware Of

*   **Null Values:** Handle null values gracefully.
*   **Empty Collections:** Handle empty collections properly.
*   **Large Datasets:** Optimize performance for large datasets.
*   **Concurrency Issues:** Handle concurrency issues properly.
*   **Network Errors:** Handle network errors gracefully.

### 6.3 Version-Specific Issues

*   **Spring Boot Version Compatibility:** Ensure that dependencies are compatible with the Spring Boot version.
*   **Java Version Compatibility:** Ensure that the Java version is compatible with the Spring Boot version.
*   **Third-Party Library Compatibility:** Ensure that third-party libraries are compatible with the Spring Boot version.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:** Ensure that the application is compatible with different browsers.
*   **Operating System Compatibility:** Ensure that the application is compatible with different operating systems.
*   **Device Compatibility:** Ensure that the application is compatible with different devices.

### 6.5 Debugging Strategies

*   **Logging:** Use logging to trace the execution flow and identify errors.
*   **Debuggers:** Use debuggers to step through the code and inspect variables.
*   **Profiling Tools:** Use profiling tools to identify performance bottlenecks.
*   **Remote Debugging:** Use remote debugging to debug applications running on remote servers.
*   **Heap Dumps:** Use heap dumps to analyze memory usage.
*   **Thread Dumps:** Use thread dumps to analyze thread activity.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **IDE:** IntelliJ IDEA, Eclipse, or Visual Studio Code.
*   **Build Tool:** Maven or Gradle.
*   **Version Control:** Git.
*   **Database Client:** DBeaver or SQL Developer.
*   **API Testing Tool:** Postman or Insomnia.

### 7.2 Build Configuration

*   **Maven:** Use `pom.xml` to define dependencies and build configuration.
*   **Gradle:** Use `build.gradle` to define dependencies and build configuration.
*   **Spring Boot Maven Plugin:** Use the Spring Boot Maven Plugin for packaging and running the application.
*   **Spring Boot Gradle Plugin:** Use the Spring Boot Gradle Plugin for packaging and running the application.

### 7.3 Linting and Formatting

*   **Checkstyle:** Use Checkstyle to enforce coding style guidelines.
*   **PMD:** Use PMD to find potential code defects.
*   **FindBugs/SpotBugs:** Use FindBugs/SpotBugs to find potential bugs.
*   **EditorConfig:** Use EditorConfig to maintain consistent coding styles across different editors.
*   **IntelliJ IDEA Code Style:** Configure IntelliJ IDEA's code style settings to match the project's coding style.

### 7.4 Deployment Best Practices

*   **Containerization:** Use Docker to containerize the application.
*   **Orchestration:** Use Kubernetes or Docker Swarm to orchestrate containers.
*   **Cloud Deployment:** Deploy the application to a cloud platform (e.g., AWS, Azure, Google Cloud).
*   **Configuration Management:** Use configuration management tools (e.g., Spring Cloud Config) to manage configuration in a distributed environment.
*   **Monitoring:** Set up monitoring to track application performance and health.
*   **Logging:** Aggregate logs to a central location for analysis.

### 7.5 CI/CD Integration

*   **Continuous Integration (CI):** Use a CI server (e.g., Jenkins, Travis CI, CircleCI) to automatically build and test the application.
*   **Continuous Delivery (CD):** Use a CD pipeline to automatically deploy the application to production.
*   **Automated Testing:** Automate unit tests, integration tests, and end-to-end tests.
*   **Code Quality Checks:** Integrate code quality checks (e.g., Checkstyle, PMD, FindBugs/SpotBugs) into the CI pipeline.