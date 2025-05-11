---
description: This rule set provides comprehensive best practices and coding standards for developing robust and maintainable Java backend applications using the Spring Boot framework. It focuses on code structure, performance, security, and testing.
globs: *.java
---
- **Code Organization and Structure**
  - **Directory Structure Best Practices**
    - **Layered Architecture:** Organize code into layers: `controller`, `service`, `repository`, `model`, `configuration`, and `exception`.  This provides clear separation of concerns. Example:
      
      src/
      └── main/
          └── java/
              └── com/example/app/
                  ├── controller/
                  │   └── OrderController.java
                  ├── service/
                  │   └── OrderService.java
                  ├── repository/
                  │   └── OrderRepository.java
                  ├── model/
                  │   └── Order.java
                  ├── configuration/
                  │   └── AppConfig.java
                  └── exception/
                      └── OrderNotFoundException.java
      
    - **Feature-Based Organization:** Group code by feature rather than layer. This keeps related files together for easier navigation.  Use when features are more important than technical separation.
      
      src/
      └── main/
          └── java/
              └── com/example/app/
                  ├── order/
                  │   ├── OrderController.java
                  │   ├── OrderService.java
                  │   ├── OrderRepository.java
                  │   └── Order.java
                  └── customer/
                      ├── CustomerController.java
                      ├── CustomerService.java
                      ├── CustomerRepository.java
                      └── Customer.java
      
  - **File Naming Conventions**
    - **Classes:** Use `PascalCase`.  Names should be nouns representing the purpose of the class. Example: `OrderService`, `ProductController`.
    - **Interfaces:** Use `PascalCase`.  Often named with an `-able` suffix (e.g., `Runnable`, `Serializable`) or prefixed with `I` (e.g., `IUserService`). Example: `OrderRepository`, `UserService`.
    - **Methods:** Use `camelCase`.  Names should be verbs indicating the action performed. Example: `createOrder()`, `calculateTotalPrice()`.
    - **Variables:** Use `camelCase`.  Descriptive and concise names providing context. Avoid abbreviations unless widely understood. Example: `orderTotal`, `userList`.
    - **Constants:** Use `UPPER_SNAKE_CASE`. Example: `MAX_RETRIES`, `DEFAULT_TIMEOUT`.
  - **Module Organization**
    - **Modular Monolith:**  Break down the application into logical modules, each with a specific responsibility.  This promotes maintainability and testability without the complexity of microservices.  Maven or Gradle can be used to manage modules.
    - **Microservices:** When appropriate, consider breaking the application into independent microservices communicating over APIs. This provides scalability and independent deployment.  Example: `Order Service`, `Payment Service`, `Customer Service`.
  - **Component Architecture**
    - **Controllers:** Handle HTTP requests and delegate business logic to services.  Keep controllers thin and focused on request mapping and response formatting.
    - **Services:** Contain the core business logic of the application. Services should be stateless and independent of HTTP concerns.
    - **Repositories:** Handle data access and persistence. Use Spring Data JPA repositories to simplify database interactions.
    - **Data Transfer Objects (DTOs):**  Use DTOs for request and response payloads to separate API contracts from domain models.  This allows for flexibility in evolving the API without affecting internal data structures.
  - **Code Splitting Strategies**
    - **By Feature:** Split code into modules based on features, improving maintainability and scalability.
    - **By Layer:** Split code into distinct layers (presentation, business, data) to separate concerns and improve testability.

- **Common Patterns and Anti-Patterns**
  - **Design Patterns**
    - **Dependency Injection:** Core to Spring. Use constructor injection for required dependencies and setter injection for optional dependencies.  Avoid field injection.
    - **Inversion of Control (IoC):** Let the Spring container manage the creation and lifecycle of beans.
    - **Aspect-Oriented Programming (AOP):** Use AOP for cross-cutting concerns like logging, security, and transaction management. Use `@Aspect` annotation.
    - **Template Method:** Apply when different implementations of an algorithm are needed. Spring's `JdbcTemplate` is a classical example.
    - **Factory Pattern:** Can be used for abstracting object creation.
    - **Singleton Pattern:** Spring beans are singletons by default, but be mindful when dealing with stateful beans.
  - **Recommended Approaches**
    - **Configuration Properties:** Externalize configuration using `@ConfigurationProperties` for type-safe access to properties.
    - **Event Handling:** Use Spring's event publishing and listening mechanism for decoupling components. Use `@EventListener` annotation.
    - **Transaction Management:** Use `@Transactional` annotation for declarative transaction management.
    - **Validation:** Use `@Validated` and JSR-303 annotations for request parameter and body validation.
  - **Anti-Patterns and Code Smells**
    - **God Classes:** Avoid classes with too many responsibilities.  Apply the Single Responsibility Principle.
    - **Long Methods:** Break down long methods into smaller, more focused methods.
    - **Primitive Obsession:** Avoid using primitive types excessively.  Encapsulate related data into value objects.
    - **Magic Numbers/Strings:** Replace hardcoded values with constants or configuration properties.
    - **Tight Coupling:** Minimize dependencies between components through dependency injection and interfaces.
  - **State Management**
    - **Stateless Services:** Services should be stateless to improve scalability and concurrency.
    - **Session Management:** Use Spring Session for managing user sessions in a distributed environment.  Consider using Redis or other external storage for session data.
    - **Caching:** Implement caching strategically to reduce database load and improve response times. Use Spring Cache abstraction with appropriate cache providers.
  - **Error Handling Patterns**
    - **Global Exception Handling:** Use `@ControllerAdvice` and `@ExceptionHandler` to handle exceptions globally and provide consistent error responses.
    - **Custom Exceptions:** Create custom exception classes to represent application-specific errors.
    - **Logging:** Log exceptions with appropriate levels (e.g., `error`, `warn`, `info`).
    - **Return Meaningful Error Responses:** Return well-structured error responses with clear error codes and messages.  Use a consistent format for error responses.

- **Performance Considerations**
  - **Optimization Techniques**
    - **Database Connection Pooling:** Use a connection pool (e.g., HikariCP) to reuse database connections and reduce overhead.
    - **Caching:** Implement caching using Spring Cache abstraction to store frequently accessed data.
    - **Asynchronous Processing:** Use `@Async` for long-running tasks that don't need to be executed synchronously.
    - **Lazy Loading:** Use lazy loading for entities and relationships in JPA to avoid loading unnecessary data.
    - **Profiling:** Use profiling tools (e.g., VisualVM, JProfiler) to identify performance bottlenecks.
  - **Memory Management**
    - **Garbage Collection Tuning:** Tune JVM garbage collection settings for optimal performance.  Monitor GC performance and adjust parameters as needed.
    - **Object Pooling:** Consider using object pooling for frequently created and destroyed objects.
    - **Avoid Memory Leaks:**  Be careful to release resources properly to prevent memory leaks. Use try-with-resources for closing streams and other resources.
  - **Rendering Optimization**
    - *(Not Directly Applicable to Spring Backend - Focus on API Response times and efficiency)*
    - **Minimize Data Transfer:** Only return the data needed by the client.
    - **Use Efficient Data Structures:** Use appropriate data structures for API responses (e.g., lists, maps).
    - **Compression:** Enable GZIP compression for API responses to reduce the size of the data transferred over the network.
  - **Bundle Size Optimization**
    - *(Not Directly Applicable to Spring Backend - Although Dependency Management is crucial)*
    - **Minimize Dependencies:** Only include necessary dependencies.
    - **Use Code Analysis Tools:** Use tools like SonarQube to identify unused code and dependencies.
  - **Lazy Loading**
    - **JPA Lazy Loading:**  Use lazy loading for JPA relationships to avoid loading unnecessary data. Use the `@OneToMany(fetch = FetchType.LAZY)` annotation.

- **Security Best Practices**
  - **Common Vulnerabilities and Prevention**
    - **SQL Injection:** Use parameterized queries or ORM frameworks (e.g., Spring Data JPA) to prevent SQL injection.
    - **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
    - **Cross-Site Request Forgery (CSRF):** Use CSRF protection mechanisms (e.g., Spring Security's CSRF support).
    - **Authentication/Authorization Flaws:**  Implement robust authentication and authorization mechanisms using Spring Security.
    - **Dependency Vulnerabilities:** Regularly scan dependencies for known vulnerabilities and update them.
  - **Input Validation**
    - **Server-Side Validation:** Always perform input validation on the server-side.
    - **JSR-303 Validation:** Use JSR-303 annotations for validating request parameters and body.
    - **Custom Validation:** Implement custom validation logic for complex validation rules.
  - **Authentication and Authorization**
    - **Spring Security:** Use Spring Security for authentication and authorization.
    - **OAuth 2.0/OpenID Connect:** Use OAuth 2.0/OpenID Connect for delegating authentication and authorization to external providers.
    - **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on user roles.
  - **Data Protection**
    - **Encryption:** Encrypt sensitive data at rest and in transit.
    - **Hashing:** Hash passwords using strong hashing algorithms (e.g., bcrypt, Argon2).
    - **Data Masking:** Mask sensitive data when displaying or logging it.
  - **Secure API Communication**
    - **HTTPS:** Use HTTPS for all API communication.
    - **API Keys:** Use API keys for authenticating clients.
    - **Rate Limiting:** Implement rate limiting to prevent abuse.

- **Testing Approaches**
  - **Unit Testing**
    - **JUnit:** Use JUnit for writing unit tests.
    - **Mockito:** Use Mockito for mocking dependencies.
    - **Test-Driven Development (TDD):** Write tests before writing code.
    - **Focus on Business Logic:**  Focus unit tests on the business logic of the application.
  - **Integration Testing**
    - **Spring Boot Test:** Use `@SpringBootTest` annotation for integration tests.
    - **TestRestTemplate:** Use `TestRestTemplate` for testing REST endpoints.
    - **Database Testing:**  Use an in-memory database (e.g., H2, embedded database) or a test database for integration tests.
    - **Verify Component Interactions:**  Verify that different components of the application work together correctly.
  - **End-to-End Testing**
    - **Selenium:** Use Selenium for end-to-end testing of web applications.
    - **Cypress:** Use Cypress as another alternative for end-to-end testing.
    - **Test the Entire System:**  Test the entire system from the user interface to the database.
  - **Test Organization**
    - **Separate Test Directory:** Keep tests in a separate directory from the source code.
    - **Test Naming Conventions:**  Use clear and consistent naming conventions for tests (e.g., `ClassNameTest`).
    - **Organize Tests by Component:**  Organize tests by component or feature.
  - **Mocking and Stubbing**
    - **Mockito Annotations:** Use Mockito annotations (e.g., `@Mock`, `@InjectMocks`) to simplify mocking.
    - **When/Then:** Use the `when()` and `thenReturn()` methods to define mock behavior.
    - **Verify Interactions:** Use the `verify()` method to verify that mock interactions occurred.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes**
    - **Over-Engineering:**  Avoid over-engineering solutions. Keep it simple and only add complexity when needed.
    - **Ignoring Exceptions:** Don't catch exceptions and ignore them. Always log exceptions and handle them appropriately.
    - **Hardcoding Values:**  Avoid hardcoding values. Use configuration properties instead.
    - **Not Using Dependency Injection Properly:**  Use dependency injection to decouple components and improve testability.
  - **Edge Cases**
    - **Null Values:**  Handle null values gracefully.
    - **Concurrency Issues:** Be aware of concurrency issues when dealing with shared resources.
    - **Resource Exhaustion:**  Handle resource exhaustion gracefully (e.g., database connections, memory).
  - **Version-Specific Issues**
    - **Check Release Notes:**  Review release notes for each version of Spring Boot and its dependencies to identify potential issues.
    - **Test Upgrades:**  Thoroughly test upgrades to ensure compatibility.
  - **Compatibility Concerns**
    - **Dependency Conflicts:**  Manage dependency conflicts using Maven or Gradle.
    - **Java Version Compatibility:**  Ensure compatibility between Spring Boot and the Java version.
  - **Debugging Strategies**
    - **Logging:** Use logging liberally to track the flow of execution and identify issues.
    - **Debuggers:** Use debuggers to step through code and inspect variables.
    - **Remote Debugging:** Use remote debugging to debug applications running in remote environments.

- **Tooling and Environment**
  - **Recommended Development Tools**
    - **IntelliJ IDEA:** Popular IDE with excellent Spring Boot support.
    - **Eclipse:** Another popular IDE with Spring Tool Suite plugin.
    - **Spring Initializr:** Web-based tool for generating Spring Boot projects.
  - **Build Configuration**
    - **Maven:** Use Maven for dependency management and building projects.
    - **Gradle:** Use Gradle as another alternative to Maven.
    - **Spring Boot Plugin:** Use the Spring Boot Maven or Gradle plugin for building executable JARs.
  - **Linting and Formatting**
    - **Checkstyle:** Use Checkstyle for enforcing coding standards.
    - **PMD:** Use PMD for finding potential bugs and code smells.
    - **SpotBugs:** Use SpotBugs to find potential bug patterns in the code.
    - **Prettier:** Use Prettier for formatting code consistently.
  - **Deployment Best Practices**
    - **Containerization:** Use Docker for containerizing applications.
    - **Cloud Platforms:** Deploy applications to cloud platforms (e.g., AWS, Azure, GCP).
    - **Immutable Infrastructure:**  Use immutable infrastructure for deploying applications.
  - **CI/CD Integration**
    - **Jenkins:** Use Jenkins for continuous integration and continuous deployment.
    - **GitLab CI:** Use GitLab CI as another CI/CD alternative.
    - **GitHub Actions:** Use GitHub Actions for automating build, test, and deployment workflows.

- **Additional Tips**
    - **Use Spring Boot DevTools:** These tools provide automatic application restarts, live reloading, and other useful features during development.
    - **Monitor Application Health:** Use Spring Boot Actuator to monitor application health and performance metrics.
    - **Stay Up-to-Date:** Keep up-to-date with the latest Spring Boot releases and best practices.