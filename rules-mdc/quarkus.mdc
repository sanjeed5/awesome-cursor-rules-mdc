---
description: This rule provides comprehensive best practices for Quarkus development, covering code structure, performance, security, testing, and tooling. It aims to guide developers in building efficient, secure, and maintainable Quarkus applications.
globs: **/*.java,**/*.kt,**/*.groovy,**/*.clj,**/application.properties,**/application.yaml,**/pom.xml,**/build.gradle
---
---
# Quarkus Best Practices

This document outlines best practices for developing Quarkus applications. Following these guidelines will help you build efficient, maintainable, and secure applications.

## 1. Code Organization and Structure

### 1.1 Project Structure

*   **Maven/Gradle Wrapper:** Always use Maven (`mvnw`) or Gradle (`gradlew`) wrappers to ensure consistent build environments across different machines.
*   **Package Naming:** Follow a consistent package naming convention. A common approach is to structure packages based on layers or features:
    *   `com.example.app.entity`: Database entities.
    *   `com.example.app.repository`: Repository interfaces and implementations.
    *   `com.example.app.service`: Business logic services.
    *   `com.example.app.resource`: REST endpoints (JAX-RS resources).
    *   `com.example.app.config`: Configuration classes.
    *   `com.example.app.mapper`: Data transfer object (DTO) mappers.
    *   `com.example.app.filter`: Servlet filters or request/response interceptors.
    *   `com.example.app.exception`: Custom exception classes and handlers.
*   **Feature-Based Packaging:** For larger codebases, consider organizing packages by feature or domain instead of by layer.

### 1.2 Code Style

*   **Coding Conventions:** Adhere to a consistent coding style. Consider using the Google Java Style Guide or Oracle Code Conventions. Tools like Checkstyle, Prettier, and Spotless can help enforce code style.
*   **Naming Conventions:**
    *   **Classes:** Use `PascalCase` (e.g., `UserResource`, `OrderService`).
    *   **Methods and Variables:** Use `camelCase` (e.g., `findUserById`, `isOrderValid`).
    *   **Constants:** Use `ALL_CAPS` (e.g., `MAX_RETRY_ATTEMPTS`, `DEFAULT_PAGE_SIZE`).
*   **Descriptive Naming:** Use descriptive and meaningful names for methods, variables, and classes.
*   **Annotations:** Use Quarkus and CDI annotations effectively (e.g., `@ApplicationScoped`, `@Inject`, `@ConfigProperty`).

### 1.3 Quarkus-Specific Structure

*   **Resources:** Use resources only for routing and handling HTTP requests. Avoid business logic in resources.
*   **Services:** Implement business logic in services. Services should be stateless and annotated with `@ApplicationScoped`.
*   **Repositories:** Use the repository pattern to abstract data access logic. Consider using Panache for simpler JPA entity and repository patterns.
*   **Data Transfer Objects (DTOs):**  Use DTOs to transfer data between layers, especially between resources and services. This promotes loose coupling and allows you to evolve your data model without affecting the API.

## 2. Common Patterns and Anti-patterns

### 2.1 Dependency Injection

*   **Constructor Injection:** Prefer constructor injection or method injection over field injection for better testability.
*   **CDI Annotations:** Use CDI annotations (`@Inject`, `@Named`, `@Singleton`, etc.) for dependency injection.

### 2.2 Configuration

*   **Externalized Configuration:** Externalize all configuration using `application.properties` or `application.yaml`.
*   **Type-Safe Configuration:** Use `@ConfigProperty` for type-safe configuration injection.
*   **Profiles:** Use Quarkus profiles (e.g., `dev`, `test`, `prod`) for environment-specific configurations.
*   **Configuration Prefix:** Use a common prefix for your application's configuration properties to avoid conflicts with other extensions.

### 2.3 Data Access

*   **Repository Pattern:** Use the Repository pattern to abstract data access concerns.
*   **Panache:** Leverage Quarkus Hibernate ORM with Panache for simpler JPA entity and repository patterns.
*   **Entity Relationships:** Implement proper entity relationships and cascading (OneToMany, ManyToOne, etc.).
*   **Schema Migrations:** Use schema migration tools such as Flyway or Liquibase for managing database schema changes.
*   **Connection Pooling:** Configure your database connection pool appropriately for your application's needs.

### 2.4 REST API Design

*   **RESTful Principles:** Follow RESTful API design principles (proper use of HTTP methods and status codes).
*   **OpenAPI:** Use the Quarkus OpenAPI extension (quarkus-smallrye-openapi) for API documentation.
*   **Detailed OpenAPI Annotations:** Provide detailed OpenAPI annotations for resources, operations, and schemas.
*   **Input Validation:** Implement thorough input validation using the Hibernate Validator extension.

### 2.5 Common Anti-patterns

*   **Global State:** Avoid global state and mutability.
*   **Tight Coupling:** Reduce tight coupling between components by using interfaces and dependency injection.
*   **Business Logic in Resources:** Avoid placing business logic directly in JAX-RS resources. Move business logic to services.
*   **Ignoring Exceptions:** Never ignore exceptions. Handle exceptions appropriately and log relevant information.
*   **Over-Engineering:** Avoid over-engineering solutions. Keep the code simple and focused on the problem at hand.

## 3. Performance Considerations

### 3.1 Native Image Optimization

*   **Native Image Builds:** Optimize for native image creation using the `quarkus.native.*` properties.
*   **Reachability Metadata:** Provide reachability metadata for reflection, resources, and proxies used by your application.
*   **Remove Unused Code:** Remove unused code to reduce the size of the native image.
*   **Profile-Guided Optimization (PGO):**  Consider using PGO to further optimize native image performance.

### 3.2 Caching

*   **Caching Strategies:** Implement caching strategies using `@CacheResult` and `@CacheInvalidate` (MicroProfile or Quarkus caching extensions).
*   **Cache Configuration:** Configure cache settings appropriately for your application's needs.

### 3.3 Reactive Programming

*   **Reactive Patterns:** Implement reactive patterns with Vert.x or Mutiny for non-blocking I/O.
*   **Asynchronous Processing:** Use asynchronous processing for long-running tasks to avoid blocking the main thread.

### 3.4 Database Optimization

*   **Database Indexing:** Employ database indexing and query optimization for performance gains.
*   **Connection Pooling:**  Properly configure database connection pooling.
*   **Avoid N+1 Problem:**  Be aware of the N+1 problem when using ORM and optimize data fetching strategies.

## 4. Security Best Practices

### 4.1 Authentication and Authorization

*   **Quarkus Security:** Use Quarkus Security for authentication and authorization (e.g., quarkus-oidc, quarkus-smallrye-jwt).
*   **MicroProfile JWT:** Integrate MicroProfile JWT for token-based security if applicable.
*   **Role-Based Access Control (RBAC):** Implement RBAC to restrict access to resources based on user roles.
*   **Principle Propagation:** Ensure principle propagation across services when using microservices architecture.

### 4.2 Data Protection

*   **Input Validation:** Implement strict input validation to prevent injection attacks.
*   **Output Encoding:** Encode output properly to prevent cross-site scripting (XSS) attacks.
*   **Password Hashing:** Use strong password hashing algorithms (e.g., bcrypt) to store passwords securely.
*   **TLS/SSL:** Enable TLS/SSL for all communication between the client and the server.

### 4.3 Other Security Considerations

*   **CORS:** Handle CORS configuration and other security headers via Quarkus extensions.
*   **Dependency Scanning:** Regularly scan dependencies for vulnerabilities using tools like OWASP Dependency-Check.
*   **Secret Management:** Use secure secret management practices to store sensitive information like API keys and database passwords.
*   **Regular Security Audits:** Conduct regular security audits to identify and address potential vulnerabilities.

## 5. Testing Approaches

### 5.1 Unit Tests

*   **JUnit 5:** Write unit tests with JUnit 5.
*   **Mocking Frameworks:** Use mocking frameworks like Mockito to isolate components during unit testing.

### 5.2 Integration Tests

*   **@QuarkusTest:** Use `@QuarkusTest` for integration tests.
*   **REST-assured:** Use REST-assured for testing REST endpoints in Quarkus (e.g., `@QuarkusTestResource`).
*   **Testcontainers:** Implement in-memory databases or test-containers for integration testing.
*   **Arquillian:** Consider using Arquillian for more complex integration tests.

### 5.3 End-to-End Tests

*   **Selenium/Cypress:** Use Selenium or Cypress for end-to-end testing of the application UI.
*   **API Testing Tools:** Use API testing tools like Postman or Insomnia to test the application's APIs.

### 5.4 General Testing Practices

*   **Test-Driven Development (TDD):** Consider using TDD to drive the development process.
*   **Code Coverage:** Aim for high code coverage with your tests.
*   **Continuous Integration (CI):** Integrate tests into your CI pipeline to ensure that tests are run automatically on every code change.

## 6. Common Pitfalls and Gotchas

*   **Reflection:** Excessive use of reflection can negatively impact native image build times and performance. Minimize reflection usage and provide reachability metadata.
*   **Blocking Operations:** Avoid blocking operations in reactive contexts. Use non-blocking alternatives whenever possible.
*   **Large Payloads:**  Be mindful of large request and response payloads, especially in native images. Optimize data serialization and deserialization.
*   **Incorrect Configuration:** Double-check configuration settings, especially when deploying to different environments.
*   **Missing Dependencies:** Ensure all required dependencies are included in your project.
*   **Lazy Initialization:** Be aware of lazy initialization issues, especially in native images. Ensure that all necessary components are initialized during the build process.
*   **Class Loading Issues:** Understand the class loading behavior in Quarkus, especially when using custom class loaders.
*   **Transaction Management:** Understand and correctly configure transaction management, particularly when using multiple data sources.

## 7. Tooling and Environment

### 7.1 IDEs

*   **IntelliJ IDEA:** IntelliJ IDEA with the Quarkus plugin is a popular choice for Quarkus development.
*   **Eclipse:** Eclipse with the Quarkus Tools plugin is also a viable option.
*   **VS Code:** VS Code with the Quarkus extension provides basic support for Quarkus development.

### 7.2 Build Tools

*   **Maven:** Maven with the Quarkus Maven plugin is the recommended build tool.
*   **Gradle:** Gradle with the Quarkus Gradle plugin is also supported.

### 7.3 Dev Services

*   **Dev Services:** Leverage Quarkus Dev Services for automatic provisioning of unconfigured services during development and testing.

### 7.4 Monitoring and Logging

*   **Logging:** Use the Quarkus logging subsystem (e.g., quarkus-logging-json) with SLF4J or JUL bridging.
*   **MicroProfile Health, Metrics, and OpenTracing:** Implement MicroProfile Health, Metrics, and OpenTracing for monitoring and diagnostics.
*   **Log Levels:** Use proper log levels (ERROR, WARN, INFO, DEBUG) and structured logging where possible.

### 7.5 Containerization

*   **Docker:** Use Docker for containerizing your Quarkus applications.
*   **Jib:** Use the Jib extension for building container images without a Dockerfile.
*   **Multi-Stage Docker Builds:** Configure multi-stage Docker builds for optimized container images.

### 7.6 Kubernetes

*   **Kubernetes Extension:** Use the Quarkus Kubernetes extension for generating Kubernetes resource files.
*   **Helm:** Consider using Helm for managing Kubernetes deployments.

### 7.7 Continuous Integration/Continuous Deployment (CI/CD)

*   **Jenkins, GitLab CI, GitHub Actions:** Integrate Quarkus applications into CI/CD pipelines using tools like Jenkins, GitLab CI, or GitHub Actions.

By following these best practices, you can develop robust, performant, and secure Quarkus applications that are well-suited for modern cloud environments.