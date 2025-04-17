---
description: Comprehensive guidelines for Pony ORM best practices, covering code organization, patterns, performance, security, testing, common pitfalls, and tooling.
globs: *.py
---
# Pony ORM Best Practices

This document outlines best practices for developing Python applications using Pony ORM. Following these guidelines will improve code maintainability, readability, performance, and security.

## 1. Code Organization and Structure

### Directory Structure

Adopt a logical directory structure to organize your project. A common pattern is to separate data access logic (Pony entities and queries) from business logic.


project/
├── models/
│   ├── __init__.py
│   ├── entities.py  # Pony entity definitions
│   └── queries.py   # Reusable queries
├── services/
│   ├── __init__.py
│   ├── user_service.py  # Business logic for users
│   └── product_service.py # Business logic for products
├── utils/
│   └── db.py        # Database connection and session management
├── tests/
│   ├── __init__.py
│   ├── test_entities.py
│   └── test_user_service.py
├── main.py          # Application entry point
└── requirements.txt


### File Naming Conventions

*   Use descriptive names for files and modules.
*   `entities.py`: Contains Pony entity definitions.
*   `queries.py`: Contains reusable database queries.
*   `[module]_service.py`: Contains business logic for a specific module.
*   `test_[module].py`: Contains unit tests for a specific module.

### Module Organization

*   Group related entities and queries into separate modules.
*   Use `__init__.py` to define module interfaces and control imports.

### Component Architecture

*   **Entities Layer**: Contains database models and relationships. Entities should primarily define data structure and basic validation.
*   **Data Access Layer**: Manages database interactions.  Implement CRUD operations (Create, Read, Update, Delete) within this layer. Avoid business logic in this layer, focusing solely on data retrieval and persistence.
*   **Business Logic Layer**:  This layer contains the core application logic. It orchestrates data flow between the data access layer and the presentation layer.
*   **Presentation Layer**: (If applicable) Handles user interaction and data display.  This could be a web framework, GUI application or CLI.

### Code Splitting

*   Break down large modules into smaller, more manageable files.
*   Use subpackages to group related modules.
*   Consider splitting modules based on functionality or entity type.

## 2. Common Patterns and Anti-patterns

### Design Patterns

*   **Repository Pattern**: Encapsulate data access logic behind a repository interface. This allows for easier testing and switching of data storage implementations.
*   **Unit of Work Pattern**: Track changes to entities within a single unit of work (database transaction) and commit all changes at once.
*   **Data Mapper Pattern**:  Map data between the database and the entity model.  Pony ORM inherently implements this pattern.
*   **Factory Pattern**:  Use factories to create entity instances, especially when instantiation logic is complex or requires external dependencies.
*   **Singleton Pattern**:  Ensure only one instance of the `Database` object exists for each database connection.  This is often used in conjunction with a dependency injection framework.

### Recommended Approaches for Common Tasks

*   **Fetching data efficiently**: Utilize Pony's query optimization features to minimize database round trips.
*   **Handling relationships**:  Properly define relationships (One-to-many, Many-to-many) using `Set`, `Required`, and `Optional` attributes.
*   **Performing complex queries**: Leverage Pony's Pythonic query syntax for readable and maintainable queries.
*   **Managing transactions**: Always use the `db_session` decorator or context manager to wrap database interactions.
*   **Implementing validation**: Define validation rules within entity classes or using separate validation functions.

### Anti-patterns and Code Smells

*   **Mixing business logic with entity definitions**: Keep entity classes focused on data representation, not application logic.
*   **Performing database operations outside of a `db_session`**: This will lead to `TransactionError` exceptions.
*   **Over-fetching data**:  Only select the data that is needed for a specific operation. Use projections in your queries.
*   **Ignoring Pony's query optimization**:  Write inefficient queries that lead to performance bottlenecks.
*   **Directly manipulating the database connection**:  Rely on Pony ORM for database interactions.
*   **Not handling exceptions**: Lack of exception handling can lead to application instability and data corruption.
*   **Using global database sessions without proper context management.** This can lead to concurrency issues.
*   **Not using indexes where appropriate.** This will lead to slow query performance.

### State Management

*   Within a `db_session`, Pony ORM manages the state of entities automatically.
*   For long-running processes, be mindful of memory usage and consider clearing the `db_session` cache periodically.
*   Avoid passing entity instances outside of the `db_session` scope if you intend to modify them.  Consider extracting relevant data and passing simple data objects (dictionaries, data transfer objects) instead.

### Error Handling

*   Use `try...except` blocks to handle potential exceptions during database operations.
*   Rollback transactions in case of errors using `db.rollback()` or let `db_session` handle it.
*   Log errors for debugging and monitoring purposes.
*   Define custom exceptions for specific error scenarios.
*   Consider retrying transactions for recoverable errors.

## 3. Performance Considerations

### Optimization Techniques

*   **Query optimization**:  Use Pony's query syntax effectively to generate optimized SQL queries. Analyze generated SQL queries to identify potential performance bottlenecks and optimize using indexes. Use `set_sql_debug(True)` to observe the queries.
*   **Caching**: Pony ORM includes Identity Map, automatic caching of queries and objects. Understand its limitations and potential benefits.
*   **Lazy loading**: Avoid loading unnecessary data by using lazy loading for related entities.
*   **Eager loading**: For well-known queries where related entities are always required, use eager loading to reduce the number of database queries.
*   **Batch operations**: Perform bulk inserts, updates, and deletes to minimize database round trips.
*   **Indexing**: Add indexes to frequently queried columns to improve query performance.
*   **Connection pooling**:  Pony ORM uses connection pooling. Configure connection pool size based on application needs.
*   **Data Type Selection**: Choose the appropriate data types for your entities to minimize storage space and improve query performance.

### Memory Management

*   Be aware of the size of the `db_session` cache and clear it periodically for long-running processes.
*   Avoid loading large datasets into memory at once.
*   Use generators for processing large datasets in chunks.

### Rendering Optimization

*   If applicable, optimize rendering logic to minimize the amount of data that needs to be displayed.
*   Use pagination for displaying large lists of data.
*   Consider caching rendered output to reduce database load.

### Bundle Size Optimization

*   If applicable to your project, minimize the size of your application bundle by removing unused code and dependencies.
*   Use code splitting to load only the code that is needed for a specific page or feature.

### Lazy Loading

*   Pony ORM supports lazy loading for related entities.
*   Use lazy loading to avoid loading unnecessary data upfront.
*   Be mindful of the N+1 problem when using lazy loading and consider eager loading for frequently accessed related entities.

## 4. Security Best Practices

### Common Vulnerabilities and Prevention

*   **SQL injection**: Prevent SQL injection by using Pony's parameterized queries.  Never construct SQL queries by concatenating strings directly. Pony ORM avoids SQL injection by using generator expressions or lambda functions.
*   **Cross-site scripting (XSS)**:  If applicable to your project, sanitize user input to prevent XSS attacks.
*   **Cross-site request forgery (CSRF)**: If applicable to your project, protect against CSRF attacks by using CSRF tokens.
*   **Authentication and authorization vulnerabilities**: Implement secure authentication and authorization mechanisms to protect sensitive data.
*   **Denial-of-service (DoS) attacks**:  Protect against DoS attacks by implementing rate limiting and other security measures.

### Input Validation

*   Validate all user input to prevent malicious data from being stored in the database.
*   Use appropriate validation rules for each entity attribute.
*   Sanitize input to remove potentially harmful characters.

### Authentication and Authorization

*   Use a secure authentication mechanism to verify user identities.
*   Implement authorization rules to control access to sensitive data and functionality.
*   Consider using a role-based access control (RBAC) system.

### Data Protection

*   Encrypt sensitive data at rest and in transit.
*   Use strong passwords and store them securely.
*   Implement data masking to protect sensitive data from unauthorized access.

### Secure API Communication

*   Use HTTPS for all API communication.
*   Implement API authentication and authorization.
*   Protect against API attacks, such as rate limiting and input validation.

## 5. Testing Approaches

### Unit Testing

*   Write unit tests for individual components, such as entity classes, queries, and services.
*   Use mocking and stubbing to isolate components from external dependencies.
*   Test edge cases and error conditions.
*   Use `in-memory` SQLite database for faster and isolated testing.

### Integration Testing

*   Write integration tests to verify the interaction between different components.
*   Test the integration between Pony ORM and the database.
*   Use a test database that is separate from the production database.

### End-to-End Testing

*   Write end-to-end tests to verify the entire application workflow.
*   Test the application from the user's perspective.
*   Use a testing framework such as Selenium or Cypress.

### Test Organization

*   Organize tests into separate modules based on the component or functionality being tested.
*   Use a consistent naming convention for test files and functions.
*   Keep tests small and focused.

### Mocking and Stubbing

*   Use mocking to replace external dependencies with controlled substitutes.
*   Use stubbing to provide predefined responses for external dependencies.
*   Use mocking frameworks such as `unittest.mock` or `pytest-mock`.

## 6. Common Pitfalls and Gotchas

### Frequent Mistakes

*   Forgetting to use `db_session` for database interactions.
*   Writing inefficient queries that lead to performance bottlenecks.
*   Not handling exceptions properly.
*   Ignoring security best practices.
*   Using deprecated features or APIs.
*   Incorrectly configuring database connections.
*   Misunderstanding how the Identity Map cache works.

### Edge Cases

*   Handling concurrent database access.
*   Dealing with large datasets.
*   Managing database migrations.
*   Handling database errors and exceptions.
*   Working with different database vendors.

### Version-Specific Issues

*   Be aware of compatibility issues between different versions of Pony ORM.
*   Consult the Pony ORM documentation for version-specific information.

### Compatibility Concerns

*   Be aware of compatibility issues between Pony ORM and other technologies, such as web frameworks and database drivers.
*   Consult the Pony ORM documentation for compatibility information.

### Debugging Strategies

*   Use Pony's SQL debugging feature to inspect generated SQL queries.
*   Use logging to track application behavior and identify errors.
*   Use a debugger to step through code and inspect variables.
*   Check the database logs for errors and warnings.

## 7. Tooling and Environment

### Recommended Development Tools

*   Python IDE (e.g., PyCharm, VS Code).
*   Database client (e.g., DBeaver, pgAdmin).
*   Testing framework (e.g., pytest, unittest).
*   Mocking framework (e.g., unittest.mock, pytest-mock).
*   Linting and formatting tools (e.g., flake8, black).

### Build Configuration

*   Use a build system such as `setuptools` or `poetry` to manage project dependencies.
*   Specify Pony ORM as a dependency in your project's `requirements.txt` or `pyproject.toml` file.

### Linting and Formatting

*   Use linting tools such as `flake8` to enforce code style guidelines.
*   Use formatting tools such as `black` to automatically format code.
*   Configure your IDE to run linting and formatting tools automatically.

### Deployment

*   Deploy your application to a production environment that is separate from the development environment.
*   Configure your application to use a production database.
*   Monitor your application for errors and performance issues.

### CI/CD Integration

*   Use a continuous integration and continuous delivery (CI/CD) system to automate the build, test, and deployment process.
*   Integrate Pony ORM tests into your CI/CD pipeline.
*   Automate database migrations as part of your CI/CD process.