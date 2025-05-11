---
description: This rule provides comprehensive best practices for using Tortoise ORM in Python projects, covering aspects like code organization, performance, security, and testing.
globs: *.py
---
- Use Tortoise ORM for asynchronous database interactions in Python.
- Always use UV when installing dependencies for faster performance, especially when combined with Tortoise ORM's asynchronous capabilities.
- Ensure you are using a supported version of Python (>= 3.8). Python 3.12 or later is recommended for the best performance and features.
- Organize code into classes for better structure and maintainability, especially when defining models and database interactions.

## 1. Code Organization and Structure:

- **Directory Structure:**
    - `src/`: Root directory for the main application code.
    - `src/models/`: Contains Tortoise ORM model definitions.
    - `src/schemas/`: Pydantic schemas for data validation and API interaction (if using FastAPI).
    - `src/routers/`: FastAPI routers or equivalent for handling API endpoints.
    - `src/database.py`: Database initialization and connection management.
    - `src/config.py`: Configuration settings for the application.
    - `tests/`: Contains unit and integration tests.
    - `migrations/`: Aerich migration files.
    - `docs/`: Documentation for the project.

- **File Naming Conventions:**
    - Models: `user_model.py`, `product_model.py`
    - Schemas: `user_schema.py`, `product_schema.py`
    - Routers: `user_router.py`, `product_router.py`
    - Database: `database.py` or `db.py`
    - Config: `config.py` or `settings.py`

- **Module Organization:**
    - Group related models, schemas, and routers into logical modules.
    - Use namespaces (packages) to avoid naming conflicts.
    - Keep modules focused and avoid overly large files.

- **Component Architecture:**
    - Models: Define database entities using Tortoise ORM's `Model` class.
    - Schemas: Use Pydantic for data validation and serialization.
    - Services: Encapsulate business logic and database interactions.
    - Repositories: Abstract data access logic from services (optional).
    - Controllers/Routers: Handle API requests and responses.

- **Code Splitting:**
    - Use lazy loading for models with large relationships.
    - Split large modules into smaller, more manageable files.
    - Consider using sub-applications (e.g., with FastAPI) to separate concerns.

## 2. Common Patterns and Anti-patterns:

- **Design Patterns:**
    - Repository Pattern: Abstract data access logic.
    - Unit of Work Pattern: Manage database transactions.
    - Factory Pattern: Create model instances with complex dependencies.
    - DTO (Data Transfer Object): Use Pydantic models for data transfer between layers.

- **Recommended Approaches:**
    - Use `Tortoise.init()` to initialize the database connection once at application startup.
    - Use `Tortoise.generate_schemas()` to create database tables.
    - Use `Tortoise.close_connections()` to close database connections when the application shuts down.
    - Utilize `prefetch_related` for eager loading of related data to optimize performance.
    - Implement proper error handling using try-except blocks to catch potential exceptions.

- **Anti-patterns and Code Smells:**
    - Initializing the database connection on every request.
    - Not closing database connections, leading to resource leaks.
    - Over-fetching data without using `prefetch_related`.
    - Ignoring exceptions, which can hide potential issues.
    - Writing complex SQL queries directly instead of using Tortoise ORM's query builder.
    - Performing database operations within a loop without batching.

- **State Management:**
    - Keep database state separate from application state.
    - Use transactions to ensure data consistency.
    - Avoid storing sensitive data in the application state.

- **Error Handling:**
    - Catch specific exceptions (e.g., `DoesNotExist`, `IntegrityError`) rather than generic `Exception`.
    - Log errors with detailed information for debugging.
    - Use custom exception classes for specific error conditions.

## 3. Performance Considerations:

- **Optimization Techniques:**
    - Use `prefetch_related` to reduce the number of database queries.
    - Use `bulk_create` and `bulk_update` for batch operations.
    - Use indexes to speed up queries on frequently accessed columns.
    - Optimize database queries by using `filter` and `exclude` to narrow down results.
    - Defer loading large fields using Tortoise ORM's field options.

- **Memory Management:**
    - Close database connections when they are no longer needed.
    - Use asynchronous generators to process large datasets.
    - Avoid loading unnecessary data into memory.

- **Rendering Optimization:**
    - This is generally not applicable to Tortoise ORM itself, but consider optimizing data serialization (e.g., using `orjson`).

- **Bundle Size Optimization:**
    - Not directly applicable to Tortoise ORM.

- **Lazy Loading:**
    - Use relationships wisely to avoid over-fetching data.
    - Implement pagination for large result sets.

## 4. Security Best Practices:

- **Common Vulnerabilities:**
    - SQL injection: Prevent by using Tortoise ORM's query builder and parameterized queries.
    - Cross-site scripting (XSS): Prevent by sanitizing user inputs.
    - Cross-site request forgery (CSRF): Implement CSRF protection in your application.

- **Input Validation:**
    - Use Pydantic to validate user inputs before saving them to the database.
    - Sanitize user inputs to prevent XSS attacks.
    - Limit the length of input fields to prevent buffer overflows.

- **Authentication and Authorization:**
    - Use a secure authentication mechanism (e.g., JWT).
    - Implement role-based access control (RBAC) to restrict access to sensitive data.
    - Use password hashing (e.g., bcrypt) to store passwords securely.

- **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use SSL/TLS to secure API communication.
    - Implement data masking to protect sensitive data in logs and reports.

- **Secure API Communication:**
    - Use HTTPS for all API endpoints.
    - Implement rate limiting to prevent denial-of-service attacks.
    - Validate API requests and responses.

## 5. Testing Approaches:

- **Unit Testing:**
    - Test model methods and business logic in isolation.
    - Use mocking to isolate database interactions.
    - Use pytest and asyncio.run for async tests.

- **Integration Testing:**
    - Test the interaction between different components of the application.
    - Use a test database to avoid affecting the production database.
    - Use fixtures to set up test data.

- **End-to-End Testing:**
    - Test the entire application from the user's perspective.
    - Use a testing framework like Playwright or Selenium.

- **Test Organization:**
    - Organize tests by module or component.
    - Use descriptive test names.
    - Keep tests small and focused.

- **Mocking and Stubbing:**
    - Use `unittest.mock` to mock database connections and queries.
    - Use stubs to replace complex dependencies with simplified versions.
    - Avoid over-mocking, which can make tests less reliable.

## 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes:**
    - Forgetting to await asynchronous calls.
    - Using synchronous code in asynchronous contexts.
    - Not handling exceptions properly.
    - Over-fetching data.
    - Not using transactions for complex operations.

- **Edge Cases:**
    - Handling concurrency and race conditions.
    - Dealing with large datasets.
    - Handling database migrations.

- **Version-Specific Issues:**
    - Refer to the Tortoise ORM changelog for breaking changes and known issues.
    - Use version pinning to ensure compatibility between Tortoise ORM and other dependencies.

- **Compatibility Concerns:**
    - Ensure compatibility between Tortoise ORM and the database driver (e.g., `asyncpg`, `aiosqlite`).
    - Ensure compatibility between Tortoise ORM and other libraries (e.g., FastAPI, Pydantic).

- **Debugging Strategies:**
    - Use logging to track the execution flow of the application.
    - Use a debugger to step through the code and inspect variables.
    - Use database profiling tools to identify slow queries.

## 7. Tooling and Environment:

- **Recommended Development Tools:**
    - IDE: PyCharm, VS Code
    - Database client: pgAdmin, Dbeaver
    - Testing framework: pytest
    - Linting: pylint, flake8
    - Formatting: black, autopep8

- **Build Configuration:**
    - Use a `pyproject.toml` file to manage dependencies.
    - Use a virtual environment to isolate dependencies.

- **Linting and Formatting:**
    - Use pylint or flake8 to enforce code style guidelines.
    - Use black or autopep8 to automatically format code.
    - Configure linters and formatters to run automatically on every commit.

- **Deployment:**
    - Use a containerization technology like Docker.
    - Use a process manager like systemd or supervisord.
    - Use a reverse proxy like Nginx or Apache.

- **CI/CD Integration:**
    - Use a CI/CD pipeline to automate testing, linting, and deployment.
    - Use a version control system like Git.
    - Use a deployment tool like Ansible or Terraform.

**Additional Considerations:**

- **Use Asynchronous Context Managers for Database Operations:** Tortoise-ORM is an async ORM. Leveraging asynchronous context managers within functions or methods that interact with the database ensures that the database connection is properly released, preventing connection exhaustion and improving overall performance. Use it with `async with`. Example:

python
async def create_user(user_data):
    async with Tortoise.get_connection('default') as conn:
        await User.create(**user_data)


- **Leverage Transactions:** Complex operations should be wrapped in transactions to ensure atomicity. Tortoise-ORM provides a transaction decorator. For example:

python
from tortoise import transaction

@transaction.atomic()
async def transfer_funds(from_account, to_account, amount):
    # Your operations here
    pass


- **Monitor and Profile Database Queries:** Regularly monitor your database queries to identify slow or inefficient queries. Tools like `pg_stat_statements` for PostgreSQL or similar for other databases can provide insights. Profile your code with tools like `cProfile` to find bottlenecks. Use `explain` command on queries to verify execution plans.

- **Regular Database Maintenance:** Perform regular database maintenance tasks such as vacuuming (PostgreSQL), optimizing tables (MySQL), and updating statistics to ensure optimal performance.