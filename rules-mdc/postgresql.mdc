---
description: Enforces PostgreSQL coding standards, best practices, and performance optimization techniques to ensure maintainable, efficient, and secure database interactions. This rule covers code formatting, data integrity, security, and performance considerations.
globs: *.sql,*.plpgsql,*.c,*.h
---
- **Code Formatting and Comments:**
  - Maintain consistent code formatting using a tool like `pgformatter` or similar.
  - Use clear and concise comments to explain complex logic and intentions. Update comments regularly to avoid confusion.
  - Use inline comments sparingly; prefer block comments for detailed explanations.
  - Write comments in plain, easy-to-follow English.
  - Add a space after line comments (`-- a comment`); do not add a space for commented-out code (`--raise notice`).
  - Keep comments up-to-date; incorrect comments are worse than no comments.

- **Naming Conventions:**
  - Use `snake_case` for identifiers (e.g., `user_id`, `customer_name`).
  - Use plural nouns for table names (e.g., `customers`, `products`).
  - Use consistent naming conventions for functions, procedures, and triggers.
  - Choose descriptive and meaningful names for all database objects.

- **Data Integrity and Data Types:**
  - Use appropriate data types for columns to ensure data integrity (e.g., `INTEGER`, `VARCHAR`, `TIMESTAMP`).
  - Use constraints (e.g., `NOT NULL`, `UNIQUE`, `CHECK`, `FOREIGN KEY`) to enforce data integrity.
  - Define primary keys for all tables.
  - Use foreign keys to establish relationships between tables.
  - Utilize domains to enforce data type constraints reusable across multiple columns.

- **SQL Injection Prevention:**
  - Always use parameterized queries or prepared statements to prevent SQL injection attacks.
  - Sanitize user inputs before using them in SQL queries.
  - Avoid constructing SQL queries by concatenating strings directly.
  - Use the `quote_literal()` and `quote_ident()` functions to properly escape strings and identifiers.

- **Transaction Management:**
  - Use explicit transactions to ensure data consistency and atomicity.
  - Start transactions with `BEGIN;` and end them with `COMMIT;` or `ROLLBACK;`.
  - Handle transaction errors properly to prevent data corruption.
  - Use savepoints to allow partial rollbacks within a transaction.

- **Indexing:**
  - Create indexes on columns frequently used in `WHERE` clauses and `JOIN` conditions.
  - Avoid over-indexing, as it can slow down write operations.
  - Consider using partial indexes for specific query patterns.
  - Use appropriate index types (e.g., `B-tree`, `Hash`, `GIN`, `GiST`) based on the data and query requirements.
  - Regularly analyze and maintain indexes using `ANALYZE` and `VACUUM`.

- **Query Optimization:**
  - Use `EXPLAIN ANALYZE` to analyze query execution plans and identify performance bottlenecks.
  - Avoid using `SELECT *` and specify only the necessary columns.
  - Use `JOIN` operations instead of subqueries where possible.
  - Optimize `WHERE` clauses to reduce the number of rows processed.
  - Consider using materialized views for frequently executed, complex queries.
  - Rewrite slow performing queries with more efficient alternatives.

- **PL/pgSQL Best Practices:**
  - Keep PL/pgSQL functions and procedures short and focused.
  - Use exception handling to gracefully handle errors.
  - Use `RAISE NOTICE`, `RAISE WARNING`, and `RAISE EXCEPTION` for logging and error reporting.
  - Avoid using cursors unless absolutely necessary; prefer set-based operations.
  - Use the `STRICT` attribute to ensure that a function returns a value.

- **Security Best Practices:**
  - Grant the least privileges necessary to database users.
  - Use roles to manage permissions.
  - Regularly audit database activity.
  - Encrypt sensitive data at rest and in transit.
  - Use the `SECURITY DEFINER` attribute for functions that require elevated privileges.
  - Configure appropriate authentication mechanisms (e.g., `SCRAM-SHA-256`).

- **Connection Management:**
  - Use connection pooling to reduce the overhead of establishing new connections.
  - Close connections when they are no longer needed.
  - Configure connection timeouts to prevent idle connections from consuming resources.

- **Backup and Recovery:**
  - Implement a robust backup and recovery strategy.
  - Regularly back up the database.
  - Test the recovery process to ensure it works as expected.
  - Consider using replication for high availability.

- **Code Organization and Structure:**
  - Organize database objects into schemas based on functionality or application modules.
  - Use version control for database scripts and migrations.
  - Follow a consistent directory structure for database-related files.

- **Common Pitfalls and Gotchas:**
  - Avoid using reserved keywords as identifiers.
  - Be aware of the limitations of data types (e.g., maximum length of `VARCHAR` columns).
  - Handle null values carefully.
  - Test database changes thoroughly before deploying them to production.

- **Tooling and Environment:**
  - Use a code editor with SQL syntax highlighting and autocompletion.
  - Use a database client tool for querying and managing the database (e.g., `pgAdmin`, `DBeaver`, `psql`).
  - Use a version control system (e.g., Git) to manage database scripts and migrations.
  - Use a CI/CD pipeline to automate database deployments.

- **C Coding Standards (When Extending PostgreSQL):**
  - Adhere to the C99 standard. Only use language features available in the C99 standard.
  - Follow PostgreSQL's source code conventions ([PostgreSQL Documentation](https://www.postgresql.org/docs/current/source-conventions.html)).
  - Manage memory carefully; avoid memory leaks.
  - Properly handle errors and exceptions.
  - Document code thoroughly.

- **Use Case specific optimizations:**
  - Time series data: Consider timescaledb extension
  - Geospatial data: Consider postgis extension

@file ./rules/postgresql_security_rules.mdc