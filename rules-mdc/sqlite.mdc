---
description: This rule provides comprehensive guidance for SQLite development, covering best practices for schema design, performance optimization, security, testing, and more. It aims to ensure efficient, secure, and maintainable SQLite database applications.
globs: *.db,*.sqlite,*.sqlite3,*.sql
---
# SQLite Best Practices and Coding Standards

This document outlines the best practices and coding standards for developing with SQLite. It covers various aspects, including schema design, performance optimization, security considerations, testing strategies, common pitfalls, and tooling.

## 1. Code Organization and Structure

### 1.1. Directory Structure

A well-organized directory structure improves code maintainability and readability.


project_root/
├── data/
│   ├── production.db # Production database
│   ├── development.db # Development database
│   ├── test.db        # Test database
│   └── migrations/   # Directory for database schema migrations
│       ├── 001_initial_schema.sql
│       └── 002_add_indexes.sql
├── src/
│   ├── database.py    # Database connection and helper functions
│   ├── models.py        # Data models (if using an ORM)
│   ├── queries.py       # Reusable SQL queries
│   └── utils.py         # Utility functions
├── tests/
│   ├── conftest.py    # pytest configuration
│   ├── test_database.py # Tests for database operations
│   └── test_models.py   # Tests for data models
├── .env               # Environment variables
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies


### 1.2. File Naming Conventions

*   Database files: Use `.db`, `.sqlite`, or `.sqlite3` extensions (e.g., `mydatabase.db`).
*   SQL migration files: Prefix with a sequence number and use descriptive names (e.g., `001_create_users_table.sql`).
*   Python modules: Use lowercase with underscores (e.g., `database.py`, `models.py`).

### 1.3. Module Organization

*   **Database Connection Module:**  Encapsulate database connection logic within a dedicated module. This promotes reusability and simplifies connection management. The module should handle connection establishment, cursor creation, and resource cleanup.

    python
    # database.py
    import sqlite3

    DATABASE_PATH = "./data/mydatabase.db"

    def get_db_connection():
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def close_db_connection(conn):
        if conn:
            conn.close()
    

*   **Data Models Module:** Define Python classes or data structures that represent database tables. This abstraction simplifies data access and manipulation, especially when using an ORM. Data models can include validation and serialization logic.

    python
    # models.py
    class User:
        def __init__(self, id, username, email):
            self.id = id
            self.username = username
            self.email = email
    

*   **Queries Module:** Store reusable SQL queries in a separate module. This promotes code reuse, reduces redundancy, and makes it easier to maintain queries. Queries can be parameterized to prevent SQL injection vulnerabilities.

    python
    # queries.py
    CREATE_USER = """INSERT INTO users (username, email) VALUES (?, ?);"""
    GET_USER_BY_ID = """SELECT id, username, email FROM users WHERE id = ?;"""
    

### 1.4. Component Architecture

A layered architecture helps separate concerns and improve testability.

*   **Data Access Layer:**  Manages interaction with the database. Contains functions for executing queries, retrieving data, and updating records. Uses the database connection and queries modules.
*   **Business Logic Layer:** Contains the application's core logic. Processes data from the data access layer and implements business rules. Interacts with data models.
*   **Presentation Layer:** Handles user interface and input/output. Communicates with the business logic layer to display data and receive user input.

### 1.5. Code Splitting Strategies

*   **Feature-based Splitting:** Group related code into modules or packages based on application features (e.g., user management, product catalog). This improves modularity and simplifies navigation.
*   **Functional Splitting:** Separate code based on functionality (e.g., data access, business logic, UI components). This promotes reuse and simplifies testing.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Repository Pattern:** Abstract the data access layer behind an interface. Allows you to easily switch data sources or implement caching.
*   **Data Mapper Pattern:**  Transfer data between domain objects and the database. Useful when working with complex data structures.
*   **Unit of Work Pattern:** Track changes to domain objects within a transaction. Ensures data consistency and simplifies transaction management.

### 2.2. Recommended Approaches for Common Tasks

*   **Connecting to the database:**  Use a connection pool to efficiently manage database connections, especially in multi-threaded environments.  See example above in Module Organization.
*   **Executing queries:** Use parameterized queries to prevent SQL injection. Always close cursors and connections when finished.
*   **Fetching data:** Use `fetchall()` or iterate over the cursor to retrieve all results. Use `fetchone()` to retrieve a single row.
*   **Handling transactions:** Use `conn.commit()` to save changes and `conn.rollback()` to undo changes in case of errors.  Use `with conn:` to ensure transactions are handled properly.

### 2.3. Anti-patterns and Code Smells

*   **Hardcoding SQL queries:**  Avoid embedding SQL queries directly within application code. Use parameterized queries and store queries in a separate module.
*   **Ignoring errors:** Always check for errors after executing SQL statements. Use try-except blocks to handle exceptions gracefully.
*   **Leaking database connections:** Ensure that connections are closed properly after use. Use context managers (`with conn:`) to automate connection management.
*   **Over-fetching data:** Only retrieve the columns that are needed for a particular operation. Use `EXPLAIN QUERY PLAN` to optimize queries.
*   **Using string concatenation for SQL queries:** **Never** use string concatenation to build SQL queries. This is a major security vulnerability that can lead to SQL injection attacks. Always use parameterized queries.

### 2.4. State Management

*   **Connection State:** Manage database connections at the application level. Use a single connection object for the entire application or connection pooling to efficiently manage multiple connections.
*   **Transaction State:**  Use transactions to ensure data consistency. Commit changes only when all operations are successful. Rollback changes in case of errors.
*   **Data Caching:** Cache frequently accessed data in memory to improve performance. Use a caching library or implement a custom caching mechanism.

### 2.5. Error Handling

*   **Use try-except blocks:** Wrap database operations in try-except blocks to catch potential exceptions.
*   **Log errors:** Log error messages to a file or console for debugging purposes.
*   **Rollback transactions:** If an error occurs during a transaction, rollback the transaction to prevent data corruption.
*   **Raise custom exceptions:** Define custom exceptions to represent specific database errors.  This allows for more fine-grained error handling in higher-level code.

    python
    class DatabaseError(Exception):
        pass

    try:
        with conn:
            cursor.execute("INSERT INTO users (username) VALUES (?)", ("invalid username",))
    except sqlite3.IntegrityError as e:
        raise DatabaseError(f"Failed to insert user: {e}")
    

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Schema Design:** Keep the database schema simple and avoid unnecessary generalization. Use strict tables to enforce type checking and ensure primary keys are non-nullable.
*   **Indexing:** Create indexes on columns that are frequently used in `WHERE` clauses or `JOIN` conditions.
*   **Prepared Statements:** Prepare SQL statements and bind values to reduce parsing overhead. Use transactions for batch operations, and prefer temporary tables for intermediate results. Avoid excessive use of indices and sub-queries, as they can degrade performance.
*   **Transactions:** Use transactions to group multiple operations into a single atomic unit.
*   **Appropriate Data Types:** Choosing the right data type can significantly impact performance.  For numeric primary keys, always use the `INTEGER` type for `rowid` aliasing.  Limit the size of TEXT columns to prevent excessive storage usage.
*   **Avoid `SELECT *`:**  Instead, specify only the required columns.  This reduces the amount of data transferred from the database.
*   **Use `EXPLAIN QUERY PLAN`:**  Analyze query execution plans to identify performance bottlenecks.  This allows you to optimize queries by adding indexes or rewriting the query.
*   **Consider `WITHOUT ROWID` tables:** For tables where the `rowid` is not needed and all columns are part of the primary key, using `WITHOUT ROWID` can save space and improve performance.
*   **Optimize table order in joins**:  Place tables without indexes on join columns at the far left of the SELECT statement.
*   **Use `pragma optimize`**: Run this statement to optimize the database after making many changes.

### 3.2. Memory Management

*   **Connection Pooling:** Use a connection pool to reuse database connections and reduce connection overhead.
*   **Cursor Management:** Close cursors after use to release resources. Use context managers to automatically close cursors.
*   **Result Set Size:** Limit the size of result sets to prevent excessive memory usage. Use pagination or filtering to reduce the amount of data retrieved.
*   **Bulk operations:**  Use `executemany` for bulk inserts or updates instead of looping and executing individual statements.
*   **Large Blobs:** If storing large binary data (BLOBs), consider storing them as separate files and only store the file path in the database.

### 3.3. Bundle Size Optimization (Applicable for applications bundling SQLite library)

*   **Strip Debug Symbols:** Remove debug symbols from the SQLite library to reduce its size.
*   **Enable Compile-Time Options:**  Disable unused SQLite features to reduce binary size.  Compile with options like `-DSQLITE_OMIT_...` to remove features that are not needed.
*   **Use a Minimal Build:**  If possible, use a pre-built SQLite library that is optimized for size.

### 3.4. Lazy Loading

*   **Lazy Loading Relationships:** Load related data only when it is needed. This can improve performance, especially when dealing with complex relationships.
*   **Virtual Tables:** Use virtual tables to access data from external sources on demand. This can reduce memory usage and improve performance.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

*   **SQL Injection:**  The most common vulnerability. Prevent it by using parameterized queries (also known as prepared statements) exclusively. Never construct SQL queries using string concatenation with user-provided input.
*   **Database File Access:**  Restrict access to the SQLite database file to only the necessary users and processes.  Use appropriate file system permissions to prevent unauthorized access.
*   **Denial of Service (DoS):** Limit the amount of memory and CPU time that SQLite can consume to prevent DoS attacks. Use the `sqlite3_limit()` and `sqlite3_hard_heap_limit64()` functions.
*   **Malicious Database Files:**  Be cautious when reading SQLite database files from untrusted sources.  A malicious database file can contain triggers or virtual tables that execute arbitrary code.  Use `PRAGMA integrity_check` and `PRAGMA quick_check` to verify the database file's integrity.

### 4.2. Input Validation

*   **Sanitize User Input:**  Always sanitize user input before using it in SQL queries.  This includes escaping special characters and validating data types.
*   **Use Constraints:**  Define constraints on database columns to enforce data integrity.  This can help prevent invalid data from being inserted into the database.
*   **Limit Input Lengths:**  Limit the length of text fields to prevent buffer overflows and DoS attacks.  Always define a maximum length on `TEXT` columns.

### 4.3. Authentication and Authorization

*   **SQLite Limitations:** SQLite itself does not provide built-in authentication or authorization mechanisms.
*   **Application-Level Implementation:** Implement authentication and authorization logic in the application code that interacts with the database.  This can involve checking user credentials against a stored hash or using a role-based access control system.
*   **Encryption:** Encrypt the SQLite database file to protect sensitive data. Use a library like SQLCipher to encrypt the database file.

### 4.4. Data Protection

*   **Encryption at Rest:**  Encrypt the database file to protect sensitive data when it is stored on disk. This protects against unauthorized access to the database file itself.
*   **Encryption in Transit:**  Use SSL/TLS to encrypt communication between the application and the database, especially if the database is accessed over a network (though rare with SQLite).
*   **Data Masking:** Mask sensitive data in query results or log files to prevent accidental disclosure. Use SQL functions or application-level logic to mask data.
*   **Backup and Recovery:** Regularly back up the SQLite database file to prevent data loss. Store backups in a secure location.

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication to protect data in transit.
*   **API Keys:** Use API keys to authenticate API requests.
*   **Rate Limiting:** Implement rate limiting to prevent DoS attacks.
*   **Input Validation:** Validate all API inputs to prevent injection attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Individual Components:** Unit tests should focus on testing individual components of the application in isolation.  This includes testing data models, data access functions, and business logic.
*   **Mock Database:** Use a mock database (e.g., an in-memory SQLite database) to isolate tests from the real database.  This allows you to run tests quickly and reliably without affecting the production database.
*   **Verify Queries:** Assert that the correct SQL queries are executed by the data access functions.  This can be done by capturing the queries executed by the cursor and comparing them to expected values.
*   **Test Edge Cases:**  Test edge cases and error conditions to ensure that the application handles them gracefully.  This includes testing invalid data inputs, database connection errors, and transaction failures.
*   **Use `pytest` fixtures:** Employ `pytest` fixtures to set up and tear down the test environment (e.g., create and populate the mock database).

    python
    # tests/conftest.py
    import pytest
    import sqlite3
    from src.database import get_db_connection, close_db_connection

    @pytest.fixture
    def test_db():
        conn = sqlite3.connect(':memory:')  # In-memory database
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
        cursor.execute("INSERT INTO users (username) VALUES (?)", ("testuser",))
        conn.commit()
        yield conn
        conn.close()

    # tests/test_database.py
    def test_get_user(test_db):
        cursor = test_db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", ("testuser",))
        user = cursor.fetchone()
        assert user['username'] == "testuser"
    

### 5.2. Integration Testing

*   **Test Interactions:** Integration tests should focus on testing the interactions between different components of the application.  This includes testing the interactions between the data access layer, the business logic layer, and the presentation layer.
*   **Use a Test Database:** Use a separate test database for integration tests. This prevents integration tests from affecting the production database.
*   **Verify Data Integrity:**  Verify that data is correctly stored and retrieved from the database.  This includes testing data validation, data transformations, and data relationships.

### 5.3. End-to-End Testing

*   **Test the Entire Application:** End-to-end tests should focus on testing the entire application from the user interface to the database.  This includes testing user workflows, data flows, and system integrations.
*   **Use a Production-Like Environment:** Use a production-like environment for end-to-end tests.  This ensures that the tests are run in an environment that is similar to the production environment.
*   **Automate Tests:** Automate end-to-end tests to ensure that they are run regularly and consistently.

### 5.4. Test Organization

*   **Separate Test Files:** Create separate test files for each component or module of the application. This makes it easier to organize and maintain tests.
*   **Descriptive Test Names:** Use descriptive test names that clearly indicate what is being tested. This makes it easier to understand test results and debug failures.
*   **Follow Arrange-Act-Assert Pattern:** Structure tests according to the Arrange-Act-Assert pattern. This makes tests more readable and maintainable.
*   **Keep Tests Independent:** Ensure that tests are independent of each other. This prevents tests from interfering with each other and makes it easier to run tests in parallel.

### 5.5. Mocking and Stubbing

*   **Mock External Dependencies:** Use mocking to isolate tests from external dependencies, such as network services or file systems. This allows you to test components in isolation without relying on external resources.
*   **Stub Database Calls:** Use stubs to replace database calls with pre-defined results. This allows you to test components that interact with the database without actually accessing the database.
*   **Verify Interactions:** Verify that the correct methods are called on mock objects or stubs. This ensures that components are interacting with each other as expected.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **SQL Injection:** Failing to use parameterized queries.
*   **Data Type Mismatches:** Inserting data with incorrect data types into database columns. Strict mode and explicit type definitions can help mitigate this.
*   **Concurrency Issues:**  Not handling concurrency correctly in multi-threaded environments.  Use connection pooling and proper transaction management.
*   **Deadlocks:** Creating deadlocks by acquiring locks in the wrong order. Be aware of transaction isolation levels and lock contention.
*   **Schema Evolution:** Failing to plan for schema evolution. Use database migrations to manage schema changes.

### 6.2. Edge Cases

*   **Large Databases:** Handling databases with large amounts of data. Use pagination, indexing, and query optimization techniques.
*   **Concurrent Access:** Handling concurrent access to the database. Use connection pooling, transactions, and locking mechanisms.
*   **Corrupted Database Files:**  Handling corrupted database files.  Use `PRAGMA integrity_check` and implement a backup and recovery strategy.
*   **Full Disk:** Handle situations where disk space is exhausted and database writes fail.

### 6.3. Version-Specific Issues

*   **Compatibility Changes:** Be aware of compatibility changes between different versions of SQLite. Consult the SQLite documentation for details.
*   **New Features:** Take advantage of new features in newer versions of SQLite. This can improve performance, security, or functionality.

### 6.4. Compatibility Concerns

*   **Cross-Platform Compatibility:** Ensure that the application is compatible with different operating systems and architectures. Test the application on different platforms.
*   **Data Type Differences:** Be aware of data type differences between SQLite and other database systems. Use compatible data types or perform data conversions as needed.

### 6.5. Debugging Strategies

*   **Logging:** Use logging to track database operations and errors. Log SQL queries, execution times, and error messages.
*   **Debugging Tools:** Use debugging tools to inspect the database schema, data, and query execution plans. Tools like DB Browser for SQLite are very useful.
*   **Error Messages:** Pay attention to error messages. Error messages can provide valuable clues about the cause of the problem.
*   **Simplify Queries:** Simplify complex queries to isolate the source of the problem.
*   **Replicate the Issue:** Try to replicate the issue in a controlled environment. This can help you identify the root cause of the problem.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **DB Browser for SQLite:** A graphical tool for managing SQLite databases.
*   **SQLite command-line shell:**  A command-line tool for interacting with SQLite databases.  Available as `sqlite3`.
*   **Python sqlite3 module:** Python's built-in module for working with SQLite databases.
*   **SQLCipher:** An open-source extension to SQLite that provides transparent encryption.
*   **ORM (Object-Relational Mapper):**  Libraries like SQLAlchemy or PonyORM can simplify database interactions, but use caution and understand the underlying SQL generated, as ORMs can sometimes lead to inefficient queries.

### 7.2. Build Configuration

*   **Dependencies:** Manage dependencies using a dependency management tool (e.g., `pip`, `poetry`, `conda`).
*   **Compiler Options:** Use appropriate compiler options to optimize the SQLite library for performance.
*   **Linking:** Link the SQLite library statically or dynamically depending on the application requirements.

### 7.3. Linting and Formatting

*   **SQL Linters:** Use SQL linters to enforce coding standards and identify potential errors in SQL queries. Use tools like `sqlfluff`.
*   **Code Formatters:** Use code formatters to automatically format SQL queries and code. Use formatters available through IDE extensions or command-line tools.
*   **Consistency:** Maintain consistency in code formatting and style across the project.

### 7.4. Deployment

*   **Database File Location:** Store the SQLite database file in a secure location on the server.
*   **File Permissions:** Set appropriate file permissions on the database file to prevent unauthorized access.
*   **Backup Strategy:** Implement a backup and recovery strategy to protect against data loss.
*   **Connection Limits:** Configure connection limits to prevent denial-of-service attacks.

### 7.5. CI/CD Integration

*   **Automated Tests:** Integrate automated tests into the CI/CD pipeline to ensure that code changes do not break existing functionality.
*   **Database Migrations:** Automate database migrations as part of the deployment process.
*   **Rollback Strategy:** Implement a rollback strategy to revert to a previous version of the application in case of deployment failures.

By following these best practices and coding standards, developers can create efficient, secure, and maintainable SQLite database applications.