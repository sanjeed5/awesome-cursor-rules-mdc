---
description: Definitive guidelines for writing robust, performant, and secure SQLite code. Focuses on schema design, query optimization, and transaction management.
globs: **/*
---
# sqlite Best Practices

SQLite is the go-to embedded SQL engine for local, reliable storage. Adhere to these rules to ensure your SQLite code is maintainable, performant, and secure.

## 1. Data Modeling & Schema Design

Design your schema for integrity and performance from day one.

*   **Primary Keys**: Always use `INTEGER PRIMARY KEY AUTOINCREMENT` for ID columns. This optimizes `rowid` lookups and simplifies ID generation.
    *   ❌ BAD:
        ```sql
        CREATE TABLE users (
            id TEXT PRIMARY KEY, -- Manual UUIDs or similar
            name TEXT NOT NULL
        );
        ```
    *   ✅ GOOD:
        ```sql
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );
        ```

*   **Data Types & Constraints**: Declare appropriate data types and enforce integrity with `NOT NULL`, `UNIQUE`, and `FOREIGN KEY` constraints.
    *   ❌ BAD:
        ```sql
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, -- Allows NULL, no uniqueness
            price REAL
        );
        ```
    *   ✅ GOOD:
        ```sql
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
        );
        ```

*   **Naming Conventions**: Use `lower_case_snake_case` for all table, column, and index names. Avoid SQLite keywords as identifiers.
    *   ❌ BAD: `CREATE TABLE My_Users ( UserId INTEGER PRIMARY KEY );`
    *   ✅ GOOD: `CREATE TABLE my_users ( user_id INTEGER PRIMARY KEY );`

## 2. Performance Considerations

Optimize for speed by minimizing I/O and leveraging the SQLite engine.

*   **Enable WAL Mode**: Always enable Write-Ahead Logging for better concurrency and write performance.
    *   ❌ BAD: Default journal mode (`DELETE`).
    *   ✅ GOOD (at database open or once):
        ```sql
        PRAGMA journal_mode = WAL;
        ```

*   **Relax Synchronous Mode**: When using WAL, set `synchronous` to `NORMAL` for faster commits, accepting minimal risk of data loss on power failure (not app crash).
    *   ❌ BAD: Default `synchronous = FULL`.
    *   ✅ GOOD (at database open or once):
        ```sql
        PRAGMA synchronous = NORMAL;
        ```

*   **Indexes**: Create indexes on columns frequently used in `WHERE`, `ORDER BY`, `GROUP BY`, or `JOIN` clauses. Avoid over-indexing.
    *   ❌ BAD:
        ```sql
        SELECT * FROM users WHERE email = 'test@example.com'; -- No index on email
        ```
    *   ✅ GOOD:
        ```sql
        CREATE INDEX idx_users_email ON users(email);
        SELECT id, name FROM users WHERE email = 'test@example.com';
        ```
    *   **Multi-column Indexes**: For queries filtering/sorting on multiple columns, create a multi-column index matching the query order.
        ```sql
        CREATE INDEX idx_products_category_price ON products(category_id, price);
        SELECT * FROM products WHERE category_id = 1 ORDER BY price DESC;
        ```

*   **Query Optimization**: Select only the columns you need. Push filtering, sorting, and aggregation into SQL.
    *   ❌ BAD:
        ```sql
        SELECT * FROM products; -- Fetch all columns
        -- Then filter/sort in application code
        ```
    *   ✅ GOOD:
        ```sql
        SELECT id, name, price FROM products WHERE stock > 0 ORDER BY price ASC LIMIT 10;
        ```

## 3. Transactions & Concurrency

Ensure data consistency and improve write performance with explicit transactions.

*   **Wrap Writes in Transactions**: Group multiple `INSERT`, `UPDATE`, `DELETE` operations within a single transaction. This significantly reduces disk I/O.
    *   ❌ BAD:
        ```sql
        INSERT INTO logs (action) VALUES ('User created');
        INSERT INTO users (name) VALUES ('New User');
        INSERT INTO logs (action) VALUES ('User name updated');
        UPDATE users SET name = 'Updated User' WHERE id = 1;
        ```
    *   ✅ GOOD:
        ```sql
        BEGIN;
        INSERT INTO logs (action) VALUES ('User created');
        INSERT INTO users (name) VALUES ('New User');
        INSERT INTO logs (action) VALUES ('User name updated');
        UPDATE users SET name = 'Updated User' WHERE id = 1;
        COMMIT;
        ```

*   **Error Handling**: Use `ROLLBACK` to revert all changes if any operation within a transaction fails.
    *   ✅ GOOD:
        ```sql
        BEGIN;
        -- Perform operations
        INSERT INTO users (name) VALUES ('Valid User');
        INSERT INTO users (name) VALUES (NULL); -- This will fail due to NOT NULL
        -- If an error occurs, catch it and:
        ROLLBACK;
        -- Else:
        COMMIT;
        ```

## 4. Security Best Practices

Prevent common vulnerabilities like SQL injection.

*   **Prepared Statements**: Always use prepared statements with bound parameters. NEVER concatenate user input directly into SQL queries.
    *   ❌ BAD:
        ```sql
        String name = userInput.getName();
        String sql = "INSERT INTO users (name) VALUES ('" + name + "');"; // SQL Injection risk!
        ```
    *   ✅ GOOD (using a typical API pattern):
        ```sql
        PreparedStatement stmt = connection.prepareStatement("INSERT INTO users (name) VALUES (?);");
        stmt.setString(1, userInput.getName());
        stmt.executeUpdate();
        ```

*   **Enable Foreign Key Enforcement**: Always enable foreign key constraints at runtime. SQLite defaults to `OFF` for backward compatibility.
    *   ❌ BAD: Forgetting to enable foreign keys, leading to orphaned records.
    *   ✅ GOOD (at database open or once per connection):
        ```sql
        PRAGMA foreign_keys = ON;
        ```

*   **File Permissions**: Store database files in write-protected directories and set restrictive file permissions to limit unauthorized access. This is OS-specific but critical.

## 5. Common Pitfalls & Gotchas

Avoid these common mistakes that lead to bugs and performance issues.

*   **Forgetting `PRAGMA foreign_keys = ON;`**: This is the most common pitfall. Always enable it.
*   **Selecting `*`**: Only retrieve the columns you actually need.
*   **Application-level Filtering/Sorting**: Delegate these operations to SQL for better performance, especially on large datasets.
*   **Not Using Transactions**: Leads to slow writes and potential data inconsistencies.
*   **Using SQLite for High-Concurrency Writes**: SQLite is a single-writer database. If multiple processes need to write concurrently, consider a client-server RDBMS.

## 6. Testing Approaches

Ensure your data access logic is robust and correct.

*   **In-Memory Databases**: Use `:memory:` databases for fast, isolated unit and integration tests of your data access layer.
    *   ✅ GOOD (example in Python, similar patterns exist in other languages):
        ```python
        import sqlite3
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)")
        # ... run tests ...
        conn.close() # Database vanishes
        ```

*   **Seed Data**: Create consistent, reproducible test data for your tests.
*   **Mocking**: For higher-level tests, mock your database interactions to focus on business logic.