---
description: This rule file outlines best practices for developing with DuckDB, covering code organization, performance optimization, security considerations, and testing strategies. It aims to improve code quality, maintainability, and overall project health when using DuckDB.
globs: *.sql,*.ddl,*.md,*.duckdb
---
# DuckDB Best Practices and Coding Standards

This document provides guidelines and best practices for developing with DuckDB. Following these recommendations will lead to more maintainable, performant, and secure DuckDB projects.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

*   **Project Root:**
    *   `.cursor/`: Contains Cursor rule files.
    *   `data/`: Stores data files (CSV, Parquet, etc.).  Consider subdirectories based on data source or type.
    *   `sql/`: Contains SQL scripts (DDL, DML, queries).
    *   `scripts/`: Python, Bash, or other scripts for data processing and automation.
    *   `docs/`: Project documentation.
    *   `tests/`: Test scripts and data.
    *   `venv/` (or `.venv/`): Virtual environment for Python dependencies (if applicable).
    *   `Makefile`: Automation of common tasks (e.g., data loading, testing).
    *   `README.md`: Project overview and instructions.
    *   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

*   **Example:**

    
    my_duckdb_project/
    ├── .cursor/
    │   └── rules.mdc
    ├── data/
    │   ├── raw/
    │   │   └── sales_data.csv
    │   └── processed/
    │       └── sales_summary.parquet
    ├── sql/
    │   ├── create_tables.sql
    │   ├── queries/
    │   │   ├── monthly_sales.sql
    │   │   └── top_customers.sql
    │   └── data_validation.sql
    ├── scripts/
    │   └── process_data.py
    ├── docs/
    │   └── user_guide.md
    ├── tests/
    │   ├── unit/
    │   │   └── test_queries.py
    │   └── integration/
    │       └── test_data_pipeline.py
    ├── venv/
    ├── Makefile
    ├── README.md
    └── .gitignore
    

### 1.2. File Naming Conventions

*   **SQL Scripts:** Use descriptive names with underscores (e.g., `create_tables.sql`, `monthly_sales_report.sql`).
*   **Data Files:** Use names reflecting the data content (e.g., `customer_data.csv`, `product_catalog.parquet`).  Prefix with dates if versioning is needed (e.g., `2024-01-01_customer_data.csv`).
*   **Scripts:** Use names indicating the script's purpose (e.g., `load_data.py`, `run_tests.sh`).
*   **DDL Files:** `schema.sql`, `tables.sql` etc. for defining database structures

### 1.3. Module Organization (if using a scripting language like Python)

*   **Separate Modules:**  Divide code into logical modules (e.g., `data_loading.py`, `query_execution.py`, `utils.py`).
*   **Clear Interfaces:** Define clear functions and classes with well-defined inputs and outputs.
*   **Avoid Global State:** Minimize the use of global variables to improve code testability and maintainability.

### 1.4. Component Architecture

*   **Data Layer:**  Handles data loading, transformation, and storage using DuckDB.
*   **Logic Layer:**  Encapsulates business logic and data processing routines.
*   **Presentation Layer:**  (If applicable)  Presents data to users (e.g., through a web application).
*   **Configuration Layer:** Loads external configurations (e.g., environment variables) that the script needs to run properly.

### 1.5. Code Splitting Strategies

*   **Split Large SQL Scripts:**  Break down complex SQL scripts into smaller, more manageable files.
*   **Modularize Python Code:** Use functions and classes to create reusable components.
*   **Separate Configuration:** Store configuration settings (database paths, API keys) in separate files.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to DuckDB

*   **Embedded Database:**  Use DuckDB as an embedded database for lightweight data analysis and prototyping.
*   **Data Pipeline:**  Build data pipelines with DuckDB for ETL (Extract, Transform, Load) processes.
*   **Analytical Queries:** Leverage DuckDB's columnar storage and vectorized execution for efficient analytical queries.
*   **Federated Querying:** Utilize DuckDB's ability to query data from various sources (CSV, Parquet, JSON, HTTP) using a unified SQL interface.

### 2.2. Recommended Approaches for Common Tasks

*   **Data Loading:**
    *   Use `COPY` command for efficient bulk loading of data from files.
    *   Consider Parquet format for optimal storage and query performance.
    *   Use `read_csv_auto` or `read_parquet` functions.
*   **Querying:**
    *   Write clear and concise SQL queries.
    *   Use parameterized queries to prevent SQL injection.
    *   Utilize window functions for complex aggregations and ranking.
*   **Data Transformation:**
    *   Use SQL functions for data cleaning and transformation.
    *   Create views for commonly used data transformations.
*   **Joining Tables:**
    *   Understand different join types (INNER, LEFT, RIGHT, FULL) and choose the appropriate one.
    *   Ensure join columns are properly indexed for performance.
    *   Use explicit `JOIN` syntax for readability.
*   **Error Handling:** Implement robust error handling to prevent unexpected crashes.
    *   Utilize `TRY/CATCH` blocks in SQL scripts.
    *   Log errors to files or databases for debugging and monitoring.

### 2.3. Anti-patterns and Code Smells to Avoid

*   `SELECT *`:  Avoid selecting all columns unless necessary.  Specify only the columns you need to improve performance.
*   Nested Subqueries:  Excessive use of nested subqueries can impact performance.  Consider using `WITH` clauses (Common Table Expressions - CTEs) to improve readability and performance.
*   Lack of Indexing:  Missing indexes on frequently queried columns can lead to slow query performance.  Add indexes where appropriate.
*   Hardcoded Values:  Avoid hardcoding values in SQL queries or scripts.  Use parameters or configuration files instead.
*   Ignoring Error Handling:  Failing to handle errors gracefully can lead to unexpected application behavior.
*   Over-complicating Queries: Strive for clarity and simplicity in SQL queries. Break down complex logic into smaller, more manageable steps.
*   Not closing database connections after usage.

### 2.4. State Management Best Practices

*   **Minimize State:** Keep the amount of mutable state in your application to a minimum.
*   **Centralized State:** If state is necessary, manage it in a centralized location (e.g., a configuration file or dedicated state management module).
*   **Immutable Data:** Prefer immutable data structures whenever possible to avoid unintended side effects.

### 2.5. Error Handling Patterns

*   **Try-Catch Blocks:** Use `TRY/CATCH` blocks in SQL scripts to handle potential errors.
*   **Logging:**  Log errors to files or databases for debugging and monitoring.
*   **Custom Error Messages:**  Provide informative error messages to help users understand and resolve issues.
*   **Graceful Degradation:** Design your application to handle errors gracefully and continue functioning (albeit with reduced functionality) if possible.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Column Selection:** Use specific column selections instead of `SELECT *` to improve performance.
*   **Filtering Before Joining:** Optimize queries by filtering data before joining tables.
*   **Prepared Statements:** Utilize prepared statements for repeated queries to enhance execution speed.
*   **Indexing:** Create indexes on frequently queried columns to speed up data retrieval.
*   **Query Plan Analysis:** Use `EXPLAIN ANALYZE` to understand query execution plans and identify bottlenecks.
*   **Chunking:** Implement chunking for large datasets to manage memory better.
*   **Data Types:** Use appropriate data types to minimize storage space and improve query performance.  Avoid using TEXT for numerical data. Use INTEGER or BIGINT where possible.
*   **Compression:** DuckDB supports lightweight compression. Consider using persistent databases to take advantage of this.
*   **Parallelism:** DuckDB parallelizes workload based on row groups. Ensure your data is large enough to benefit from multi-core processing. Manually limit the number of threads if necessary using `SET threads = X`.

### 3.2. Memory Management

*   **Memory Limits:** Monitor memory usage and set appropriate memory limits.
*   **Spilling to Disk:** Larger-than-memory workloads are supported by spilling to disk. Configure the temporary directory using `SET temp_directory = '/path/to/temp_dir.tmp/'`.
*   **Avoid Large Intermediate Results:**  Optimize queries to minimize the size of intermediate results.
*   **`preserve_insertion_order`:** Disable `preserve_insertion_order` for large imports/exports if order is not important (`SET preserve_insertion_order = false`).

### 3.3. Avoiding Reading Unnecessary Data (Remote Files)

*   **Column Selection:** Avoid `SELECT *`. Only select columns that are actually used.
*   **Filter Pushdown:** Apply filters on remote parquet files when possible. DuckDB can use these filters to reduce the amount of data that is scanned.
*   **Sorting/Partitioning:** Either sort or partition data by columns that are regularly used for filters: this increases the effectiveness of the filters in reducing IO.
*   **Avoid Reading Data More Than Once:**  If data needs to be accessed multiple times, store it locally.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

*   **SQL Injection:**  Prevent SQL injection by using parameterized queries and avoiding string concatenation in SQL statements.
*   **Data Exposure:**  Protect sensitive data by encrypting it at rest and in transit.  Use appropriate access controls to restrict access to sensitive data.
*   **Denial of Service (DoS):**  Limit resource consumption (memory, CPU) to prevent DoS attacks.
*   **Unauthorized Access:** Implement strong authentication and authorization mechanisms to prevent unauthorized access to the database.

### 4.2. Input Validation

*   **Validate User Inputs:** Validate all user inputs to prevent malicious data from entering the database. Check data types, ranges, and formats.
*   **Sanitize Inputs:** Sanitize user inputs to remove potentially harmful characters or code.

### 4.3. Authentication and Authorization Patterns

*   **Role-Based Access Control (RBAC):** Implement RBAC to grant users only the necessary permissions to access data.
*   **Least Privilege Principle:**  Grant users the minimum level of access required to perform their tasks.
*   **Secure Connection:** Ensure that connections to the database are encrypted using TLS/SSL.

### 4.4. Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Masking:** Mask sensitive data in non-production environments to protect user privacy.
*   **Data Auditing:**  Track all data access and modification activities for auditing purposes.
*   **Regular Backups:** Regularly back up the database to prevent data loss in case of failures.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

*   **Test Individual Components:**  Write unit tests to verify the functionality of individual SQL functions, Python modules, or other components.
*   **Mock Database Connections:** Mock database connections to isolate the code being tested from the actual database.
*   **Assertion Frameworks:** Use assertion frameworks (e.g., `pytest`, `unittest`) to verify expected results.

### 5.2. Integration Testing

*   **Test Interactions:**  Write integration tests to verify the interactions between different components of the system.
*   **Real Database:** Use a real DuckDB database for integration tests (preferably a test database).
*   **Data Validation:** Verify that data is correctly loaded, transformed, and stored in the database.

### 5.3. End-to-End Testing

*   **Test Complete Workflows:** Write end-to-end tests to verify the functionality of complete workflows from start to finish.
*   **Simulate User Actions:** Simulate user actions to test the system's behavior under realistic conditions.

### 5.4. Test Organization

*   **Separate Test Directory:** Create a separate `tests/` directory to store test scripts.
*   **Organize Tests:** Organize tests into subdirectories based on the component or feature being tested.
*   **Descriptive Names:** Use descriptive names for test files and functions.

### 5.5. Mocking and Stubbing

*   **Mock External Dependencies:** Mock external dependencies (e.g., file systems, network connections) to isolate the code being tested.
*   **Stub Database Interactions:** Stub database interactions to control the behavior of the database during testing.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

*   **Not Using Indexes:**  Forgetting to create indexes on frequently queried columns.
*   **Incorrect Join Types:** Using the wrong join type (e.g., INNER instead of LEFT).
*   **SQL Injection Vulnerabilities:** Failing to prevent SQL injection attacks.
*   **Memory Leaks:** Not properly closing database connections or releasing resources.
*   **Lack of Error Handling:**  Ignoring potential errors in SQL scripts or Python code.

### 6.2. Edge Cases to Be Aware Of

*   **Null Values:**  Handle null values correctly in SQL queries and data transformations.
*   **Empty Datasets:**  Test the system's behavior with empty datasets.
*   **Large Datasets:**  Test the system's behavior with large datasets to ensure performance.
*   **Concurrency:** Consider potential concurrency issues when multiple users or processes access the database simultaneously.

### 6.3. Version-Specific Issues

*   **Check Release Notes:**  Review the release notes for each new version of DuckDB to identify any breaking changes or known issues.
*   **Compatibility Testing:** Perform compatibility testing to ensure that the application works correctly with different versions of DuckDB.

### 6.4. Compatibility Concerns

*   **Operating System:** Ensure that the application is compatible with the target operating system (Windows, macOS, Linux).
*   **Programming Language:**  Ensure that the application is compatible with the programming language being used (Python, Java, etc.).
*   **Database Drivers:** Use compatible database drivers.

### 6.5. Debugging Strategies

*   **Logging:** Add logging statements to the code to track execution flow and variable values.
*   **Print Statements:** Use print statements to display intermediate results and debug issues.
*   **Debuggers:** Use debuggers (e.g., `pdb` in Python) to step through the code and inspect variables.
*   **Query Plan Analysis:** Use `EXPLAIN ANALYZE` to understand query execution plans and identify bottlenecks.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **Text Editor/IDE:** VSCode, Sublime Text, IntelliJ IDEA, etc.
*   **Database Client:** DBeaver, DataGrip, SQL Developer, DuckDB CLI.
*   **Version Control:** Git.
*   **Virtual Environment Manager:** `venv` (Python), Conda.
*   **Task Runner:** Make, Invoke (Python).

### 7.2. Build Configuration

*   **Dependency Management:** Use a dependency management tool (e.g., `pip` for Python) to manage project dependencies.
*   **Configuration Files:**  Use configuration files (e.g., `config.ini`, `settings.json`) to store project settings and parameters.

### 7.3. Linting and Formatting

*   **Linters:** Use linters (e.g., `flake8`, `pylint` for Python, `sqlfluff` for SQL) to enforce code style and identify potential errors.
*   **Formatters:** Use formatters (e.g., `black` for Python, `sqlformat` for SQL) to automatically format code according to a defined style.

### 7.4. Deployment Best Practices

*   **Containerization:** Use containerization (e.g., Docker) to package the application and its dependencies into a single unit.
*   **Infrastructure as Code (IaC):** Use IaC tools (e.g., Terraform, CloudFormation) to automate the provisioning and management of infrastructure.
*   **Configuration Management:** Use configuration management tools (e.g., Ansible, Chef, Puppet) to automate the configuration of servers and applications.

### 7.5. CI/CD Integration

*   **Continuous Integration (CI):** Integrate automated testing into the CI pipeline to ensure code quality.
*   **Continuous Delivery (CD):** Automate the deployment process to ensure frequent and reliable releases.