---
description: This guide provides opinionated, actionable best practices for writing high-performance, maintainable, and robust DuckDB SQL queries and scripts, focusing on modern analytical workloads.
globs: **/*
---
# duckdb Best Practices

DuckDB is our go-to for analytical workloads. These rules ensure our DuckDB scripts are not only performant but also readable, reproducible, and easy to maintain. Adhere to these guidelines for all new and refactored DuckDB code.

## 1. Code Organization and Structure

### 1.1. Standardize Script Headers with Pragmas

Always start DuckDB scripts with a clear header that sets session-level pragmas, imports necessary extensions, and defines initial data sources. This ensures consistent execution environments.

❌ **BAD:** Inconsistent settings, implicit behavior.
```sql
-- analytics/daily_summary.sql
SELECT * FROM read_parquet('s3://my-bucket/data.parquet');
```

✅ **GOOD:** Explicitly configured environment.
```sql
-- analytics/daily_summary.sql
PRAGMA threads=4; -- Allocate 4 CPU threads for this session
PRAGMA memory_limit='8GB'; -- Limit memory usage to 8GB
INSTALL httpfs; -- Install S3/HTTP(S) extension
LOAD httpfs; -- Load the extension for use

-- Subsequent queries will benefit from these settings
SELECT
    date_trunc('day', event_timestamp) AS day,
    COUNT(*) AS daily_events
FROM read_parquet('s3://my-bucket/events_2024.parquet')
GROUP BY 1;
```

### 1.2. Structure Complex Logic with CTEs

Use Common Table Expressions (CTEs) to break down complex queries into logical, readable steps. This improves clarity, debugging, and often allows DuckDB's optimizer to produce better plans.

❌ **BAD:** Nested subqueries are hard to read and debug.
```sql
SELECT
    final.region,
    final.avg_sales
FROM (
    SELECT
        sub.region,
        AVG(sub.total_sales) AS avg_sales
    FROM (
        SELECT
            o.region,
            SUM(li.price * li.quantity) AS total_sales
        FROM orders o
        JOIN line_items li ON o.order_id = li.order_id
        WHERE o.order_date >= '2024-01-01'
        GROUP BY o.region
    ) sub
    GROUP BY sub.region
) final
WHERE final.avg_sales > 1000;
```

✅ **GOOD:** CTEs provide clear, modular steps.
```sql
WITH regional_sales AS (
    SELECT
        o.region,
        SUM(li.price * li.quantity) AS total_sales
    FROM orders o
    JOIN line_items li ON o.order_id = li.order_id
    WHERE o.order_date >= '2024-01-01'
    GROUP BY o.region
),
average_regional_sales AS (
    SELECT
        region,
        AVG(total_sales) AS avg_sales
    FROM regional_sales
    GROUP BY region
)
SELECT
    region,
    avg_sales
FROM average_regional_sales
WHERE avg_sales > 1000;
```

### 1.3. Adopt a Consistent SQL Style

Enforce a consistent SQL style across the team. We use `SQLFluff` with a shared `.sqlfluff` configuration (similar to dbt's style guide). This enhances readability and reduces cognitive load.

❌ **BAD:** Inconsistent capitalization, indentation, and comma placement.
```sql
select id,name,
email from users where status = 'active'
ORDER BY name asc
```

✅ **GOOD:** Standardized, SQLFluff-compliant style.
```sql
SELECT
    id,
    name,
    email
FROM
    users
WHERE
    status = 'active'
ORDER BY
    name ASC;
```

## 2. Performance Considerations

### 2.1. Explicitly Set Session Pragmas

Always configure `PRAGMA threads` and `PRAGMA memory_limit` at the beginning of your script. Tailor these to the execution environment (e.g., local machine vs. CI/CD runner) to prevent resource exhaustion or underutilization.

❌ **BAD:** Relying on default settings, leading to unpredictable performance.
```sql
-- Potentially slow on large datasets if defaults are low
SELECT COUNT(*) FROM large_table;
```

✅ **GOOD:** Optimized for the expected workload.
```sql
PRAGMA threads=8; -- Use 8 threads for parallel processing
PRAGMA memory_limit='16GB'; -- Allocate sufficient memory
PRAGMA enable_external_access=true; -- Allow reading external files (e.g., S3)

SELECT COUNT(*) FROM large_table;
```

### 2.2. Use `COPY ... FROM` for Efficient Data Loading

For bulk loading data from CSV, Parquet, or JSON files, `COPY ... FROM` is the most performant method. It leverages DuckDB's vectorized engine and handles schema inference efficiently.

❌ **BAD:** Manual `INSERT` statements or `CREATE TABLE AS SELECT FROM read_csv(...)` for large files.
```sql
-- Inefficient for large CSV files
CREATE TABLE my_data AS SELECT * FROM read_csv_auto('data.csv');
```

✅ **GOOD:** `COPY ... FROM` for speed and robustness.
```sql
CREATE TABLE my_data (
    id INTEGER,
    name VARCHAR,
    value DOUBLE
);
COPY my_data FROM 'data.csv' (HEADER true, DELIMITER ',');
```
Or, for direct querying without creating a persistent table:
```sql
SELECT * FROM read_csv_auto('data.csv'); -- DuckDB optimizes this internally
```

### 2.3. Understand When to Use Indexes

DuckDB is a columnar database, and its primary performance comes from vectorized scans. B-tree indexes are useful for highly selective point lookups or small range scans on large tables, but they are *not* a silver bullet. Avoid over-indexing.

❌ **BAD:** Creating indexes indiscriminately, adding overhead without benefit.
```sql
CREATE TABLE events (
    event_id UUID,
    event_type VARCHAR,
    timestamp TIMESTAMP,
    user_id INTEGER
);
CREATE INDEX event_type_idx ON events (event_type); -- Likely unnecessary
CREATE INDEX timestamp_idx ON events (timestamp); -- Potentially useful for specific range queries
```

✅ **GOOD:** Indexing only when a selective predicate benefits.
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username VARCHAR,
    email VARCHAR UNIQUE
);
-- Index on user_id is implicit with PRIMARY KEY
-- Index on email is implicit with UNIQUE
-- These are useful for direct lookups: WHERE user_id = 123 or WHERE email = 'test@example.com'

-- For a large fact table where we frequently filter by a specific dimension:
CREATE TABLE fact_sales (
    sale_id UUID,
    product_id INTEGER,
    sale_date DATE,
    amount DECIMAL(10, 2)
);
-- If we frequently query `WHERE product_id = X` and `product_id` has high cardinality:
CREATE INDEX product_id_btree_idx ON fact_sales (product_id);
```

## 3. Common Patterns and Anti-patterns

### 3.1. Embrace "Friendly SQL" but Maintain Clarity

DuckDB's "Friendly SQL" dialect allows for relaxed syntax (e.g., omitting `FROM` for `SELECT 1;`). While convenient, prioritize explicit and clear statements for production code.

❌ **BAD:** Overly terse syntax that sacrifices readability.
```sql
-- What is this querying?
SELECT * FROM 'data.parquet' WHERE col1 > 10;
```

✅ **GOOD:** Explicitly define sources and aliases.
```sql
-- Clear data source and alias
SELECT
    t.column_a,
    t.column_b
FROM
    read_parquet('data.parquet') AS t
WHERE
    t.column_c > 10;
```

### 3.2. Prefer `time_bucket` for Flexible Windowing

For time-series analysis, `time_bucket` offers more control over window alignment and offset than `date_trunc`, especially for non-standard intervals.

❌ **BAD:** `date_trunc` for arbitrary window sizes.
```sql
-- Only truncates to standard units (hour, day, etc.)
SELECT
    date_trunc('hour', event_time) AS window_start,
    COUNT(*)
FROM events
GROUP BY 1;
```

✅ **GOOD:** `time_bucket` for custom, flexible windows.
```sql
-- Tumbling window of 15 minutes, starting at 00:00:00
SELECT
    time_bucket(INTERVAL 15 MINUTE, event_time) AS window_start,
    COUNT(*) AS event_count
FROM events
GROUP BY 1
ORDER BY 1;

-- Hopping window of 30 minutes, hopping every 10 minutes
WITH time_range AS (
    SELECT
        range AS window_start,
        range + INTERVAL 30 MINUTE AS window_end
    FROM range(
        '2024-01-01 00:00:00'::TIMESTAMP,
        '2024-01-02 00:00:00'::TIMESTAMP,
        INTERVAL 10 MINUTE -- hopping size
    )
)
SELECT
    tr.window_start,
    tr.window_end,
    COUNT(e.event_time) AS event_count
FROM time_range tr
LEFT JOIN events e ON e.event_time >= tr.window_start AND e.event_time < tr.window_end
GROUP BY ALL
ORDER BY 1;
```

### 3.3. Avoid `SELECT *` in Production Queries

Always explicitly list columns in `SELECT` statements. This prevents unexpected schema changes from breaking downstream consumers and improves query performance by only fetching necessary data.

❌ **BAD:** Fragile and inefficient.
```sql
SELECT * FROM my_table;
```

✅ **GOOD:** Robust and explicit.
```sql
SELECT
    id,
    name,
    created_at
FROM
    my_table;
```

## 4. Query Optimization

### 4.1. Leverage Window Functions for Time-Series Analytics

DuckDB excels at window functions. Use them for calculating running totals, moving averages, or ranking within partitions, especially for temporal data.

❌ **BAD:** Self-joins or subqueries for cumulative metrics.
```sql
-- Inefficient for large datasets
SELECT
    e1.event_date,
    COUNT(e1.id) AS daily_events,
    (SELECT COUNT(e2.id) FROM events e2 WHERE e2.event_date <= e1.event_date) AS cumulative_events
FROM events e1
GROUP BY e1.event_date
ORDER BY e1.event_date;
```

✅ **GOOD:** Efficient window functions.
```sql
SELECT
    event_date,
    COUNT(id) AS daily_events,
    SUM(COUNT(id)) OVER (ORDER BY event_date) AS cumulative_events
FROM events
GROUP BY event_date
ORDER BY event_date;
```

### 4.2. Use `QUALIFY` for Filtering Window Function Results

`QUALIFY` provides a concise and efficient way to filter results based on window function output, similar to `HAVING` for aggregates.

❌ **BAD:** Wrapping window functions in subqueries for filtering.
```sql
WITH ranked_events AS (
    SELECT
        user_id,
        event_time,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS rn
    FROM events
)
SELECT
    user_id,
    event_time
FROM ranked_events
WHERE rn = 1;
```

✅ **GOOD:** `QUALIFY` for direct filtering.
```sql
SELECT
    user_id,
    event_time
FROM events
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) = 1;
```

## 5. Data Modeling

### 5.1. Define Explicit Schemas and Types

Always define explicit column names and data types when creating tables. This prevents auto-detection errors, ensures data quality, and improves query planning.

❌ **BAD:** Relying solely on `read_csv_auto` for schema.
```sql
-- Schema inferred, might not be optimal or correct
CREATE TABLE my_data AS SELECT * FROM read_csv_auto('data.csv');
```

✅ **GOOD:** Explicit schema definition.
```sql
CREATE TABLE my_data (
    transaction_id UUID PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    transaction_date DATE,
    amount DECIMAL(10, 2),
    description VARCHAR
);
COPY my_data FROM 'data.csv' (HEADER true, DELIMITER ',');
```

## 6. Testing Approaches

### 6.1. Implement `SQLLogictest` for Critical SQL Logic

For critical analytical queries and transformations, use `SQLLogictest` to ensure correctness and prevent regressions. Store `.sql` test files alongside your query files.

❌ **BAD:** Manual verification or ad-hoc assertions.
```sql
-- No automated test for this logic
SELECT SUM(amount) FROM daily_sales WHERE sales_date = '2024-01-01';
```

✅ **GOOD:** Automated testing with `SQLLogictest`.

`tests/test_daily_sales.test` (SQLLogictest file):
```
statement ok
CREATE TABLE daily_sales (sales_date DATE, amount DECIMAL(10,2));

statement ok
INSERT INTO daily_sales VALUES ('2024-01-01', 100.00), ('2024-01-01', 50.00), ('2024-01-02', 200.00);

query II
SELECT sales_date, SUM(amount) FROM daily_sales GROUP BY sales_date ORDER BY sales_date;
----
2024-01-01 150.00
2024-01-02 200.00
```