---
description: This guide defines the definitive best practices for writing clean, performant, and maintainable PostgreSQL SQL, focusing on modern conventions and avoiding common pitfalls.
globs: **/*
---
# PostgreSQL Best Practices

This document outlines the mandatory best practices for all PostgreSQL SQL development. Adherence ensures consistency, readability, performance, and maintainability across our codebase.

## 1. Code Organization and Structure

### 1.1. Naming Conventions
Always use `snake_case` for all database identifiers (tables, columns, functions, schemas). SQL keywords must be lowercase. Keep names descriptive but concise.

❌ **BAD**:
```sql
CREATE TABLE UserData (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(255)
);

SELECT UserID, UserName FROM UserData;
```

✅ **GOOD**:
```sql
create table user_data (
    user_id bigint generated always as identity primary key,
    user_name text not null
);

select user_id, user_name from user_data;
```

### 1.2. Formatting
Structure queries for maximum readability. Root keywords (`SELECT`, `FROM`, `WHERE`) belong on their own line, with arguments indented. Use explicit `AS` for all aliases.

❌ **BAD**:
```sql
SELECT t.client_id, DATE(t.created_at) day FROM telemetry t, users u WHERE t.user_id = u.id AND t.submission_date > '2019-07-01' GROUP BY 1, 2;
```

✅ **GOOD**:
```sql
select
    t.client_id as client_id,
    date(t.created_at) as day
from
    telemetry as t
inner join
    users as u
    on t.user_id = u.id
where
    t.submission_date > '2019-07-01'
    and t.sample_id = '10'
group by
    t.client_id,
    day;
```

### 1.3. Comments
Use block comments (`/* ... */`) for multi-line descriptions and line comments (`--`) for single-line notes.

```sql
/*
This query retrieves active users and their recent orders.
It joins the users table with the orders table.
*/
select
    u.user_id, -- Unique user identifier
    u.user_name,
    o.order_id,
    o.order_date
from
    users as u
inner join
    orders as o
    on u.user_id = o.user_id;
```

## 2. Common Patterns and Anti-patterns

### 2.1. Explicit JOINs
Always use explicit `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, or `FULL JOIN`. Never use implicit joins in the `FROM` clause.

❌ **BAD**:
```sql
select u.user_name, o.order_date from users u, orders o where u.user_id = o.user_id;
```

✅ **GOOD**:
```sql
select
    u.user_name,
    o.order_date
from
    users as u
inner join
    orders as o
    on u.user_id = o.user_id;
```

### 2.2. Common Table Expressions (CTEs)
Prefer CTEs (`WITH` clauses) over nested subqueries for improved readability and modularity.

❌ **BAD**:
```sql
select
    count(*)
from (
    select
        user_id
    from
        orders
    where
        order_date >= '2023-01-01'
    group by
        user_id
    having
        count(*) > 5
) as frequent_buyers;
```

✅ **GOOD**:
```sql
with frequent_buyers as (
    select
        user_id
    from
        orders
    where
        order_date >= '2023-01-01'
    group by
        user_id
    having
        count(*) > 5
)
select
    count(*)
from
    frequent_buyers;
```

### 2.3. Avoid `NOT IN`
Never use `NOT IN` due to its problematic behavior with `NULL` values. Prefer `NOT EXISTS` or `LEFT JOIN ... IS NULL`.

❌ **BAD**:
```sql
select user_name from users where user_id not in (select user_id from orders where status = 'cancelled');
```

✅ **GOOD` (NOT EXISTS)`**:
```sql
select
    user_name
from
    users as u
where not exists (
    select 1 from orders as o where o.user_id = u.user_id and o.status = 'cancelled'
);
```

✅ **GOOD` (LEFT JOIN ... IS NULL)`**:
```sql
select
    u.user_name
from
    users as u
left join
    orders as o
    on u.user_id = o.user_id and o.status = 'cancelled'
where
    o.user_id is null;
```

## 3. Performance Considerations

### 3.1. Explicit Column Selection
Always specify columns explicitly. Avoid `SELECT *` in production code. This improves performance, reduces network traffic, and prevents issues when schema changes.

❌ **BAD**:
```sql
select * from products;
```

✅ **GOOD**:
```sql
select
    product_id,
    product_name,
    price,
    stock_quantity
from
    products;
```

### 3.2. Query Optimization
Use `EXPLAIN ANALYZE` to understand and optimize query plans. Focus on reducing sequential scans and improving index usage.

```sql
explain analyze
select
    o.order_id,
    c.customer_name
from
    orders as o
inner join
    customers as c
    on o.customer_id = c.customer_id
where
    o.order_date >= '2023-01-01'
order by
    o.order_date desc
limit 100;
```

## 4. Common Pitfalls and Gotchas

### 4.1. Date/Time Storage
Store all timestamps as `timestamp with time zone` (`timestamptz`) and always in UTC. Convert to local time zones only at the application layer for display.

❌ **BAD**:
```sql
create table events (
    event_id int,
    event_time timestamp without time zone -- Prone to timezone issues
);
```

✅ **GOOD**:
```sql
create table events (
    event_id bigint generated always as identity primary key,
    event_time timestamptz default now() -- Always store in UTC
);
```

### 4.2. `BETWEEN` with Timestamps
Avoid `BETWEEN` for date/time ranges, especially when precision matters. It includes both start and end points, which can lead to off-by-one errors. Use explicit `>=` and `<` operators.

❌ **BAD**:
```sql
select * from orders where order_date between '2023-01-01' and '2023-01-31';
```

✅ **GOOD**:
```sql
select
    *
from
    orders
where
    order_date >= '2023-01-01T00:00:00Z'
    and order_date < '2023-02-01T00:00:00Z'; -- Correctly handles the entire month of January
```

### 4.3. Data Type Choices
*   **IDs**: Use `bigint generated always as identity` for primary keys. Never use `serial` or `bigserial`.
*   **Text**: Prefer `text` over `varchar(n)` unless there's a specific, strict length constraint. Never use `char(n)`.
*   **Money**: Never use the `money` type. Use `numeric` or `decimal` with explicit precision (e.g., `numeric(19, 4)`).

❌ **BAD**:
```sql
create table products (
    id serial primary key,
    product_code char(10),
    description varchar(255),
    price money
);
```

✅ **GOOD**:
```sql
create table products (
    product_id bigint generated always as identity primary key,
    product_code varchar(10) not null, -- Use varchar for fixed-length codes
    description text,
    price numeric(19, 4) not null -- Explicit precision for currency
);
```

### 4.4. Avoid Legacy Constructs
Never use PostgreSQL `rules` or `table inheritance`. Use `triggers` for event-driven logic and native `table partitioning` for large tables.

## 5. Security Best Practices

### 5.1. Prepared Statements
Always use prepared statements (e.g., via parameterized queries in your application code) to prevent SQL injection vulnerabilities.

❌ **BAD` (Application Code)`**:
```python
cursor.execute(f"select * from users where user_name = '{user_input}';")
```

✅ **GOOD` (Application Code)`**:
```python
cursor.execute("select * from users where user_name = %s;", (user_input,))
```

### 5.2. Least Privilege
Grant database roles only the minimum necessary permissions. Avoid granting `ALL PRIVILEGES`.

```sql
-- Create a read-only role
create role app_reader nologin;
grant connect on database my_app_db to app_reader;
grant usage on schema public to app_reader;
grant select on all tables in schema public to app_reader;
alter default privileges in schema public grant select on tables to app_reader;

-- Create a read-write role
create role app_writer nologin;
grant app_reader to app_writer; -- Inherit read permissions
grant insert, update, delete on all tables in schema public to app_writer;
alter default privileges in schema public grant insert, update, delete on tables to app_writer;
```

## 6. Data Modeling

### 6.1. Foreign Key Naming
Name foreign key columns as `<referenced_table>_id` (singular form of the referenced table name).

❌ **BAD**:
```sql
create table orders (
    order_id bigint generated always as identity primary key,
    customerid bigint not null -- Inconsistent naming
);
```

✅ **GOOD**:
```sql
create table orders (
    order_id bigint generated always as identity primary key,
    customer_id bigint not null, -- Correct foreign key naming
    constraint fk_customer foreign key (customer_id) references customers (customer_id)
);
```

### 6.2. Table Comments
Always add descriptive comments to tables and columns using `COMMENT ON`. This is crucial for documentation and understanding schema intent.

```sql
comment on table users is 'Stores information about application users.';
comment on column users.user_name is 'Full name of the user.';
```

## 7. Testing Approaches

### 7.1. Transactional Tests
Wrap database tests in transactions that are rolled back at the end. This ensures a clean state for each test run.

```sql
begin; -- Start transaction
    -- Insert test data
    insert into users (user_name) values ('Test User');

    -- Run assertions
    select count(*) from users where user_name = 'Test User'; -- Should be 1

rollback; -- Rollback all changes
```