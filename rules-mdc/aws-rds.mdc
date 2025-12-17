---
description: Definitive guidelines for secure, performant, and maintainable database development and operations with AWS RDS, emphasizing modern best practices and common anti-patterns.
globs: **/*
---
# aws-rds Best Practices

This guide outlines the definitive best practices for interacting with and managing AWS RDS instances. Adhere to these guidelines to ensure your database operations are secure, performant, and maintainable.

## 1. Security Best Practices

Security is paramount. Always assume your database is a target.

### 1.1. Manage Credentials with AWS Secrets Manager

Never hardcode database credentials or store them in environment variables directly. Use AWS Secrets Manager with automatic rotation enabled.

❌ BAD:
```python
# app.py
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST)
```

✅ GOOD:
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# app.py
secret = get_secret("your-rds-db-credentials")
DB_USER = secret['username']
DB_PASS = secret['password']
DB_HOST = secret['host']
conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST)
```
**Context**: Secrets Manager handles rotation, encryption, and secure retrieval, significantly reducing the risk of credential compromise.

### 1.2. Enforce Least Privilege with IAM

Control access to RDS API actions (create, modify, delete clusters, security groups, parameter groups) using IAM identities, not the root account. Grant only the minimum permissions required. Organize permissions with IAM groups.

❌ BAD:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "rds:*",
            "Resource": "*"
        }
    ]
}
```
**Context**: This grants full access to all RDS resources, a massive security risk.

✅ GOOD:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "rds:DescribeDBInstances",
                "rds:DescribeDBSnapshots",
                "rds:Connect"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "rds:ModifyDBInstance",
                "rds:DeleteDBInstance"
            ],
            "Resource": "arn:aws:rds:REGION:ACCOUNT_ID:db:my-specific-app-db"
        }
    ]
}
```
**Context**: This policy grants read-only access to all RDS resources and specific modification/deletion rights to a single, named DB instance.

### 1.3. Use Security Groups and VPC Endpoints

Restrict network access to your RDS instances using tightly scoped security groups. For private access from within your VPC, use VPC Endpoints (AWS PrivateLink).

❌ BAD:
```json
{
  "IpProtocol": "tcp",
  "FromPort": 5432,
  "ToPort": 5432,
  "IpRanges": [{"CidrIp": "0.0.0.0/0"}] // Allows access from anywhere
}
```

✅ GOOD:
```json
{
  "IpProtocol": "tcp",
  "FromPort": 5432,
  "ToPort": 5432,
  "IpRanges": [{"CidrIp": "10.0.0.0/16"}] // Allows access only from your VPC CIDR
}
```
**Context**: Only allow ingress from specific application servers, bastion hosts, or other trusted AWS services.

## 2. Code Organization and Structure

Maintain clean, readable, and version-controlled SQL.

### 2.1. Adopt Consistent Naming Conventions

Use `snake_case` for all database identifiers (tables, columns, indexes). Use `PascalCase` for stored procedures and functions.

❌ BAD:
```sql
CREATE TABLE User_Accounts (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(255),
    EmailAddress VARCHAR(255)
);

-- Stored procedure
create procedure get_user_data(IN p_user_id INT)
BEGIN
    SELECT * FROM User_Accounts WHERE UserID = p_user_id;
END;
```

✅ GOOD:
```sql
CREATE TABLE user_accounts (
    user_id INT PRIMARY KEY,
    user_name VARCHAR(255),
    email_address VARCHAR(255)
);

-- Stored procedure
CREATE PROCEDURE GetUserData(IN p_user_id INT)
BEGIN
    SELECT user_id, user_name, email_address FROM user_accounts WHERE user_id = p_user_id;
END;
```
**Context**: Consistency improves readability and reduces cognitive load for developers.

### 2.2. Version Control All SQL

Treat your SQL schema and stored procedures as application code. Store them in your version control system (e.g., Git) and integrate them into your CI/CD pipeline.

**Context**: Enables collaboration, change tracking, and automated deployment.

## 3. Common Patterns and Anti-patterns

Follow established patterns to build robust and secure applications.

### 3.1. Prevent SQL Injection with Parameterized Queries

Always use parameterized queries or prepared statements. Never concatenate user input directly into SQL strings.

❌ BAD:
```python
# Python with psycopg2
user_input = "'; DROP TABLE users; --"
query = f"SELECT * FROM users WHERE username = '{user_input}'"
cursor.execute(query)
```

✅ GOOD:
```python
# Python with psycopg2
user_input = "malicious_user"
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (user_input,))
```
**Context**: This is the single most important security practice for SQL.

### 3.2. Avoid `SELECT *`

Explicitly list columns in `SELECT` statements. This improves performance, reduces network traffic, and makes schema changes safer.

❌ BAD:
```sql
SELECT * FROM orders WHERE customer_id = 123;
```

✅ GOOD:
```sql
SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE customer_id = 123;
```
**Context**: Prevents fetching unnecessary data, especially when tables grow wide.

### 3.3. Utilize RDS Proxy for Connection Management

For serverless applications (e.g., Lambda) or high-concurrency scenarios, use Amazon RDS Proxy to pool connections, manage credentials, and handle automatic failovers.

❌ BAD:
```python
# Lambda function establishing a new DB connection on every invocation
conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST)
# ... query ...
conn.close()
```
**Context**: High overhead, potential for connection storms, and slow cold starts.

✅ GOOD:
```python
# Lambda function connecting via RDS Proxy
conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=RDS_PROXY_ENDPOINT)
# ... query ...
# Connection is returned to the pool, not closed
```
**Context**: RDS Proxy maintains a pool of connections, reducing overhead and improving performance for transient workloads.

### 3.4. Implement Blue/Green Deployments for Schema Changes

For zero-downtime schema changes, use RDS Blue/Green Deployments. Ensure schema modifications are replication-compatible (e.g., adding columns at the end of a table).

❌ BAD:
```sql
-- Directly alter production table
ALTER TABLE users RENAME COLUMN email TO email_address;
```
**Context**: This breaks replication and causes downtime during switchover.

✅ GOOD:
```sql
-- On Green environment, ensure compatibility
ALTER TABLE users ADD COLUMN email_address VARCHAR(255);
-- Backfill data if necessary
UPDATE users SET email_address = email;
-- Update application to use new column
-- Perform switchover
-- On new Blue (old Green), drop old column
ALTER TABLE users DROP COLUMN email;
```
**Context**: Allows for a safe, tested, and reversible deployment process.

### 3.5. Migrate from Aurora Serverless v1 to v2

Aurora Serverless v1 reaches end-of-life on March 31, 2025. Migrate to Aurora Serverless v2 or provisioned Aurora for instant auto-scaling and lower latency.

**Context**: v2 offers superior performance and scalability characteristics. Plan and test this migration using a blue/green strategy.

## 4. Performance Considerations

Optimize your queries and database configuration for speed and efficiency.

### 4.1. Strategic Indexing

Create indexes that match your common query patterns (e.g., `WHERE` clauses, `JOIN` conditions, `ORDER BY` clauses). Regularly review query performance using `EXPLAIN` plans.

❌ BAD:
```sql
-- Query frequently filters by customer_name but no index exists
SELECT * FROM customers WHERE customer_name LIKE 'John%';
```

✅ GOOD:
```sql
CREATE INDEX idx_customer_name ON customers (customer_name);
SELECT customer_id, customer_name, email FROM customers WHERE customer_name LIKE 'John%';
```
**Context**: Indexes are critical for fast data retrieval, but over-indexing can hurt write performance.

### 4.2. Monitor and Tune Parameters

Use Amazon CloudWatch and RDS Performance Insights to monitor CPU, memory, I/O, and active transactions. Adjust DB parameter groups based on workload analysis.

**Context**: Proactive monitoring helps identify bottlenecks before they impact users.

## 5. Query Optimization

Write efficient SQL queries to minimize resource consumption.

### 5.1. Use `EXPLAIN` to Analyze Queries

Always use `EXPLAIN` (or `EXPLAIN ANALYZE`) to understand how your database executes a query and identify potential bottlenecks.

```sql
EXPLAIN ANALYZE SELECT order_id, total_amount FROM orders WHERE order_date >= '2023-01-01';
```
**Context**: Provides insights into index usage, join order, and row counts.

### 5.2. Avoid N+1 Query Problems

Fetch related data in a single query using `JOIN`s or batching, rather than making multiple individual queries.

❌ BAD:
```python
# Fetch users
users = cursor.execute("SELECT user_id, user_name FROM users").fetchall()
for user in users:
    # For each user, fetch their orders
    orders = cursor.execute(f"SELECT * FROM orders WHERE user_id = {user['user_id']}").fetchall()
    user['orders'] = orders
```

✅ GOOD:
```python
# Fetch users and their orders in a single query
query = """
SELECT u.user_id, u.user_name, o.order_id, o.total_amount
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id;
"""
cursor.execute(query)
# Process results in application to structure data
```
**Context**: Reduces round trips to the database, significantly improving performance.

## 6. Data Modeling

Design your schema for integrity, performance, and scalability.

### 6.1. Choose Appropriate Data Types

Select the smallest, most appropriate data type for each column to conserve storage and improve performance.

❌ BAD:
```sql
CREATE TABLE products (
    product_id VARCHAR(255) PRIMARY KEY, -- UUIDs are better as BINARY(16)
    price DECIMAL(38, 18) -- Too much precision for typical currency
);
```

✅ GOOD:
```sql
CREATE TABLE products (
    product_id BINARY(16) PRIMARY KEY, -- For UUIDs
    price DECIMAL(10, 2) -- Standard currency precision
);
```
**Context**: Efficient data types reduce disk I/O and memory usage.

### 6.2. Enforce Referential Integrity

Use foreign keys to maintain relationships between tables and ensure data consistency.

❌ BAD:
```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT, -- No foreign key constraint
    order_date DATE
);
```

✅ GOOD:
```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE,
    CONSTRAINT fk_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id)
        ON DELETE CASCADE -- Or SET NULL, RESTRICT, NO ACTION
);
```
**Context**: Prevents orphaned records and ensures data integrity.

## 7. Testing Approaches

Integrate database testing into your development workflow.

### 7.1. Thoroughly Test Blue/Green Deployments

Before any production switchover, thoroughly test the green environment. Keep it read-only until cut-over to prevent unintended data writes.

**Context**: Ensures the new environment functions correctly and schema changes are compatible.

### 7.2. Unit and Integration Testing for SQL

Write automated tests for your SQL queries, stored procedures, and schema migrations. Use a dedicated test database that mirrors production.

**Context**: Catches bugs early and ensures changes don't break existing functionality.