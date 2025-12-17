---
description: Comprehensive guidelines for writing high-performance, secure, and maintainable Supabase applications, covering SQL style, data modeling, security, and testing.
globs: **/*
---
# Supabase Best Practices

This guide outlines the definitive best practices for developing with Supabase, ensuring your projects are secure, performant, and maintainable. Adhere to these rules for consistent, high-quality code.

## 1. SQL Style & Code Organization

Consistency is paramount. Follow these rules for all SQL schema, migrations, and functions.

### 1.1 Naming Conventions
- **Identifiers:** Use `snake_case` for all table, column, and function names. Names must begin with a letter and contain only letters, numbers, and underscores.
- **Keywords:** Always use lowercase for SQL keywords (e.g., `select`, `from`, `where`).
- **Table Names:** Prefer plural, `snake_case` names (e.g., `users`, `products`).
- **Column Names:** Prefer singular, `snake_case` names (e.g., `id`, `name`, `created_at`).
- **Foreign Keys:** Use `singular_table_name_id` (e.g., `user_id` for `users` table).
- **Triggers:** Prefix with a numeric order (e.g., `_200_update_timestamp`) as triggers run in lexicographical order.

❌ BAD
```sql
CREATE TABLE UserProfiles (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(255)
);
SELECT UserID FROM UserProfiles;
```

✅ GOOD
```sql
CREATE TABLE user_profiles (
    id bigint generated always as identity primary key,
    user_name text not null,
    user_id bigint references users (id) -- Example FK
);
SELECT id FROM user_profiles;

-- Trigger naming for ordered execution
CREATE TRIGGER _200_update_user_profile_timestamp
BEFORE UPDATE ON user_profiles
FOR EACH ROW EXECUTE FUNCTION update_timestamp();
```

### 1.2 Formatting & Readability
- **Indentation:** Use consistent indentation for clauses and arguments.
- **Aliases:** Always use explicit `AS` for aliases. Make them meaningful.
- **Boolean Operators:** Place `AND`/`OR` at the beginning of the line for multi-line `WHERE` clauses.
- **Parentheses:** Align closing parentheses with the starting line of the expression.

❌ BAD
```sql
SELECT t.client_id, DATE(t.created_at) day
FROM telemetry t, users u
WHERE t.user_id = u.id AND t.submission_date > '2019-07-01'
GROUP BY 1, 2;
```

✅ GOOD
```sql
SELECT
    t.client_id AS client_identifier,
    DATE(t.created_at) AS created_day
FROM
    telemetry AS t
INNER JOIN
    users AS u ON t.user_id = u.id
WHERE
    t.submission_date > '2019-07-01'
    AND t.sample_id = '10'
GROUP BY
    t.client_id,
    created_day;
```

### 1.3 Common Table Expressions (CTEs)
- **Prefer CTEs:** Use CTEs (`WITH` clauses) for complex queries to improve readability and modularity over nested subqueries.

❌ BAD
```sql
SELECT
    (SELECT COUNT(*) FROM orders WHERE user_id = u.id) AS order_count,
    u.email
FROM
    users AS u;
```

✅ GOOD
```sql
WITH user_order_counts AS (
    SELECT
        user_id,
        COUNT(*) AS order_count
    FROM
        orders
    GROUP BY
        user_id
)
SELECT
    u.email,
    uoc.order_count
FROM
    users AS u
LEFT JOIN
    user_order_counts AS uoc ON u.id = uoc.user_id;
```

## 2. Data Modeling & Schema Design

Design your schema for clarity, integrity, and performance.

### 2.1 Primary Keys
- **Standard PK:** Every table *must* have an `id` column of type `bigint generated always as identity primary key`.

❌ BAD
```sql
CREATE TABLE products (
    product_code text PRIMARY KEY,
    name text
);
```

✅ GOOD
```sql
CREATE TABLE products (
    id bigint generated always as identity primary key,
    product_code text unique not null,
    name text not null
);
```

### 2.2 Foreign Key Indexes
- **Always Index FKs:** PostgreSQL does *not* automatically index foreign keys. Always add an index to every foreign key column to prevent expensive lookups, especially for reverse relations.

❌ BAD
```sql
CREATE TABLE orders (
    id bigint generated always as identity primary key,
    user_id bigint references users (id)
);
-- Missing index on user_id
```

✅ GOOD
```sql
CREATE TABLE orders (
    id bigint generated always as identity primary key,
    user_id bigint references users (id)
);
CREATE INDEX ON orders (user_id);
```

### 2.3 Table Comments
- **Document Tables:** Always add a descriptive comment to each table using `COMMENT ON TABLE`.

❌ BAD
```sql
CREATE TABLE posts (
    id bigint generated always as identity primary key,
    title text
);
```

✅ GOOD
```sql
CREATE TABLE posts (
    id bigint generated always as identity primary key,
    title text not null,
    content text
);
COMMENT ON TABLE posts IS 'Blog posts published by users.';
```

## 3. Security Best Practices

Security is non-negotiable. Implement these measures from day one.

### 3.1 Row-Level Security (RLS)
- **Enable RLS Everywhere:** Enable RLS on *every* table in your database, especially those exposed via API. This is your primary defense.

❌ BAD
```sql
CREATE TABLE sensitive_data (
    id bigint generated always as identity primary key,
    secret text
);
-- RLS NOT ENABLED
```

✅ GOOD
```sql
CREATE TABLE sensitive_data (
    id bigint generated always as identity primary key,
    secret text,
    user_id uuid references auth.users(id)
);
ALTER TABLE sensitive_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY select_own_secrets ON sensitive_data FOR SELECT USING (auth.uid() = user_id);
```

### 3.2 Function Security
- **Default to `SECURITY INVOKER`:** Functions should run with the permissions of the calling user. Use `SECURITY DEFINER` only when absolutely necessary and with extreme caution.
- **Explicit `search_path`:** If using `SECURITY DEFINER`, *always* set `search_path = ''` and use fully-qualified names (e.g., `public.table_name`) to prevent privilege escalation via malicious objects in other schemas.

❌ BAD
```sql
CREATE FUNCTION get_all_users()
RETURNS SETOF users
LANGUAGE plpgsql
SECURITY DEFINER -- Dangerous without search_path
AS $$
BEGIN
    RETURN QUERY SELECT * FROM users;
END;
$$;
```

✅ GOOD
```sql
CREATE FUNCTION get_current_user_profile()
RETURNS SETOF user_profiles
LANGUAGE plpgsql
SECURITY INVOKER -- Default and safer
SET search_path = '' -- Always set, even for INVOKER for clarity
AS $$
BEGIN
    RETURN QUERY SELECT * FROM public.user_profiles WHERE user_id = auth.uid();
END;
$$;

-- If SECURITY DEFINER is truly needed (rare):
CREATE FUNCTION admin_only_function()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = '' -- CRITICAL for SECURITY DEFINER
AS $$
BEGIN
    -- Use fully qualified names for ALL objects
    INSERT INTO public.audit_log (action) VALUES ('Admin function executed');
END;
$$;
```

### 3.3 Grant Permissions
- **Table-level `SELECT/DELETE`:** Grant `SELECT` and `DELETE` permissions at the table level.
- **Column-level `INSERT/UPDATE`:** Grant `INSERT` and `UPDATE` permissions at the column level for explicitness and to avoid optimizer issues.
- **Avoid Column-level `SELECT`:** This severely limits optimizations and functionality (e.g., `SELECT *`, `RETURNING *`).
- **Avoid Table-level `INSERT/UPDATE`:** Lacks explicitness.

❌ BAD
```sql
GRANT SELECT (id, name) ON users TO authenticated; -- Avoid column-level SELECT
GRANT INSERT ON users TO authenticated; -- Avoid table-level INSERT
```

✅ GOOD
```sql
GRANT SELECT ON users TO authenticated;
GRANT DELETE ON users TO authenticated;
GRANT INSERT (name, email) ON users TO authenticated;
GRANT UPDATE (name, email) ON users TO authenticated;
```

## 4. Performance & Query Optimization

Optimize your database interactions for speed and efficiency.

### 4.1 Prefer `LANGUAGE SQL` for Functions
- **Inlining:** Use `LANGUAGE sql` over `LANGUAGE plpgsql` for functions whenever possible. SQL functions can often be inlined by the query planner, leading to significant performance gains.

❌ BAD
```sql
CREATE FUNCTION add_one(a int)
RETURNS int
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN a + 1;
END;
$$;
```

✅ GOOD
```sql
CREATE FUNCTION add_one(a int)
RETURNS int
LANGUAGE sql
IMMUTABLE -- Declare as IMMUTABLE if results depend only on inputs
AS $$
    SELECT a + 1;
$$;
```

### 4.2 Function Volatility
- **Immutable/Stable Defaults:** Declare functions as `IMMUTABLE` (returns same result for same inputs, no side effects) or `STABLE` (results change only within a single scan, no side effects) where appropriate. Default to `VOLATILE` only if the function modifies data or has external side effects. This allows PostgreSQL to optimize queries more effectively.

## 5. Testing & Linting

Ensure your database schema and functions are robust and error-free.

### 5.1 Database Testing
- **`supabase test db`:** Use the Supabase CLI (`supabase test db`) with `pgTAP` for comprehensive database unit and integration testing. Integrate this into your CI/CD pipeline.

### 5.2 Database Linting
- **`supabase db lint`:** Leverage `supabase db lint` (powered by `plpgsql_check`) to catch typing errors, unused variables, dead code, and potential SQL injection vulnerabilities early.

### 5.3 Type-Safe Query Builders
- **Kysely for Edge Functions:** For application code, especially in Edge Functions, use type-safe query builders like Kysely with `deno-postgres`. This provides compile-time guarantees and excellent autocompletion.

```typescript
// Example using Kysely in a Supabase Edge Function
import { Kysely, PostgresDialect } from 'kysely';
import { Pool } from 'https://deno.land/x/postgres@v0.17.0/mod.ts'; // Deno Postgres driver

interface Database {
  users: {
    id: number;
    email: string;
    created_at: Date;
  };
  posts: {
    id: number;
    user_id: number;
    title: string;
  };
}

const db = new Kysely<Database>({
  dialect: new PostgresDialect({
    pool: new Pool({
      database: 'postgres',
      hostname: Deno.env.get('DB_HOSTNAME'),
      user: 'postgres',
      password: Deno.env.get('DB_PASSWORD'),
      port: 5432,
      tls: { caCertificates: [Deno.env.get('DB_SSL_CERT')!] },
    }, 1), // Single connection pool for Edge Functions
  }),
});

// Type-safe query example
const user = await db.selectFrom('users')
  .where('id', '=', 1)
  .select(['id', 'email'])
  .executeTakeFirst();

console.log(user?.email); // Autocompletion and type safety
```