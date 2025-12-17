---
description: This guide defines the definitive best practices for writing Cypher queries and modeling data in Neo4j, ensuring readability, performance, and security for all team projects.
globs: **/*
---
# neo4j Best Practices

This document outlines the mandatory guidelines for developing with Neo4j. Adhering to these rules ensures our Cypher queries are performant, secure, and maintainable.

## 1. Code Organization and Structure

### 1.1. Standardize Naming Conventions

Consistency is paramount. Use the recommended casing for all identifiers to prevent subtle bugs due to Cypher's case-sensitivity.

*   **Node Labels:** `PascalCase`
*   **Relationship Types:** `UPPER_SNAKE_CASE`
*   **Property Keys:** `camelCase`
*   **Variables:** `camelCase`

**❌ BAD:**
```cypher
MATCH (p:person)-[r:friend_of]->(f:Friend)
WHERE p.first_name = 'Alice'
RETURN f.last_name
```

**✅ GOOD:**
```cypher
MATCH (p:Person)-[r:FRIEND_OF]->(f:Person)
WHERE p.firstName = 'Alice'
RETURN f.lastName
```

### 1.2. Escape Special Characters Judiciously

Only use backticks (`` ` ``) when an identifier *must* contain special characters, spaces, or start with a non-alphabetic character. Avoid them otherwise to keep queries clean.

**❌ BAD:**
```cypher
MATCH (`my node`:`User Label`)
WHERE `my node`.`user-id` = 123
RETURN `my node`.`user-name`
```

**✅ GOOD:**
```cypher
MATCH (u:User)
WHERE u.userId = 123
RETURN u.userName
```
*If a special character is truly unavoidable:*
```cypher
MATCH (n:`1stUser`)
WHERE n.`user-id` = 123
RETURN n.userName
```

### 1.3. Always Use Cypher 25

New features and performance improvements are exclusively added to Cypher 25. Avoid older versions like 5.x.

## 2. Data Modeling

### 2.1. Design for Query-Ability

Model your graph around the most common traversals and business questions. Prioritize relationships that directly answer real-world queries over mimicking relational schemas.

**❌ BAD:** (Relational thinking, too many properties on nodes, generic relationships)
```cypher
// Modeling a "User" and their "Address" as separate nodes, but linking them generically
CREATE (u:User {id: 'u1', name: 'Alice'})
CREATE (a:Address {id: 'a1', street: '123 Main St', city: 'Anytown'})
CREATE (u)-[:HAS]->(a) // Generic relationship
```

**✅ GOOD:** (Graph-native, explicit relationship type, Address properties on User if 1:1)
```cypher
// If Address is always tied to a User and not shared, keep properties on the User node.
// If Address can be shared or has complex relationships, make it a node with a specific relationship.
CREATE (u:User {userId: 'u1', name: 'Alice', street: '123 Main St', city: 'Anytown'})

// If Address is a separate entity (e.g., shared by multiple users, or has its own complex relationships)
CREATE (u:User {userId: 'u1', name: 'Alice'})
CREATE (a:Address {addressId: 'a1', street: '123 Main St', city: 'Anytown'})
CREATE (u)-[:LIVES_AT]->(a) // Specific relationship type
```

### 2.2. Leverage Constraints for Data Integrity

Enforce uniqueness, existence, and type constraints on key properties. This ensures data quality and automatically creates backing indexes for performance.

**❌ BAD:** (No constraints, allowing duplicate or missing critical data)
```cypher
CREATE (p:Person {email: 'alice@example.com'})
CREATE (p:Person {email: 'alice@example.com'}) // Duplicate allowed
```

**✅ GOOD:** (Unique constraint on email, existence constraint on name)
```cypher
// Run these DDL statements once during schema setup
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE;
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL;
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS OF TYPE INTEGER; // Cypher 25 type constraint

// Now, duplicate email will fail, and missing name will fail
CREATE (p:Person {email: 'alice@example.com', name: 'Alice', age: 30})
```

## 3. Common Patterns and Anti-patterns

### 3.1. Always Use Parameters for Dynamic Values

Parameterized queries prevent Cypher injection vulnerabilities and allow the query planner to cache execution plans, significantly improving performance.

**❌ BAD:** (String concatenation, injection risk, no plan caching)
```cypher
// In application code:
const userId = "123 OR 1=1"; // Malicious input
const query = `MATCH (u:User {userId: '${userId}'}) RETURN u.name`;
// Result: MATCH (u:User {userId: '123 OR 1=1'}) RETURN u.name
```

**✅ GOOD:** (Parameterized query, safe, plan caching)
```cypher
// In application code:
const userId = "123 OR 1=1"; // Malicious input
const query = `MATCH (u:User {userId: $userId}) RETURN u.name`;
const params = { userId: userId };
// Result: MATCH (u:User {userId: '123 OR 1=1'}) RETURN u.name (as a literal string value for $userId)
```

### 3.2. Leverage Native Map Projections

Use map projections (`{ .property }`) to return specific properties or computed values efficiently, especially with Cypher 25's enhancements.

**❌ BAD:** (Returning entire nodes/relationships, then filtering in application code)
```cypher
MATCH (u:User {userId: $userId})
RETURN u
// Application code then extracts u.name, u.email
```

**✅ GOOD:** (Returning only necessary properties)
```cypher
MATCH (u:User {userId: $userId})
RETURN { name: u.name, email: u.email, age: u.age } AS userDetails
// Or, for all properties: RETURN u { .* }
```

## 4. Performance Considerations

### 4.1. Create Indexes on Frequently Filtered Properties

Indexes drastically speed up `MATCH` and `WHERE` clauses on specific properties. Always create them for properties used in lookups, filtering, or ordering. Constraints often create indexes automatically.

**❌ BAD:** (Scanning all nodes for a property without an index)
```cypher
MATCH (p:Product)
WHERE p.sku = 'XYZ-789' // No index on Product.sku
RETURN p.name
```

**✅ GOOD:** (Index on `Product.sku`)
```cypher
// Run this DDL statement once during schema setup
CREATE INDEX FOR (p:Product) ON (p.sku);

// Now, this query uses the index for fast lookup
MATCH (p:Product)
WHERE p.sku = 'XYZ-789'
RETURN p.name
```

### 4.2. Profile and Explain Hot Queries

Always use `PROFILE` or `EXPLAIN` to understand query execution plans. This is the only way to identify bottlenecks and ensure indexes are being used correctly.

**✅ GOOD:**
```cypher
PROFILE
MATCH (u:User)-[:PURCHASED]->(p:Product)
WHERE u.country = 'USA' AND p.category = 'Electronics'
RETURN u.name, p.name
```
Analyze the output to see operations, costs, and if indexes are utilized.

## 5. Common Pitfalls and Gotchas

### 5.1. Case-Sensitivity of Identifiers

Remember that Cypher identifiers are case-sensitive. `Person` is different from `person`. Adhere to naming conventions to avoid this.

**❌ BAD:**
```cypher
MATCH (p:person) // Label is 'Person' in schema
RETURN p.name
```

**✅ GOOD:**
```cypher
MATCH (p:Person) // Correct label casing
RETURN p.name
```

### 5.2. Variable Re-use Within Scope

Node and relationship variables must be unique within the same query scope.

**❌ BAD:**
```cypher
MATCH (a)-[a]->(b) // 'a' used for both node and relationship
RETURN a, b
```

**✅ GOOD:**
```cypher
MATCH (a)-[r]->(b) // Unique variables 'a' and 'r'
RETURN a, b
```

## 6. Security Best Practices

### 6.1. Parameterized Queries (Reiterated)

This is the single most important security measure for Cypher. Never concatenate user input directly into queries.

### 6.2. Least Privilege Principle

Grant database users only the minimum necessary permissions. Avoid using a single, highly privileged user for all application interactions.

## 7. Query Optimization

### 7.1. Start `MATCH` Patterns with Indexed Nodes

When possible, begin your `MATCH` clause with a node that has a label and an indexed property used in the `WHERE` clause. This allows the query planner to efficiently find a starting point.

**❌ BAD:** (Starting with a generic pattern, then filtering)
```cypher
MATCH (n)-[r]->(m)
WHERE n.id = $id AND m.type = $type
RETURN n, m
```

**✅ GOOD:** (Starting with a specific, indexed node)
```cypher
MATCH (n:User {id: $id})-[:OWNS]->(m:Product {type: $type})
RETURN n, m
```

### 7.2. Limit Results Early

If you only need a subset of results, apply `LIMIT` as early as possible in your query to reduce the amount of data processed and transferred.

**❌ BAD:** (Returning many results, then limiting in application)
```cypher
MATCH (n:User)-[:FOLLOWS]->(f:User)
RETURN n, f
// Application takes first 10
```

**✅ GOOD:** (Limiting in Cypher)
```cypher
MATCH (n:User)-[:FOLLOWS]->(f:User)
RETURN n, f
LIMIT 10
```

## 8. Testing Approaches

### 8.1. Unit Test Cypher Query Construction

For complex queries built dynamically, write unit tests to ensure the generated Cypher strings are correct and adhere to naming conventions and parameterization.

### 8.2. Integration Tests Against a Test Database

Execute your application's Cypher queries against a dedicated, isolated test Neo4j instance. This validates query correctness, data integrity, and performance under realistic conditions. Use tools like Testcontainers for ephemeral Neo4j instances.