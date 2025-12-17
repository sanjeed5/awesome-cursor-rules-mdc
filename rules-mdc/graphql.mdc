---
description: Enforce modern, performant, and secure GraphQL schema design and operation best practices for maintainable and scalable APIs.
globs: **/*.{graphql,gql}
---
# GraphQL Best Practices

This guide outlines the definitive best practices for designing, implementing, and operating GraphQL APIs. Adhere to these rules to ensure your schemas are consistent, performant, secure, and developer-friendly.

## 1. Schema Design & Naming Conventions

Consistency is paramount. Follow standard GraphQL naming conventions strictly.

### 1.1 Casing Rules

*   **Types (Object, Input, Enum, Interface, Union, Scalar)**: `PascalCase`
*   **Fields, Arguments, Directives**: `camelCase`
*   **Enum Values**: `SCREAMING_SNAKE_CASE`

❌ **BAD**
```graphql
type user { # Type name not PascalCase
  firstName: String!
  getProducts(productId: ID): [product] # Field name has verb, arg not camelCase
}

enum UserRole { # Enum value not SCREAMING_SNAKE_CASE
  admin
  guest
}
```

✅ **GOOD**
```graphql
type User {
  id: ID!
  firstName: String!
  products(limit: Int): [Product!]! # No verb, arg is camelCase
  role: UserRole!
}

enum UserRole {
  ADMIN
  GUEST
}
```

### 1.2 Query Field Naming

Avoid verb prefixes (`get`, `list`, `fetch`) on query (read) fields. This maintains consistency with nested fields.

❌ **BAD**
```graphql
type Query {
  getUsers: [User!]!
  listProducts: [Product!]!
}

query MyQuery {
  getUsers {
    id
    getProducts { # Inconsistent with root field
      name
    }
  }
}
```

✅ **GOOD**
```graphql
type Query {
  users: [User!]!
  products: [Product!]!
}

query MyQuery {
  users {
    id
    products { # Consistent with root field
      name
    }
  }
}
```

### 1.3 Mutation Field Naming

Start mutation fields with an imperative verb (e.g., `create`, `update`, `delete`).

❌ **BAD**
```graphql
type Mutation {
  userCreate(input: CreateUserInput!): CreateUserPayload!
}
```

✅ **GOOD**
```graphql
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(input: DeleteUserInput!): DeleteUserPayload!
}
```

### 1.4 Input & Payload Types

*   **Input Types**: Suffix with `Input`.
*   **Mutation Payload Types**: Suffix with `Payload` or `Response`. These should typically include the created/updated object and a `success` boolean or `errors` array.

❌ **BAD**
```graphql
input UserData { # Generic name
  name: String!
}

type Mutation {
  createUser(data: UserData!): User! # Returns raw object, no success/errors
}
```

✅ **GOOD**
```graphql
input CreateUserInput {
  name: String!
  email: String!
}

type CreateUserPayload {
  success: Boolean!
  user: User
  errors: [Error!] # Custom error type for detailed feedback
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

## 2. Global Object Identification & Pagination

Implement Relay-style global object identification and cursor-based pagination for robust caching and efficient data fetching.

### 2.1 Global Object ID

Every object that can be refetched or cached independently MUST implement the `Node` interface and expose a globally unique `id: ID!`.

```graphql
interface Node {
  id: ID!
}

type User implements Node {
  id: ID! # Global ID
  username: String!
  email: String!
}

type Query {
  node(id: ID!): Node
  user(id: ID!): User # Specific fetch for convenience
}
```

### 2.2 Cursor-Based Pagination

Always use cursor-based pagination (Relay Connection Specification) for lists that can grow large. Avoid offset-based pagination.

❌ **BAD**
```graphql
type Query {
  users(offset: Int = 0, limit: Int = 10): [User!]! # Offset-based pagination
}
```

✅ **GOOD**
```graphql
type Query {
  users(first: Int, after: String, last: Int, before: String): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

## 3. Nullability & Input Validation

Be explicit about nullability and always validate inputs server-side.

### 3.1 Explicit Nullability

Use `!` for non-nullable fields when the data is always expected. Default to nullable if the field might legitimately be null.

❌ **BAD**
```graphql
type User {
  name: String # Is name always present?
  email: String!
}
```

✅ **GOOD**
```graphql
type User {
  id: ID!
  name: String! # Name is always required
  email: String # Email might be null if user hasn't provided it
}
```

### 3.2 Server-Side Input Validation

GraphQL's type system provides basic validation, but complex business logic validation MUST occur in resolvers. Return specific errors in the `errors` array.

```graphql
input CreateUserInput {
  name: String! # GraphQL ensures it's a string and non-null
  email: String!
}

type CreateUserPayload {
  success: Boolean!
  user: User
  errors: [Error!] # Custom error type for detailed feedback
}

type Error {
  message: String!
  code: String! # e.g., "INVALID_EMAIL", "NAME_TOO_SHORT"
  path: [String!] # Field path where error occurred
}
```

## 4. Performance Considerations

Prevent N+1 query problems and optimize data fetching.

### 4.1 DataLoader for N+1 Problems

Always use `DataLoader` (or equivalent batching/caching mechanism) within your resolvers to prevent N+1 query issues when fetching related data. This is a server-side implementation detail but critical for GraphQL API performance.

❌ **BAD** (Conceptual - direct resolver calls)
```javascript
// In a resolver for User.products
user.productIds.map(productId => db.products.findById(productId)); // N+1 query
```

✅ **GOOD** (Conceptual - using DataLoader)
```javascript
// In a resolver for User.products
dataLoaders.productLoader.loadMany(user.productIds); // Batches and caches
```

## 5. Security

Implement robust security measures to protect your API from common attack vectors.

### 5.1 Query Depth and Complexity Limiting

Configure your GraphQL server to limit the maximum depth and complexity of incoming queries. This prevents malicious or accidental denial-of-service (DoS) attacks.

```graphql
# This is a server-side configuration, not schema.
# Example: Max depth 10, max complexity 1000.
# If exceeded, the server rejects the query before execution.
```

### 5.2 Rate Limiting

Implement rate limiting at the HTTP layer to prevent abuse and DoS attacks. This should be applied to all API endpoints, including your GraphQL endpoint.

## 6. Error Handling

Provide consistent and informative error responses.

### 6.1 Standardized Error Responses

Return errors in the `errors` array of the GraphQL response. Use the `extensions` field for custom error codes, details, and stack traces (in development environments only).

❌ **BAD**
```json
{
  "data": null,
  "httpStatus": 400,
  "customError": {
    "code": "INVALID_INPUT",
    "details": "Email is malformed"
  }
}
```

✅ **GOOD**
```json
{
  "data": null,
  "errors": [
    {
      "message": "Invalid input for field 'email'",
      "locations": [{ "line": 2, "column": 3 }],
      "path": ["createUser", "input", "email"],
      "extensions": {
        "code": "BAD_USER_INPUT",
        "validationErrors": [
          {
            "field": "email",
            "message": "Email address is not valid."
          }
        ]
      }
    }
  ]
}
```

## 7. Request/Response Patterns (HTTP)

Adhere to the "GraphQL Over HTTP" specification.

### 7.1 HTTP Method and Content-Type

Always use `POST` requests for GraphQL queries and mutations. The `Content-Type` header MUST be `application/json`.

❌ **BAD**
```
GET /graphql?query={users{id}}
```

✅ **GOOD**
```
POST /graphql
Content-Type: application/json

{
  "query": "query { users { id name } }",
  "variables": {},
  "operationName": null
}
```

### 7.2 HTTP Status Codes

A GraphQL server MUST respond with `200 OK` for all well-formed GraphQL requests, even if the GraphQL response contains errors in the `errors` array. Only use `4xx` or `5xx` status codes for HTTP-level errors (e.g., malformed request, authentication failure *before* GraphQL execution, server unavailable).

❌ **BAD**
```json
// HTTP Status: 400 Bad Request
{
  "errors": [{ "message": "User not found" }]
}
```

✅ **GOOD**
```json
// HTTP Status: 200 OK
{
  "data": { "user": null },
  "errors": [{ "message": "User not found" }]
}
```

## 8. Testing Approaches

Implement a comprehensive testing strategy covering schema, resolvers, and operations.

### 8.1 Schema Stability

Use snapshot testing for your GraphQL schema to catch unintended or breaking changes early in the development cycle. Integrate schema diff tools into your CI/CD pipeline.

```javascript
// Example using Jest and jest-serializer-graphql-schema
import { buildSchema } from 'graphql';
import { lexicographicSortSchema } from 'graphql/utilities';

const schema = buildSchema(`
  type User {
    id: ID!
    name: String!
  }
  type Query {
    users: [User!]!
  }
`);

test('schema should not change unexpectedly', () => {
  expect(lexicographicSortSchema(schema)).toMatchSnapshot();
});
```

### 8.2 Resolver Unit Tests

Unit test individual resolvers, especially those containing complex business logic, authorization checks, or data transformations. Mock external dependencies.

### 8.3 Operation Integration Tests

Write integration tests that execute full GraphQL operations (queries, mutations) against your server. These tests validate the end-to-end flow, including resolver composition and data fetching.

```javascript
// Example using Apollo Server's test utility
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

const typeDefs = `...`; // Your schema
const resolvers = { ... }; // Your resolvers

const server = new ApolloServer({ typeDefs, resolvers });

test('fetches users correctly', async () => {
  const { url } = await startStandaloneServer(server, { listen: { port: 0 } });
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `query { users { id name } }`,
    }),
  });
  const { data } = await response.json();
  expect(data.users).toEqual(expect.arrayContaining([
    { id: '1', name: 'Alice' },
  ]));
});
```

### 8.4 Schema Linting

Use schema linting tools (e.g., `graphql-schema-linter`, `@graphql-eslint/eslint-plugin`) in your CI/CD to enforce naming conventions, descriptions, and deprecation patterns.