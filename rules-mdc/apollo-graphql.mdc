---
description: Definitive guidelines for building robust, performant, and maintainable apollo-graphql applications using modern best practices and tooling.
globs: **/*.{js,jsx}
---
# apollo-graphql Best Practices

This guide outlines our definitive best practices for developing with Apollo GraphQL, covering both server-side (Apollo Server 5) and client-side (Apollo Client) implementations. Adhering to these rules ensures consistency, performance, and maintainability across our GraphQL ecosystem.

## 1. Schema Design and Organization

Your GraphQL schema is the contract. Design it meticulously, document it thoroughly, and organize it logically.

### 1.1 Schema-First Development with SDL

Always start by defining your schema using Schema Definition Language (SDL). This forces clear API design before implementation.

❌ BAD
```javascript
// Implicit schema definition through resolvers
const resolvers = {
  Query: {
    users: () => [{ id: '1', name: 'Alice' }],
  },
};
// Schema is inferred or built programmatically, harder to review
```

✅ GOOD
```graphql
# schema/user.graphql
"""Represents a user in the system."""
type User {
  id: ID!
  name: String!
  email: String
}

type Query {
  """Fetches a list of all users."""
  users: [User!]!
  """Fetches a single user by their ID."""
  user(id: ID