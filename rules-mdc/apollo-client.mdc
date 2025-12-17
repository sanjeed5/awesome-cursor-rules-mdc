---
description: This guide defines the definitive best practices for using Apollo Client 3.x in our React applications, focusing on cache hygiene, type safety, performance, and maintainability.
globs: **/*.{js,jsx}
---
# apollo-client Best Practices

Apollo Client is our standard for GraphQL state management. Adhering to these guidelines ensures predictable data flow, optimal performance, and a scalable codebase.

---

## 1. Client Setup & Initialization

**Always instantiate `ApolloClient` as a singleton.** Avoid creating multiple client instances, as each client manages its own `InMemoryCache`, leading to inconsistent state and wasted resources.

❌ **BAD:** Multiple `ApolloClient` instances

```javascript
// In multiple files
const client = new ApolloClient({ /* ... */ });
```

✅ **GOOD:** Singleton `ApolloClient`

```javascript
// src/apolloClient.js
import { ApolloClient, InMemoryCache, HttpLink, from } from '@apollo/client';
import { onError } from '@apollo/client/link/error';
import { RetryLink } from '@apollo/client/link/retry';

const httpLink = new HttpLink({
  uri: process.env.GRAPHQL_ENDPOINT || 'http://localhost:4000/graphql',
  headers: {
    authorization: localStorage.getItem('token') || '',
    'client-name': 'OurApp [web]',
    'client-version': '1.0.0',
  },
});

const errorLink = onError(({ graphQLErrors, networkError }) => {
  if (graphQLErrors)
    graphQLErrors.forEach(({ message, locations, path }) =>
      console.error(`[GraphQL error]: Message: ${message}, Location: ${locations}, Path: ${path}`)
    );
  if (networkError) console.error(`[Network error]: ${networkError}`);
});

const retryLink = new RetryLink({
  delay: {
    initial: 300,
    max: Infinity,
    jitter: true
  },
  attempts: {
    max: 5,
    retryIf: (error, _operation) => !!error
  }
});

export const client = new ApolloClient({
  link: from([errorLink, retryLink, httpLink]),
  cache: new InMemoryCache({
    typePolicies: {
      // Define type policies here for custom cache behavior
      User: {
        keyFields: ['id'], // Ensure consistent identification
      },
      Query: {
        fields: {
          // Example: Custom merge for a paginated list
          users: {
            keyArgs: false, // Don't include arguments in cache key
            merge(existing = [], incoming) {
              return [...existing, ...incoming];
            },
          },
        },
      },
    },
  }),
  // Enable Automatic Persisted Queries for performance
  persistedQueries: {
    ttl: 3600, // 1 hour
  },
});

// src/App.js
import { ApolloProvider } from '@apollo/client';
import { client } from './apolloClient';

function App() {
  return (
    <ApolloProvider client={client}>
      {/* Your application */}
    </ApolloProvider>
  );
}
```

## 2. Data Fetching with React Hooks

**Always use Apollo Client's React hooks (`useQuery`, `useMutation`, `useSubscription`) for data operations within components.** These hooks provide declarative data fetching, automatic UI updates, and robust state management. Avoid direct `client.query()` or `client.mutate()` calls in components unless for specific server-side rendering or pre-fetching scenarios outside the React lifecycle.

❌ **BAD:** Imperative data fetching in components

```javascript
import { client } from '../apolloClient';

function MyComponent() {
  const [data, setData] = useState(null);
  useEffect(() => {
    client.query({ query: GET_USERS }).then(res => setData(res.data));
  }, []);
  // ...
}
```

✅ **GOOD:** Declarative data fetching with `useQuery`

```javascript
import { useQuery, gql } from '@apollo/client';

const GET_USERS = gql`
  query GetUsers($limit: Int!) {
    users(limit: $limit) {
      id
      name
    }
  }
`;

function UserList({ limit }) {
  const { loading, error, data } = useQuery(GET_USERS, {
    variables: { limit },
    fetchPolicy: 'cache-and-network', // Recommended for most lists
  });

  if (loading) return <p>Loading users...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <ul>
      {data.users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## 3. Cache Hygiene & Management

**Prioritize immutable cache updates and configure `InMemoryCache` with `TypePolicy` and `FieldPolicy`.** This prevents stale data, ensures garbage collection, and customizes merging logic for complex data structures.

**`fetchPolicy` Strategy:**
*   `cache-first` (default): Use for stable data that rarely changes.
*   `cache-and-network`: Use for data that might change frequently, providing instant UI feedback while fetching fresh data.
*   `network-only`: Use for critical, real-time data or mutations where the latest server state is paramount.
*   `no-cache`: Use for one-off data (e.g., authentication tokens) that should *never* be stored in the cache.

❌ **BAD:** Mutating cache objects directly or neglecting `keyFields`

```javascript
// This can lead to inconsistent cache state
cache.data.data.User:123.name = 'New Name';
```

✅ **GOOD:** Immutable cache updates with `cache.modify` or `writeFragment`

```javascript
import { useMutation, gql } from '@apollo/client';

const UPDATE_USER_NAME = gql`
  mutation UpdateUserName($id: ID!, $name: String!) {
    updateUserName(id: $id, name: $name) {
      id
      name
    }
  }
`;

function UserProfile({ userId, currentName }) {
  const [updateName] = useMutation(UPDATE_USER_NAME);

  const handleUpdate = async (newName) => {
    await updateName({
      variables: { id: userId, name: newName },
      update: (cache, { data: { updateUserName } }) => {
        cache.modify({
          id: cache.identify(updateUserName), // Get normalized cache ID
          fields: {
            name() {
              return updateUserName.name; // Update the name field
            },
          },
        });
      },
    });
  };

  return (
    <div>
      <p>Current Name: {currentName}</p>
      <button onClick={() => handleUpdate('Jane Doe')}>Change Name</button>
    </div>
  );
}
```

## 4. Type Safety & Code Generation

**Mandate `@graphql-codegen` for all GraphQL operations.** This generates TypeScript types for queries, mutations, and subscriptions, along with typed React hooks, ensuring end-to-end type safety and reducing runtime errors.

❌ **BAD:** Manual type definitions or `any`

```javascript
// Manually defining types is error-prone and tedious
interface UserData {
  users: Array<{ id: string; name: string }>;
}
const { data } = useQuery<UserData>(GET_USERS);
```

✅ **GOOD:** Generated types and hooks

```javascript
// graphql.config.js (example)
module.exports = {
  schema: 'http://localhost:4000/graphql',
  documents: 'src/**/*.graphql',
  overwrite: true,
  generates: {
    './src/generated/graphql.tsx': {
      plugins: [
        'typescript',
        'typescript-operations',
        'typescript-react-apollo',
      ],
      config: {
        withHooks: true,
        withHOC: false,
        withComponent: false,
      },
    },
  },
};

// src/components/UserList.graphql (separate file for GraphQL document)
// query GetUsers($limit: Int!) {
//   users(limit: $limit) {
//     id
//     name
//   }
// }

// src/components/UserList.tsx
import { useGetUsersQuery } from '../generated/graphql'; // Generated hook

function UserList({ limit }) {
  const { loading, error, data } = useGetUsersQuery({ variables: { limit } });

  if (loading) return <p>Loading users...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <ul>
      {data?.users.map(user => ( // Data is now fully typed
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## 5. Error Handling & Resilience

**Implement centralized error handling using Apollo Link's `onError` and `RetryLink`.** This provides a consistent way to log, display, and recover from network and GraphQL errors.

❌ **BAD:** Inconsistent error handling across components

```javascript
// Every component handles errors differently
if (error) {
  alert(error.message);
}
```

✅ **GOOD:** Centralized error handling with `onError` link (see Section 1 for example) and component-level display.

```javascript
// In a component using useQuery or useMutation
function MyComponent() {
  const { loading, error, data } = useQuery(MY_QUERY);

  if (loading) return <p>Loading...</p>;
  if (error) {
    // Specific UI for this component's error
    return <p className="error-message">Failed to load data: {error.message}</p>;
  }
  // ...
}
```

## 6. Code Organization & Naming Conventions

**Colocate GraphQL documents (`.graphql` files) with the components that use them.** Follow GraphQL schema naming conventions (`camelCase` for fields/arguments, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for enum values) and enforce them with GraphOS schema linting.

❌ **BAD:** Inline GraphQL strings, generic naming

```javascript
// In a large component file
const MY_HUGE_QUERY = gql`query { /* ... */ }`;
```

✅ **GOOD:** Dedicated GraphQL files, clear naming

```
// src/components/UserList/
// ├── UserList.tsx
// └── UserList.graphql
```

```graphql
# src/components/UserList/UserList.graphql
query GetUsers($limit: Int!) {
  users(limit: $limit) {
    id
    firstName # Use camelCase for fields
    lastName
  }
}
```

## 7. Performance Considerations

**Utilize fragments for query reuse and to avoid over-fetching.** Consider Automatic Persisted Queries (APQ) for production environments to reduce network payload sizes. Optimize `fetchPolicy` based on data volatility.

❌ **BAD:** Repeated field selections, large monolithic queries

```graphql
query GetUserAndPosts {
  user(id: "1") { id name email }
  posts(userId: "1") { id title content }
}
```

✅ **GOOD:** Use fragments for modularity and reuse

```graphql
# src/fragments/UserFragment.graphql
fragment UserDetails on User {
  id
  name
  email
}

# src/components/UserPosts/UserPosts.graphql
query GetUserAndPosts($userId: ID!) {
  user(id: $userId) {
    ...UserDetails
  }
  posts(userId: $userId) {
    id
    title
    content
  }
}
```

## 8. Testing Approaches

**Use `@apollo/client/testing`'s `MockedProvider` for unit and integration tests of components that use Apollo hooks.** This allows you to mock GraphQL operations and control the data returned, ensuring consistent and isolated tests.

❌ **BAD:** Relying on a live GraphQL API for component tests

```javascript
// This makes tests slow, flaky, and dependent on external services
render(<MyComponent />);
```

✅ **GOOD:** Mocking GraphQL responses with `MockedProvider`

```javascript
import { render, screen, waitFor } from '@testing-library/react';
import { MockedProvider } from '@apollo/client/testing';
import { GET_USERS } from './UserList.graphql'; // Import your GraphQL document
import UserList from './UserList';

const mocks = [
  {
    request: {
      query: GET_USERS,
      variables: { limit: 10 },
    },
    result: {
      data: {
        users: [
          { id: '1', name: 'Alice' },
          { id: '2', name: 'Bob' },
        ],
      },
    },
  },
];

test('renders user list', async () => {
  render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <UserList limit={10} />
    </MockedProvider>
  );

  expect(screen.getByText(/loading users.../i)).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
  });
});
```