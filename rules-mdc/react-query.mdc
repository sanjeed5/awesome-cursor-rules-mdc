---
description: Definitive guidelines for using TanStack Query (formerly React Query) to manage server state efficiently, ensure type safety, and optimize performance in React applications.
globs: **/*.{js,jsx,ts,tsx}
---
# TanStack Query (react-query) Best Practices

This document outlines the definitive best practices for using TanStack Query in our React applications. Adhering to these guidelines ensures consistent, performant, and maintainable data fetching and state management.

## 1. Query Keys: The Foundation of Caching

**ALWAYS** use stable, descriptive array keys. These are fundamental for caching, refetching, and invalidation. For dynamic data, embed parameters directly into the array.

### ✅ GOOD: Stable Array Keys & Key Factories

```typescript
// 1. Simple, static key
const USERS_KEY = ['users'];

// 2. Dynamic key with parameters
const userKeys = {
  all: ['users'] as const,
  lists: () => [...userKeys.all, 'list'] as const,
  list: (filters: { status?: string; page?: number }) =>
    [...userKeys.lists(), { filters }] as const,
  details: () => [...userKeys.all, 'detail'] as const,
  detail: (id: string) => [...userKeys.details(), id] as const,
};

// Usage:
// useQuery(userKeys.all, fetchAllUsers);
// useQuery(userKeys.list({ status: 'active', page: 1 }), fetchUsers);
// useQuery(userKeys.detail(userId), fetchUserById);
```

### ❌ BAD: Unstable or Generic Keys

```typescript
// String keys are less flexible for dynamic data and filtering
useQuery('users', fetchUsers);

// Anonymous object keys are unstable and break caching
useQuery(['users', { id: userId }], fetchUserById); // Object literal creates new reference each render
```

## 2. Custom Hooks: Encapsulate Logic

**ALWAYS** wrap `useQuery` and `useMutation` calls in custom hooks. This centralizes data fetching logic, improves reusability, enhances type safety, and keeps components clean.

### ✅ GOOD: Dedicated Custom Hooks

```typescript
// hooks/useUsers.ts
import { useQuery } from '@tanstack/react-query';
import { fetchUsers, User } from '../api'; // Assume api.ts defines fetchUsers

const userKeys = {
  all: ['users'] as const,
  list: (filters: { status?: string }) => [...userKeys.all, { filters }] as const,
};

export function useUsers(filters?: { status?: string }) {
  return useQuery<User[], Error>({
    queryKey: userKeys.list(filters || {}),
    queryFn: () => fetchUsers(filters),
  });
}

// components/UserList.tsx
import { useUsers } from '../hooks/useUsers';

function UserList({ statusFilter }: { statusFilter?: string }) {
  const { data: users, isLoading, error } = useUsers({ status: statusFilter });

  if (isLoading) return <div>Loading users...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {users?.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### ❌ BAD: Direct `useQuery` in Components

```typescript
// components/UserList.tsx
import { useQuery } from '@tanstack/react-query';
import { fetchUsers } from '../api';

function UserList({ statusFilter }: { statusFilter?: string }) {
  // Logic is duplicated if another component needs users
  // Query key is less organized
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users', { status: statusFilter }],
    queryFn: () => fetchUsers({ status: statusFilter }),
  });

  // ... rest of component
}
```

## 3. Query Functions: Separate and Stable

**NEVER** pass anonymous functions directly to `queryFn`. **ALWAYS** declare `queryFn` separately to ensure stability, prevent unnecessary re-renders, and improve testability.

### ✅ GOOD: Separated Query Functions

```typescript
// api.ts
export async function fetchUserById(id: string): Promise<User> {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) throw new Error('Failed to fetch user');
  return response.json();
}

// hooks/useUser.ts
import { useQuery } from '@tanstack/react-query';
import { fetchUserById } from '../api';

const userKeys = {
  detail: (id: string) => ['users', id] as const,
};

export function useUser(userId: string) {
  return useQuery({
    queryKey: userKeys.detail(userId),
    queryFn: () => fetchUserById(userId), // Stable reference to fetchUserById
    enabled: !!userId, // Only run if userId exists
  });
}
```

### ❌ BAD: Anonymous Query Functions

```typescript
// hooks/useUser.ts
import { useQuery } from '@tanstack/react-query';

export function useUser(userId: string) {
  return useQuery({
    queryKey: ['users', userId],
    // This anonymous function is recreated on every render, potentially causing issues
    queryFn: async () => {
      const response = await fetch(`/api/users/${userId}`);
      if (!response.ok) throw new Error('Failed to fetch user');
      return response.json();
    },
    enabled: !!userId,
  });
}
```

## 4. Conditional Fetching: Use `enabled`

**ALWAYS** use the `enabled` option for conditional fetching. This is the explicit and recommended way to control when a query runs.

### ✅ GOOD: Using `enabled`

```typescript
// hooks/useUserProfile.ts
import { useQuery } from '@tanstack/react-query';
import { fetchUserProfile } from '../api';

export function useUserProfile(userId?: string) {
  return useQuery({
    queryKey: ['userProfile', userId],
    queryFn: () => fetchUserProfile(userId!),
    enabled: !!userId, // Query only runs if userId is truthy
  });
}
```

### ❌ BAD: Conditional Hook Calls

```typescript
// components/UserProfile.tsx
import { useUserProfile } from '../hooks/useUserProfile';

function UserProfile({ userId }: { userId?: string }) {
  // React Hook Rules: Hooks must be called unconditionally
  // This breaks the rules and will cause bugs
  if (!userId) {
    return null;
  }
  const { data: user, isLoading } = useUserProfile(userId);
  // ...
}
```

## 5. Data Transformation: Use `select`

**ALWAYS** use the `select` option within `useQuery` for transforming or filtering data. This ensures the transformation happens once at the query level, optimizing performance and preventing redundant calculations in components.

### ✅ GOOD: `select` for Transformations

```typescript
// hooks/useActiveUsers.ts
import { useQuery } from '@tanstack/react-query';
import { fetchUsers, User } from '../api';

export function useActiveUsers() {
  return useQuery<User[], Error, string[]>({ // Specify transformed data type
    queryKey: ['users', 'all'],
    queryFn: fetchUsers,
    select: (data) => data.filter(user => user.status === 'active').map(user => user.name),
  });
}

// components/ActiveUserNames.tsx
import { useActiveUsers } from '../hooks/useActiveUsers';

function ActiveUserNames() {
  const { data: activeUserNames, isLoading } = useActiveUsers();

  if (isLoading) return <div>Loading active users...</div>;
  return (
    <ul>
      {activeUserNames?.map((name) => (
        <li key={name}>{name}</li>
      ))}
    </ul>
  );
}
```

### ❌ BAD: Transforming Data in Every Component

```typescript
// components/ActiveUserNames.tsx
import { useQuery } from '@tanstack/react-query';
import { fetchUsers } from '../api';

function ActiveUserNames() {
  const { data: users, isLoading } = useQuery({
    queryKey: ['users', 'all'],
    queryFn: fetchUsers,
  });

  // Transformation logic repeated or inefficiently placed
  const activeUserNames = users?.filter(user => user.status === 'active').map(user => user.name);

  if (isLoading) return <div>Loading active users...</div>;
  return (
    <ul>
      {activeUserNames?.map((name) => (
        <li key={name}>{name}</li>
      ))}
    </ul>
  );
}
```

## 6. Mutations and Cache Invalidation

**ALWAYS** use `useMutation` for CUD (Create, Update, Delete) operations. After a successful mutation, **ALWAYS** invalidate relevant queries to ensure the UI reflects the latest server state. For immediate feedback, consider optimistic updates with `setQueryData`.

### ✅ GOOD: Invalidation after Mutation

```typescript
// hooks/useCreateTodo.ts
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createTodo, Todo } from '../api';

export function useCreateTodo() {
  const queryClient = useQueryClient();
  return useMutation<Todo, Error, { title: string }>({
    mutationFn: createTodo,
    onSuccess: () => {
      // Invalidate all 'todos' queries to refetch fresh data
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });
}

// components/TodoForm.tsx
import { useCreateTodo } from '../hooks/useCreateTodo';

function TodoForm() {
  const { mutate, isLoading } = useCreateTodo();

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget as HTMLFormElement);
    const title = formData.get('title') as string;
    mutate({ title });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="title" placeholder="New todo" />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Adding...' : 'Add Todo'}
      </button>
    </form>
  );
}
```

### ✅ GOOD: Optimistic Updates

```typescript
// hooks/useUpdateTodo.ts
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { updateTodo, Todo } from '../api';

export function useUpdateTodo() {
  const queryClient = useQueryClient();
  return useMutation<Todo, Error, Partial<Todo> & { id: string }>({
    mutationFn: updateTodo,
    // Optimistically update the cache
    onMutate: async (newTodo) => {
      await queryClient.cancelQueries({ queryKey: ['todos'] });
      const previousTodos = queryClient.getQueryData<Todo[]>(['todos']);
      queryClient.setQueryData<Todo[]>(['todos'], (old) =>
        old ? old.map((todo) => (todo.id === newTodo.id ? { ...todo, ...newTodo } : todo)) : []
      );
      return { previousTodos }; // Context for onError
    },
    onError: (err, newTodo, context) => {
      // Rollback on error
      queryClient.setQueryData(['todos'], context?.previousTodos);
    },
    onSettled: () => {
      // Always refetch after error or success to ensure data is in sync
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });
}
```

## 7. Performance: Prefetching & Defaults

**LEVERAGE** TanStack Query's defaults (e.g., `staleTime: 0`, automatic retries, refetch on window focus) and **STRATEGICALLY** use prefetching for critical user flows.

### ✅ GOOD: Prefetching for Router Integration

```typescript
// utils/routeLoaders.ts (Example with a router loader)
import { QueryClient } from '@tanstack/react-query';
import { fetchProjectById } from '../api';

export const projectLoader = (queryClient: QueryClient) => async ({ params }: { params: { projectId: string } }) => {
  const queryKey = ['projects', params.projectId];
  // Prefetch the project data during navigation
  await queryClient.prefetchQuery({
    queryKey,
    queryFn: () => fetchProjectById(params.projectId),
  });
  return null; // Or return initial data if needed
};

// components/ProjectLink.tsx
import { Link } from 'react-router-dom'; // Assuming react-router-dom
import { useQueryClient } from '@tanstack/react-query';
import { fetchProjectById } from '../api';

function ProjectLink({ projectId, projectName }: { projectId: string; projectName: string }) {
  const queryClient = useQueryClient();
  const handleMouseEnter = () => {
    // Prefetch on hover for instant page loads
    queryClient.prefetchQuery({
      queryKey: ['projects', projectId],
      queryFn: () => fetchProjectById(projectId),
      staleTime: 5 * 60 * 1000, // Keep data fresh for 5 minutes
    });
  };

  return (
    <Link to={`/projects/${projectId}`} onMouseEnter={handleMouseEnter}>
      {projectName}
    </Link>
  );
}
```

## 8. ESLint Plugin: Enforce Standards

**ALWAYS** install and configure the `@tanstack/query-eslint-plugin`. It enforces many of these best practices automatically, catching common mistakes early.

```json
// .eslintrc.json
{
  "plugins": ["@tanstack/query"],
  "rules": {
    "@tanstack/query/exhaustive-deps": "error",
    "@tanstack/query/prefer-query-object": "error",
    "@tanstack/query/stable-query-client": "error"
  }
}
```