---
description: This guide defines the definitive best practices for using Redux in our projects, mandating Redux Toolkit (RTK) for all state management logic to ensure consistency, maintainability, and optimal performance.
globs: **/*.{js,jsx}
---
# redux Best Practices

Redux is our standard for predictable state management. **All new Redux code MUST use Redux Toolkit (RTK)**. RTK simplifies Redux, enforces best practices, and eliminates boilerplate. Do not use vanilla Redux APIs directly.

## 1. Code Organization and Structure

Organize your Redux logic by feature, using a "slice" approach. This co-locates all related state, actions, and reducers, improving discoverability and maintainability.

### 1.1. Feature-First Directory Structure

Group all Redux-related files for a specific feature within a single directory.

```
src/
├── app/
│   ├── store.js             // Root store configuration
│   └── hooks.js             // Pre-typed Redux hooks (for TS)
└── features/
    ├── users/
    │   ├── usersSlice.js    // User slice (reducer, actions)
    │   └── UserList.jsx     // Component using user slice
    └── posts/
        ├── postsSlice.js    // Post slice
        └── PostEditor.jsx   // Component using post slice
```

### 1.2. Single-File Slices

Define your reducer, initial state, and action creators in a single file using `createSlice`. This is the "ducks" pattern, simplified by RTK.

❌ **BAD: Scattered Redux files**
```javascript
// src/features/posts/actions.js
export const addPost = (payload) => ({ type: 'posts/addPost', payload });

// src/features/posts/reducers.js
const postsReducer = (state = [], action) => { /* ... */ };

// src/features/posts/constants.js
const ADD_POST = 'posts/addPost';
```

✅ **GOOD: Co-located slice with `createSlice`**
```javascript
// src/features/posts/postsSlice.js
import { createSlice, nanoid } from '@reduxjs/toolkit';

const postsSlice = createSlice({
  name: 'posts',
  initialState: [],
  reducers: {
    postAdded: {
      reducer(state, action) {
        state.push(action.payload);
      },
      prepare(title, content, userId) {
        return {
          payload: {
            id: nanoid(),
            title,
            content,
            user: userId,
            date: new Date().toISOString(),
          },
        };
      },
    },
    // ... other reducers
  },
});

export const { postAdded } = postsSlice.actions;
export default postsSlice.reducer;
```

### 1.3. Root Store Configuration

Centralize your store setup in `src/app/store.js` using `configureStore`. This automatically includes `redux-thunk`, Redux DevTools integration, and development-time checks for common mistakes.

```javascript
// src/app/store.js
import { configureStore } from '@reduxjs/toolkit';
import usersReducer from '../features/users/usersSlice';
import postsReducer from '../features/posts/postsSlice';

export const store = configureStore({
  reducer: {
    users: usersReducer,
    posts: postsReducer,
  },
  // middleware and devTools are configured by default
});
```

## 2. State Organization

Keep your state as flat and normalized as possible, especially for collections of items.

### 2.1. Normalize Data for Collections

Store collections of items (e.g., users, posts) in a normalized way: an object mapping IDs to item objects, plus an array of IDs. This prevents data duplication and simplifies updates.

❌ **BAD: Array of objects (hard to update)**
```javascript
// state.posts = [{ id: '1', title: 'A', author: { id: 'u1', name: 'Alice' } }, ...]
// Updating Alice's name requires iterating through all posts.
```

✅ **GOOD: Normalized state (easy to update)**
```javascript
// state.posts = {
//   ids: ['1', '2'],
//   entities: {
//     '1': { id: '1', title: 'A', authorId: 'u1' },
//     '2': { id: '2', title: 'B', authorId: 'u2' },
//   },
// };
// state.users = {
//   ids: ['u1', 'u2'],
//   entities: {
//     'u1': { id: 'u1', name: 'Alice' },
//     'u2': { id: 'u2', name: 'Bob' },
//   },
// };
// Updating Alice's name is a single direct lookup: state.users.entities['u1'].name = 'Alicia'
```
Use RTK's `createEntityAdapter` to simplify managing normalized state.

## 3. Side Effects

Handle all side effects (async logic, API calls) outside of reducers.

### 3.1. Use RTK Query for Data Fetching

**RTK Query is the preferred solution for data fetching and caching.** It drastically reduces boilerplate for API interactions.

❌ **BAD: Manual `createAsyncThunk` for every API call**
```javascript
// src/features/posts/postsSlice.js
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import axios from 'axios';

export const fetchPosts = createAsyncThunk('posts/fetchPosts', async () => {
  const response = await axios.get('/api/posts');
  return response.data;
});
// ... then manually handle loading states, errors, caching
```

✅ **GOOD: RTK Query for data fetching**
```javascript
// src/services/postsApi.js
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const postsApi = createApi({
  reducerPath: 'postsApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getPosts: builder.query({
      query: () => 'posts',
    }),
    addPost: builder.mutation({
      query: (newPost) => ({
        url: 'posts',
        method: 'POST',
        body: newPost,
      }),
    }),
  }),
});

export const { useGetPostsQuery, useAddPostMutation } = postsApi;
// Automatically handles loading, error, caching, re-fetching.
```

### 3.2. Use `createAsyncThunk` for Complex Async Logic

For async logic that isn't simple data fetching (e.g., complex multi-step processes, interacting with non-API services), use `createAsyncThunk`.

```javascript
// src/features/notifications/notificationsSlice.js
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';

export const markAllNotificationsRead = createAsyncThunk(
  'notifications/markAllRead',
  async (_, { getState }) => {
    const { notifications } = getState();
    // Simulate complex logic, e.g., batch update in DB
    await someExternalService.updateMany(notifications.map(n => n.id));
    return notifications.map(n => n.id);
  }
);
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to prevent bugs and ensure predictable state.

### 4.1. Never Mutate State Outside of Reducers (or Immer)

Directly modifying the Redux state is the #1 cause of bugs. `createSlice` uses Immer, allowing "mutative" syntax safely within reducers.

❌ **BAD: Mutating state directly**
```javascript
// In a component or thunk
dispatch(someAction());
store.getState().posts[0].title = 'New Title'; // 🚨 DANGER!
```

❌ **BAD: Mutating state in a vanilla reducer**
```javascript
// In a reducer without Immer
const postsReducer = (state = [], action) => {
  if (action.type === 'posts/update') {
    state[0].title = action.payload.title; // 🚨 DANGER!
    return state; // Returns mutated state
  }
  return state;
};
```

✅ **GOOD: Safe immutable updates with `createSlice` (Immer)**
```javascript
// In a createSlice reducer
const postsSlice = createSlice({
  name: 'posts',
  initialState: [],
  reducers: {
    postUpdated(state, action) {
      const { id, title } = action.payload;
      const existingPost = state.find(post => post.id === id);
      if (existingPost) {
        existingPost.title = title; // ✅ Safe with Immer!
      }
    },
  },
});
```

### 4.2. Reducers Must Be Pure

Reducers must be pure functions: given the same inputs (state, action), they must always produce the same output, without side effects.

❌ **BAD: Reducer with side effects**
```javascript
const postsReducer = (state = [], action) => {
  if (action.type === 'posts/add') {
    console.log('Adding post!'); // 🚨 Side effect
    return [...state, { ...action.payload, timestamp: Date.now() }]; // 🚨 Non-deterministic
  }
  return state;
};
```

✅ **GOOD: Pure reducer**
```javascript
const postsSlice = createSlice({
  name: 'posts',
  initialState: [],
  reducers: {
    postAdded: {
      reducer(state, action) {
        state.push(action.payload);
      },
      prepare(title, content) {
        return {
          payload: {
            id: nanoid(),
            title,
            content,
            date: new Date().toISOString(), // ✅ Deterministic if called in prepare
          },
        };
      },
    },
  },
});
```

### 4.3. Avoid Non-Serializable Values in State or Actions

Do not store Promises, Symbols, Maps/Sets, functions, or class instances in your Redux state or dispatched actions. This breaks DevTools and state rehydration.

❌ **BAD: Non-serializable value in state**
```javascript
// state.user = { id: '1', name: 'Alice', promise: somePromise() } // 🚨 DANGER!
```

✅ **GOOD: Serializable values only**
```javascript
// state.user = { id: '1', name: 'Alice', status: 'pending' } // ✅
```
**Exception**: Non-serializable values are allowed in actions *if* a middleware (like `redux-thunk`) intercepts and consumes them before they reach the reducers.

## 5. Performance Considerations

Optimize your components to re-render only when necessary.

### 5.1. Use Memoized Selectors

Use `createSelector` (from RTK or Reselect) to create memoized selectors. These re-compute derived data only when their inputs change, preventing unnecessary component re-renders.

❌ **BAD: Inline selector in `useSelector` (can cause unnecessary re-renders)**
```javascript
// In a component
const activeUsers = useSelector(state =>
  state.users.filter(user => user.isActive) // 🚨 New array reference on every render
);
```

✅ **GOOD: Memoized selector**
```javascript
// src/features/users/usersSelectors.js
import { createSelector } from '@reduxjs/toolkit';

const selectUsers = state => state.users;
export const selectActiveUsers = createSelector(
  [selectUsers],
  users => users.filter(user => user.isActive) // ✅ Only re-computes if users array changes
);

// In a component
const activeUsers = useSelector(selectActiveUsers);
```

### 5.2. Select Minimal Data

Components should select the smallest possible amount of data they need from the store. This reduces the chance of unnecessary re-renders.

❌ **BAD: Selecting entire slice**
```javascript
const user = useSelector(state => state.users); // 🚨 Re-renders if *any* user data changes
```

✅ **GOOD: Selecting specific data**
```javascript
const userName = useSelector(state => state.users.currentUser.name); // ✅ Re-renders only if name changes
```

## 6. Hooks Best Practices

Always use the pre-typed `useAppSelector` and `useAppDispatch` hooks for type safety and consistency.

### 6.1. Pre-Typed Hooks

Define and export pre-typed versions of `useSelector` and `useDispatch` in `src/app/hooks.js`. This provides compile-time type safety across your application.

```javascript
// src/app/hooks.js
import { useDispatch, useSelector } from 'react-redux';

// Import RootState and AppDispatch from your store
// (assuming store.js exports them for TypeScript)
// import type { RootState, AppDispatch } from './store';

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = useDispatch; // as () => AppDispatch; (for TS)
export const useAppSelector = useSelector; // as <T>(selector: (state: RootState) => T) => T; (for TS)
```

### 6.2. Dispatch Actions Correctly

Use `useAppDispatch` to get the dispatch function and call your RTK-generated action creators.

```javascript
// In a component
import { useAppDispatch } from '@/app/hooks';
import { postAdded } from '@/features/posts/postsSlice';

const AddPostForm = () => {
  const dispatch = useAppDispatch();
  const handleSubmit = () => {
    dispatch(postAdded('My Title', 'My Content', 'user123'));
  };
  // ...
};
```

## 7. Debugging

Leverage RTK's built-in debugging capabilities.

### 7.1. Redux DevTools

`configureStore` automatically sets up Redux DevTools integration. Ensure you have the browser extension installed.

### 7.2. Immutable State Invariant Middleware

In development, `configureStore` includes middleware that checks for accidental state mutations. Pay attention to console warnings. If you see them, fix the mutation immediately.