---
description: Definitive guidelines for building robust, performant, and maintainable React-Redux applications using modern Redux Toolkit practices.
globs: **/*.{js,jsx}
---
# react-redux Best Practices

This guide outlines the essential patterns and anti-patterns for developing with `react-redux` in 2025, emphasizing Redux Toolkit (RTK) and React hooks. Adhere to these rules to ensure your application is scalable, performant, and easy to maintain.

## 1. Embrace Redux Toolkit (RTK) as the Standard

**Always use Redux Toolkit (RTK) for all new Redux logic.** RTK eliminates boilerplate, enforces immutability with Immer, and provides sensible defaults for store setup and development tooling.

❌ **BAD: Manual Redux Setup**
```javascript
// store.js
import { createStore, combineReducers, applyMiddleware } from 'redux';
import thunk from 'redux-thunk'; // Need to add middleware manually

const initialState = { count: 0 };
function counterReducer(state = initialState, action) {
  switch (action.type) {
    case 'INCREMENT': return { ...state, count: state.count + 1 };
    default: return state;
  }
}
const store = createStore(counterReducer, applyMiddleware(thunk));
```

✅ **GOOD: Redux Toolkit `configureStore` and `createSlice`**
```javascript
// src/app/store.js
import { configureStore } from '@reduxjs/toolkit';
import counterReducer from '../features/counter/counterSlice'; // Import slices

export const store = configureStore({
  reducer: {
    counter: counterReducer,
  },
  // Redux DevTools and Redux Thunk are included by default
});

// src/features/counter/counterSlice.js
import { createSlice } from '@reduxjs/toolkit';

const counterSlice = createSlice({
  name: 'counter',
  initialState: { count: 0 },
  reducers: {
    increment: (state) => {
      state.count += 1; // Immer allows direct mutation syntax, which is immutable under the hood
    },
    decrement: (state) => {
      state.count -= 1;
    },
  },
});

export const { increment, decrement } = counterSlice.actions;
export default counterSlice.reducer;
```

## 2. Prefer React-Redux Hooks (`useSelector`, `useDispatch`)

**Always use the hooks API (`useSelector`, `useDispatch`) for connecting components to the Redux store.** This simplifies component logic, improves readability, and leverages modern React patterns. Avoid the legacy `connect` HOC for new components.

❌ **BAD: `connect` HOC (for new code)**
```javascript
import { connect } from 'react-redux';

const MyComponent = ({ count, increment }) => (
  <div>
    Count: {count}
    <button onClick={increment}>Increment</button>
  </div>
);

const mapStateToProps = (state) => ({ count: state.counter.count });
const mapDispatchToProps = { increment }; // RTK actions work directly
export default connect(mapStateToProps, mapDispatchToProps)(MyComponent);
```

✅ **GOOD: `useSelector` and `useDispatch`**
```javascript
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { increment } from '../../features/counter/counterSlice'; // Action creator

const MyComponent = () => {
  const count = useSelector((state) => state.counter.count);
  const dispatch = useDispatch();

  return (
    <div>
      Count: {count}
      <button onClick={() => dispatch(increment())}>Increment</button>
    </div>
  );
};
export default MyComponent;
```

## 3. Type Your Hooks for TypeScript

When using TypeScript, **always create and use typed versions of `useDispatch` and `useSelector`**. This provides full type safety for your store state and dispatched actions, catching errors at compile time.

✅ **GOOD: Typed Hooks (e.g., `src/app/hooks.ts`)**
```typescript
// src/app/hooks.ts
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from './store'; // Adjust path to your store

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = useDispatch.withTypes<AppDispatch>();
export const useAppSelector = useSelector.withTypes<RootState>();

// Usage in component:
// const count = useAppSelector((state) => state.counter.count);
// const dispatch = useAppDispatch();
```

## 4. Organize Code by Feature (Feature Slices)

**Structure your Redux logic by feature, using the "ducks" pattern where each feature's Redux slice (reducer, actions, selectors) resides in a single file within its feature folder.** This improves co-location, maintainability, and discoverability.

❌ **BAD: By Type Folder Structure**
```
src/
  actions/
    userActions.js
    productActions.js
  reducers/
    userReducer.js
    productReducer.js
  selectors/
    userSelectors.js
    productSelectors.js
  components/
    UserList.jsx
    ProductList.jsx
```

✅ **GOOD: Feature Folder Structure**
```
src/
  features/
    user/
      UserList.jsx
      userSlice.js    // Contains reducer, actions, selectors for user
      userSelectors.js // (Optional) for complex memoized selectors
    product/
      ProductList.jsx
      productSlice.js // Contains reducer, actions, selectors for product
  app/
    store.js
    hooks.ts
    App.jsx
```

## 5. Use RTK Query for Data Fetching

**Leverage RTK Query for all server-side data fetching and caching.** It drastically reduces boilerplate for async operations, providing automatic caching, revalidation, loading states, and error handling out of the box.

❌ **BAD: Manual Thunks for Data Fetching**
```javascript
// userSlice.js (manual thunk)
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

export const fetchUsers = createAsyncThunk('users/fetchUsers', async () => {
  const response = await axios.get('/api/users');
  return response.data;
});

const userSlice = createSlice({
  name: 'users',
  initialState: { entities: [], status: 'idle' },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => { state.status = 'loading'; })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.entities = action.payload;
      })
      .addCase(fetchUsers.rejected, (state) => { state.status = 'failed'; });
  },
});
// ... component would use useDispatch and useSelector to manage state
```

✅ **GOOD: RTK Query API Slice**
```javascript
// src/services/usersApi.js
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const usersApi = createApi({
  reducerPath: 'usersApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: () => 'users',
    }),
    addUser: builder.mutation({
      query: (newUser) => ({
        url: 'users',
        method: 'POST',
        body: newUser,
      }),
      invalidatesTags: ['Users'], // Invalidate cache for 'Users' tag
    }),
  }),
  tagTypes: ['Users'], // Define tag types for caching
});

export const { useGetUsersQuery, useAddUserMutation } = usersApi;

// Usage in component:
// const { data: users, isLoading, error } = useGetUsersQuery();
// const [addUser, { isLoading: isAdding }] = useAddUserMutation();
```

## 6. Memoize Selectors for Performance

**Always use `createSelector` (re-exported from Reselect by RTK) for deriving computed state.** This memoizes the selector's result, preventing unnecessary re-renders of components when input state hasn't changed.

❌ **BAD: Non-memoized selector**
```javascript
// In component render or useSelector directly
const activeUsers = useSelector((state) =>
  state.users.filter(user => user.isActive) // Creates new array on every render
);
```

✅ **GOOD: Memoized Selector**
```javascript
// src/features/user/userSelectors.js
import { createSelector } from '@reduxjs/toolkit';

const selectUsers = (state) => state.user.entities; // Basic input selector

export const selectActiveUsers = createSelector(
  [selectUsers], // Input selectors
  (users) => users.filter(user => user.isActive) // Output selector
);

// Usage in component:
// const activeUsers = useAppSelector(selectActiveUsers);
```

## 7. Keep State Normalized

**Store entities in a normalized fashion (flat objects by ID) whenever possible.** This simplifies updates, ensures data consistency, and avoids deeply nested, redundant data structures. RTK's `createEntityAdapter` is the best tool for this.

❌ **BAD: Nested, Duplicated Data**
```javascript
// state.posts = [
//   { id: '1', title: 'Post 1', author: { id: 'a', name: 'Alice' } },
//   { id: '2', title: 'Post 2', author: { id: 'a', name: 'Alice' } }, // Author duplicated
// ];
```

✅ **GOOD: Normalized State with `createEntityAdapter`**
```javascript
// src/features/posts/postsSlice.js
import { createSlice, createEntityAdapter } from '@reduxjs/toolkit';

const postsAdapter = createEntityAdapter(); // Manages { ids: [], entities: {} }

const postsSlice = createSlice({
  name: 'posts',
  initialState: postsAdapter.getInitialState(),
  reducers: {
    postAdded: postsAdapter.addOne,
    postUpdated: postsAdapter.updateOne,
    // ... other reducers
  },
  // ... extraReducers for async data fetching
});

export const { postAdded, postUpdated } = postsSlice.actions;
export const { selectAll: selectAllPosts, selectById: selectPostById } =
  postsAdapter.getSelectors((state) => state.posts); // Generate selectors
export default postsSlice.reducer;

// Example state shape:
// state.posts = {
//   ids: ['1', '2'],
//   entities: {
//     '1': { id: '1', title: 'Post 1', authorId: 'a' },
//     '2': { id: '2', title: 'Post 2', authorId: 'a' },
//   },
// };
// (Authors would be in their own normalized slice: state.authors = { ids: ['a'], entities: { 'a': { id: 'a', name: 'Alice' } } })
```

## 8. Avoid Non-Serializable Values in State or Actions

**Never store non-serializable values (e.g., Promises, Functions, Dates, Maps, Sets, class instances) in your Redux state or dispatch them in actions that reach reducers.** This is crucial for Redux DevTools functionality (time-travel debugging) and state rehydration. Middleware (like Redux Thunk) is the only exception for handling non-serializable values *before* they reach reducers.

❌ **BAD: Storing non-serializable data**
```javascript
// In reducer:
// state.user.loggedInAt = new Date(); // Date object
// state.api.promise = fetch('/data'); // Promise object

// In action:
// dispatch({ type: 'SET_CALLBACK', payload: () => console.log('hi') }); // Function
```

✅ **GOOD: Serializable data**
```javascript
// In reducer:
state.user.loggedInAt = new Date().toISOString(); // Store as string
state.api.lastFetchTime = Date.now(); // Store as timestamp

// In action:
// dispatch(fetchData()); // Thunks (handled by middleware) are the exception for async logic
```

## 9. Separate UI-Local State from Global State

**Use React's `useState` or `useReducer` for component-specific UI state.** Only use Redux for truly global application state that needs to be shared across multiple, distant components or persisted across routes. This keeps your Redux store lean and improves performance.

❌ **BAD: All state in Redux**
```javascript
// In Redux state:
// ui.loginForm.username = '...';
// ui.loginForm.password = '...';
// ui.loginForm.isValid = false;
```

✅ **GOOD: UI-local state in component**
```javascript
import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { loginUser } from '../../features/auth/