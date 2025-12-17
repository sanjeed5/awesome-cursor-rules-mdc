---
description: Definitive guidelines for using axios to build robust, maintainable, and performant HTTP clients in JavaScript/React applications.
globs: **/*.{js,jsx}
---
# axios Best Practices

`axios` is the go-to HTTP client for modern JavaScript applications due to its robust features and promise-based API. These guidelines ensure your team leverages `axios` effectively, promoting clean code, centralized logic, and predictable error handling.

## 1. Centralize Your `axios` Instance

Always create a single, pre-configured `axios` instance for your application. This centralizes `baseURL`, `timeout`, and default headers, adhering to the DRY principle and simplifying configuration changes.

❌ BAD: Scattered `axios` calls
```javascript
// In component A
axios.get('https://api.example.com/users', { timeout: 5000 });

// In component B
axios.post('https://api.example.com/products', data, { headers: { 'Content-Type': 'application/json' } });
```

✅ GOOD: Dedicated `apiClient` instance
```javascript
// src/api/apiClient.js
import axios from 'axios';

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'https://api.example.com',
  timeout: 10000, // 10 seconds
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

export default apiClient;
```

## 2. Abstract API Endpoints into Modules

Encapsulate each API endpoint or resource into its own module. This promotes the single-responsibility principle, making your API calls testable, reusable, and easy to understand.

❌ BAD: API logic directly in components
```javascript
// src/components/UserList.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios'; // Direct axios import

function UserList() {
  const [users, setUsers] = useState([]);
  useEffect(() => {
    axios.get('https://api.example.com/users') // Hardcoded URL
      .then(response => setUsers(response.data))
      .catch(error => console.error(error));
  }, []);
  // ...
}
```

✅ GOOD: Dedicated API service modules
```javascript
// src/api/users.js
import apiClient from './apiClient'; // Use the centralized instance

export const getUsers = async () => {
  const response = await apiClient.get('/users');
  return response.data;
};

export const createUser = async (userData) => {
  const response = await apiClient.post('/users', userData);
  return response.data;
};

// src/components/UserList.jsx
import React, { useEffect, useState } from 'react';
import { getUsers } from '../api/users'; // Import specific API functions

function UserList() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const data = await getUsers();
        setUsers(data);
      } catch (err) {
        setError('Failed to fetch users.'); // User-friendly error
        console.error(err); // Log original error for debugging
      } finally {
        setLoading(false);
      }
    };
    fetchUsers();
  }, []);

  if (loading) return <div>Loading users...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## 3. Leverage Interceptors for Global Logic

Use `axios` interceptors for cross-cutting concerns like authentication, global error handling, logging, or request/response transformations. This keeps your API functions clean and focused on data fetching.

### Request Interceptors (e.g., Auth Tokens)

Inject authentication tokens automatically into every request.

```javascript
// src/api/apiClient.js (continued)
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken'); // Or from a state management solution
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);
```

### Response Interceptors (e.g., Global Error Handling)

Handle common error statuses (e.g., 401 Unauthorized, 500 Server Error) globally. Map raw errors to user-friendly messages.

```javascript
// src/api/apiClient.js (continued)
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // Handle 401 Unauthorized globally
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      // Example: Redirect to login page or attempt token refresh
      console.warn('Unauthorized request, redirecting to login...');
      // window.location.href = '/login'; // Or dispatch a global event
      return Promise.reject(new Error('Session expired. Please log in again.')); // Return a user-friendly error
    }

    // Map other HTTP errors to user-friendly messages
    let errorMessage = 'An unexpected error occurred.';
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // Server responded with a status other than 2xx
        errorMessage = error.response.data?.message || `Server Error: ${error.response.status}`;
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = 'No response from server. Please check your internet connection.';
      } else {
        // Something else happened while setting up the request
        errorMessage = error.message;
      }
    } else {
      errorMessage = error.message || 'An unknown error occurred.';
    }

    console.error('API Error:', error); // Log full error for debugging
    return Promise.reject(new Error(errorMessage)); // Propagate user-friendly error
  }
);
```

## 4. Implement Request Cancellation with `AbortController`

Prevent memory leaks and unnecessary network activity, especially in React components that might unmount before a request completes. Always use `AbortController` for cancellation; `CancelToken` is deprecated.

❌ BAD: Ignoring cancellation or using deprecated `CancelToken`
```javascript
// In a React component
useEffect(() => {
  axios.get('/long-running-data').then(...); // Request might complete after component unmounts
}, []);

// Using deprecated CancelToken
const CancelToken = axios.CancelToken;
const source = CancelToken.source();
axios.get('/data', { cancelToken: source.token });
source.cancel('Operation cancelled.');
```

✅ GOOD: Using `AbortController` in `useEffect` cleanup
```javascript
// src/components/DataFetcher.jsx
import React, { useEffect, useState } from 'react';
import apiClient from '../api/apiClient';
import axios from 'axios'; // Import axios to use axios.isCancel

function DataFetcher() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController(); // Create AbortController
    const signal = controller.signal;

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await apiClient.get('/some-data', { signal }); // Pass signal to axios
        setData(response.data);
      } catch (err) {
        if (axios.isCancel(err)) {
          console.log('Request cancelled:', err.message);
        } else {
          setError(err.message); // Use the user-friendly message from interceptor
          console.error('Fetch error:', err);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => {
      controller.abort('Component unmounted'); // Abort request on cleanup
    };
  }, []);

  if (loading) return <div>Loading data...</div>;
  if (error) return <div>Error: {error}</div>;
  return <div>{JSON.stringify(data)}</div>;
}
```

## 5. Validate Payloads Before Sending

Always validate request payloads on the client-side before sending them to the server. This reduces unnecessary network requests and provides immediate feedback to the user.

❌ BAD: Sending potentially invalid data
```javascript
const handleSubmit = async (formData) => {
  // No client-side validation
  await createUser(formData);
};
```

✅ GOOD: Client-side validation
```javascript
import { createUser } from '../api/users';

const handleSubmit = async (formData) => {
  if (!formData.name || formData.name.length < 3) {
    alert('Name must be at least 3 characters.');
    return;
  }
  if (!formData.email || !/^\S+@\S+\.\S+$/.test(formData.email)) {
    alert('Please enter a valid email address.');
    return;
  }

  try {
    await createUser(formData);
    alert('User created successfully!');
  } catch (error) {
    alert(`Failed to create user: ${error.message}`); // Display user-friendly error
  }
};
```

## 6. Use TypeScript for API Contracts

Define explicit TypeScript interfaces for your request payloads and response data. This improves maintainability, provides compile-time safety, and enhances developer experience with autocompletion.

```typescript
// src/types/api.ts
export interface User {
  id: number;
  name: string;
  email: string;
}

export interface CreateUserPayload {
  name: string;
  email: string;
  password?: string;
}

// src/api/users.ts (example with TypeScript)
import apiClient from './apiClient';
import { User, CreateUserPayload } from '../types/api';

export const getUsers = async (): Promise<User[]> => {
  const response = await apiClient.get<User[]>('/users');
  return response.data;
};

export const createUser = async (userData: CreateUserPayload): Promise<User> => {
  const response = await apiClient.post<User>('/users', userData);
  return response.data;
};
```

## 7. Test API Interactions with Mocking

For unit and integration tests, mock `axios` requests to avoid making actual network calls. This makes tests faster, more reliable, and independent of external API availability. `axios-mock-adapter` is a common choice.

```javascript
// src/api/users.test.js
import MockAdapter from 'axios-mock-adapter';
import apiClient from './apiClient';
import { getUsers, createUser } from './users';

const mock = new MockAdapter(apiClient);

describe('users API', () => {
  afterEach(() => {
    mock.reset(); // Reset mocks after each test
  });

  it('should fetch users successfully', async () => {
    const mockUsers = [{ id: 1, name: 'Alice', email: 'alice@example.com' }];
    mock.onGet('/users').reply(200, mockUsers);

    const users = await getUsers();
    expect(users).toEqual(mockUsers);
  });

  it('should create a user successfully', async () => {
    const newUserPayload = { name: 'Bob', email: 'bob@example.com' };
    const createdUser = { id: 2, ...newUserPayload };
    mock.onPost('/users').reply(201, createdUser);

    const user = await createUser(newUserPayload);
    expect(user).toEqual(createdUser);
  });

  it('should handle API errors gracefully', async () => {
    mock.onGet('/users').reply(500, { message: 'Internal Server Error' });

    await expect(getUsers()).rejects.toThrow('Server Error: 500'); // Based on interceptor error message
  });
});
```