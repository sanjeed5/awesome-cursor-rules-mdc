---
description: Enforces modern JavaScript best practices and leverages Bun's integrated tooling for high-performance, maintainable backend services.
globs: **/*.{js,jsx}
---
# bun Best Practices

Bun is our go-to runtime for high-performance JavaScript/TypeScript backend services. It's an all-in-one toolkit designed for speed and developer experience. Adhere to these guidelines to maximize Bun's potential and maintain code quality.

## 1. Embrace Bun's Integrated Toolchain

Bun's strength lies in its unified toolchain. Always default to Bun's built-in features over external alternatives unless a specific project requirement dictates otherwise.

### ✅ GOOD: Use Bun's native tools
- **Package Management:** `bun install` for dependencies, `bun add` for new packages.
- **Bundling:** `bun build` for zero-config bundling and standalone executables.
- **Testing:** `bun test` for Jest-compatible testing, including watch mode and coverage.
- **Runtime:** `bun run` or `bun <file>` for execution.

```bash
# Install dependencies (faster than npm/yarn)
bun install

# Add a new package
bun add zod

# Run tests
bun test --coverage

# Build for deployment (e.g., a serverless function)
bun build ./src/index.ts --target=bun --outfile=./dist/server
```

### ❌ BAD: Mixing package managers or external bundlers unnecessarily
Avoid `npm install` or `yarn add` in Bun projects. Don't use Webpack or Rollup if `bun build` suffices.

## 2. Modern JavaScript Language Features

Always use current ECMAScript features (ES2015+). This improves readability, reduces bugs, and aligns with modern development.

### ✅ GOOD: Modern JS syntax
```javascript
// 1. Prefer `const` and `let` over `var`
const API_URL = 'https://api.example.com';
let retryCount = 0;

// 2. Use ES Modules for all imports/exports
import { serve } from 'bun'; // Bun's native HTTP server
import { z } from 'zod';

// 3. Use Classes for object-oriented patterns
class UserService {
  #users = new Map(); // 4. Private class fields for true encapsulation

  constructor() {
    this.#users.set('1', { id: '1', name: 'Alice' });
  }

  // 5. Arrow functions for methods to preserve `this` context
  getUser = (id) => {
    return this.#users.get(id);
  };
}

// 6. Nullish coalescing (??) for default values
const user = new UserService().getUser('2');
const userName = user?.name ?? 'Guest'; // 7. Optional chaining (?.) for safe property access

// 8. Async/await for asynchronous operations
async function fetchData(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch:', error);
    return null;
  }
}

// 9. Use `Map` for key-value pairs where keys aren't always strings or order matters
const configMap = new Map([
  ['port', 3000],
  ['debugMode', true],
]);

// 10. Reliable Array checks
const data = [];
if (Array.isArray(data)) {
  console.log('Data is an array.');
}
```

### ❌ BAD: Outdated JS patterns
```javascript
// var is function-scoped and allows redeclaration
var API_URL = 'https://api.example.com';

// CommonJS modules
const { serve } = require('bun');

// Prototype-based functions
function UserServiceOld() {
  this.users = {};
}
UserServiceOld.prototype.getUser = function(id) { /* ... */ };

// Relying on || for default values (treats 0, '', false as falsy)
const count = 0;
const defaultCount = count || 10; // defaultCount is 10, not 0 (incorrect)

// Chaining .then() for async operations
fetch('/api').then(res => res.json()).catch(err => console.error(err));
```

## 3. Code Organization and Structure

Maintain a clear, consistent project structure. This enhances navigability and maintainability, especially in larger applications.

### ✅ GOOD: Logical directory structure
```
.
├── src/
│   ├── api/             # HTTP route handlers
│   │   ├── users.js
│   │   └── index.js
│   ├── services/        # Business logic, data access
│   │   ├── userService.js
│   │   └── authService.js
│   ├── utils/           # Helper functions
│   │   └── validators.js
│   ├── middleware/      # Express-style middleware
│   │   └── auth.js
│   └── index.js         # Main application entry point
├── tests/               # Unit and integration tests
│   ├── api.test.js
│   └── services.test.js
├── .env                 # Environment variables
├── bunfig.toml          # Bun-specific configuration
└── package.json
```

## 4. Performance Considerations (Bun Specific)

Leverage Bun's native APIs for I/O and common tasks. They are highly optimized and significantly faster than Node.js equivalents or external libraries.

### ✅ GOOD: Utilize native Bun APIs
```javascript
// Native HTTP server with Bun.serve
Bun.serve({
  port: Bun.env.PORT || 3000,
  fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === '/') {
      return new Response('Hello, Bun!');
    }
    if (url.pathname === '/users') {
      // Use Bun's native SQLite or Redis client for data
      const db = new Bun.SQLite('mydb.sqlite');
      const users = db.query('SELECT * FROM users').all();
      return Response.json(users);
    }
    return new Response('404 Not Found', { status: 404 });
  },
});

// Reading files efficiently
const fileContent = await Bun.file('./data.json').json();
```

### ❌ BAD: Relying on Node.js-compatible but slower alternatives
```javascript
// Using Express.js for simple HTTP servers (adds overhead)
import express from 'express';
const app = express();
app.get('/', (req, res) => res.send('Hello, Express!'));
app.listen(3000);

// Reading files with fs/promises (slower for large files than Bun.file)
import { readFile } from 'node:fs/promises';
const content = JSON.parse(await readFile('./data.json', 'utf8'));
```

## 5. Error Handling

Implement robust error handling to prevent crashes and provide meaningful feedback. Centralize error handling for consistency.

### ✅ GOOD: Centralized async error handling
```javascript
// src/middleware/errorHandler.js
export function errorHandler(err, req, res, next) {
  console.error(err); // Log the full error for debugging
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    message: err.message || 'An unexpected error occurred.',
    ...(Bun.env.NODE_ENV === 'development' && { stack: err.stack }), // Include stack in dev
  });
}

// src/index.js (example with Bun.serve)
Bun.serve({
  port: Bun.env.PORT || 3000,
  async fetch(req) {
    try {
      // ... your route logic ...
      if (req.url.includes('/error')) {
        throw new Error('Simulated error!');
      }
      return new Response('OK');
    } catch (error) {
      // Pass error to a custom handler or return a generic error response
      return new Response(JSON.stringify({ message: error.message }), {
        status: error.statusCode || 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  },
});
```

### ❌ BAD: Unhandled promise rejections or silent errors
```javascript
// Missing try/catch in async function
async function processData() {
  const data = await fetchData('/non-existent-api'); // This will crash if not caught
  console.log(data);
}
processData();
```

## 6. Security Best Practices

Security is paramount. Always protect sensitive data and validate inputs.

### ✅ GOOD: Secure practices
```javascript
// 1. Use .env for sensitive data and access via Bun.env
// .env file:
// DB_PASSWORD=supersecret
// JWT_SECRET=anothersecret

// Access in code:
const dbPassword = Bun.env.DB_PASSWORD;
const jwtSecret = Bun.env.JWT_SECRET;

// 2. Input validation with Zod
const userSchema = z.object({
  username: z.string().min(3).max(20),
  email: z.string().email(),
  password: z.string().min(8),
});

function validateUser(data) {
  try {
    return userSchema.parse(data);
  } catch (error) {
    throw new Error(`Validation failed: ${error.errors.map(e => e.message).join(', ')}`);
  }
}

// 3. Configure CORS explicitly
const corsHeaders = {
  'Access-Control-Allow-Origin': 'https://our-frontend.com', // Specific origin
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

Bun.serve({
  fetch(req) {
    if (req.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    // ... other logic ...
    return new Response('Data', { headers: corsHeaders });
  },
});

// 4. Regularly audit dependencies
// Run `bun audit` in CI/CD pipelines.
```

### ❌ BAD: Hardcoding secrets or trusting user input
```javascript
// Hardcoding secrets
const DB_PASSWORD = 'supersecretpassword';

// No input validation
function createUser(data) {
  // Directly use data without checking types or constraints
  console.log(data.username);
}

// Wildcard CORS (exposes API to all origins)
const corsHeaders = { 'Access-Control-Allow-Origin': '*' };
```

## 7. Testing Approaches

Use `bun test` as your primary test runner. Write focused unit tests and broader integration tests to ensure reliability.

### ✅ GOOD: Comprehensive testing with `bun test`
```javascript
// tests/userService.test.js
import { expect, test, describe, beforeEach } from 'bun:test';
import { UserService } from '../src/services/userService';

describe('UserService', () => {
  let userService;

  beforeEach(() => {
    userService = new UserService();
  });

  test('should return a user by ID', () => {
    const user = userService.getUser('1');
    expect(user).toEqual({ id: '1', name: 'Alice' });
  });

  test('should return undefined for non-existent user', () => {
    const user = userService.getUser('99');
    expect(user).toBeUndefined();
  });

  // Example of snapshot testing
  test('should match user list snapshot', () => {
    expect(userService.getAllUsers()).toMatchSnapshot();
  });
});
```

### ❌ BAD: Skipping tests or using external test runners unnecessarily
Avoid relying solely on manual testing or introducing Jest/Vitest if `bun test` covers your needs.