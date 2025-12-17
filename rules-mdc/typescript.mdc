---
description: Enforce modern TypeScript best practices for robust, type-safe JavaScript applications, focusing on strictness, clear type definitions, and runtime validation.
globs: **/*.{js,jsx}
---
# TypeScript Best Practices (for JS/JSX Type-Checking)

This guide outlines essential TypeScript best practices for teams working with JavaScript or JSX files that are type-checked by TypeScript (e.g., via `tsconfig.json` with `allowJs` and/or JSDoc annotations). While many examples use native TypeScript syntax for clarity and conciseness, the underlying principles and type-safety benefits apply directly to your `.js`/`.jsx` codebase.

## 1. Enable Strict Mode in `tsconfig.json`

This is the single most impactful change you can make. `strict: true` enables a suite of crucial checks that catch the vast majority of common type-related bugs at compile time.

**Action**: Ensure your `tsconfig.json` includes:

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true, // Enables all strict type-checking options
    "noImplicitAny": true, // Catches untyped variables/parameters
    "strictNullChecks": true, // Prevents `null`/`undefined` access without checks
    "strictPropertyInitialization": true, // Ensures class properties are initialized
    "allowJs": true, // Crucial for type-checking .js/.jsx files
    "checkJs": true, // Enables type-checking in .js/.jsx files
    // ... other options
  },
  "include": ["**/*.js", "**/*.jsx"] // Ensure your JS/JSX files are included
}
```

## 2. Define Clear Type Contracts

Use interfaces and type aliases to describe the shape of your data, especially for API payloads, component props, and complex objects. For `.js`/`.jsx` files, leverage JSDoc to apply these types.

### ✅ GOOD: Interfaces for Object Shapes & Classes

Interfaces are ideal for defining the shape of objects and for implementing explicit contracts when working with classes. Define these in `.d.ts` or `.ts` files, then reference them in JSDoc.

```typescript
// types.d.ts or types.ts
interface UserProfile {
  id: string;
  name: string;
  email: string;
  age?: number; // Optional property
}

interface Point {
  readonly x: number;
  readonly y: number;
}
```

```javascript
// user-service.js
/**
 * @typedef {import('./types').UserProfile} UserProfile
 */

/**
 * @param {UserProfile} user
 * @returns {string}
 */
export function greetUser(user) {
  return `Hello, ${user.name}!`;
}
```

### ✅ GOOD: Type Aliases for Complex Types & Unions

Type aliases are powerful for defining unions, intersections, primitive aliases, and tuples. Define these in `.d.ts` or `.ts` files, then reference them in JSDoc.

```typescript
// types.d.ts or types.ts
type ID = string | number;
type UserRole = "admin" | "user" | "guest"; // Strict literal type
type Coords = [number, number];
```

```javascript
// auth-service.js
/**
 * @typedef {import('./types').ID} ID
 * @typedef {import('./types').UserRole} UserRole
 */

/**
 * @param {ID} userId
 * @param {UserRole} role
 * @returns {void}
 */
export function assignRole(userId, role) {
  console.log(`User ${userId} assigned role: ${role}`);
}
```

## 3. Avoid `any` and Prefer `unknown` for Untyped Data

`any` completely bypasses TypeScript's checks, reintroducing JavaScript's runtime errors. `unknown` is a safer alternative that forces you to narrow the type before use.

### ❌ BAD: Using `any`

```javascript
// data-processor.js
/**
 * @param {any} data // Bypasses all type checks
 */
export function processDataBad(data) {
  // No type safety here, data.foo might not exist
  console.log(data.foo.bar);
}
```

### ✅ GOOD: Using `unknown` (and Type Guards)

When dealing with data from external sources (e.g., API responses), use `unknown` and then narrow its type using runtime checks.

```javascript
// data-processor.js
/**
 * @param {unknown} data // Forces runtime checks
 */
export function processDataGood(data) {
  if (typeof data === 'object' && data !== null && 'foo' in data) {
    /** @type {{ foo: { bar: string } }} */
    const typedData = data; // Type assertion after narrowing
    console.log(typedData.foo.bar);
  } else {
    console.error('Invalid data structure');
  }
}
```

## 4. Implement Robust Runtime Type Validation (Type Guards)

TypeScript's compile-time checks are erased at runtime. For data from external sources (APIs, user input), you *must* perform runtime validation to prevent crashes.

### ✅ GOOD: `typeof` for Primitives

```javascript
// utils.js
/**
 * @param {unknown} value
 * @returns {boolean}
 */
export function isString(value) {
  return typeof value === 'string';
}

/**
 * @param {unknown} value
 * @returns {boolean}
 */
export function isNumber(value) {
  return typeof value === 'number' && !isNaN(value);
}
```

### ✅ GOOD: Custom Type Guards for Complex Objects

For interfaces or complex object shapes, create custom functions that perform runtime checks and act as type predicates.

```typescript
// types.d.ts or types.ts
interface Product {
  id: string;
  name: string;
  price: number;
}

// Custom type guard (can be in a .ts file, or its logic used directly in .js)
/**
 * @param {unknown} obj
 * @returns {obj is Product}
 */
function isProduct(obj) {
  return (
    typeof obj === 'object' && obj !== null &&
    'id' in obj && typeof obj.id === 'string' &&
    'name' in obj && typeof obj.name === 'string' &&
    'price' in obj && typeof obj.price === 'number'
  );
}
```

```javascript
// api-client.js
/**
 * @typedef {import('./types').Product} Product
 * @type {typeof import('./types').isProduct}
 */
import { isProduct } from './types'; // Assuming isProduct is exported from types.ts/d.ts

/**
 * @param {unknown} apiResponse
 */
export function handleApiResponse(apiResponse) {
  if (isProduct(apiResponse)) {
    // TypeScript now knows apiResponse is Product here
    console.log(`Fetched product: ${apiResponse.name} at $${apiResponse.price}`);
  } else {
    console.error('API response is not a valid Product:', apiResponse);
  }
}
```

## 5. Prefer Union Types over Traditional Enums

For simple sets of related constants, strict literal union types (`"admin" | "user"`) are generally preferred over TypeScript's `enum` keyword because they offer better type safety and simpler runtime representation. If you must use enums, prefer `const enum` or string enums.

### ❌ BAD: Numeric Enums

```typescript
// In a .ts file or JSDoc
enum UserStatus {
  Active, // 0
  Inactive, // 1
  Pending // 2
}
// Problem: Can be assigned any number at runtime, e.g., UserStatus.Active = 99
```

### ✅ GOOD: String Literal Unions or `const enum`

```typescript
// types.d.ts or types.ts
/** @typedef {'active' | 'inactive' | 'pending'} UserStatus */
// Or using native TS:
type UserStatus = 'active' | 'inactive' | 'pending';

// If an enum is truly necessary, use a const string enum for compile-time safety and runtime efficiency:
const enum UserRole {
  Admin = "admin",
  User = "user",
  Guest = "guest",
}
```

```javascript
// user-management.js
/**
 * @typedef {import('./types').UserStatus} UserStatus
 */

/**
 * @param {UserStatus} status
 */
export function updateUserStatus(status) {
  console.log(`User status updated to: ${status}`);
}

updateUserStatus('active');
// updateUserStatus('invalid'); // Type error
```

## 6. Use Generics for Reusable Components/Functions

Generics allow you to write flexible and reusable code that works with a variety of types while maintaining type safety.

```javascript
// utils.js
/**
 * @template T
 * @param {T} arg
 * @returns {T}
 */
export function identity(arg) {
  return arg;
}

const num = identity(123); // num is inferred as number
const str = identity("hello"); // str is inferred as string

// With constraints (requires native TS syntax or complex JSDoc)
// For complex generics with constraints, consider writing them in a .ts file
// and importing their JSDoc type definitions.
```

## 7. Enforce Consistent Code Organization

Maintainable codebases rely on clear structure.

### ✅ GOOD: Named Exports (No Default Exports)

Named exports promote explicit imports and make refactoring easier. Avoid default exports entirely.

```javascript
// user-service.js
// ❌ BAD: default export
// export default class UserService { /* ... */ }

// ✅ GOOD: named export
export class UserService { /* ... */ }
export const DEFAULT_USER = { /* ... */ };
```

### ✅ GOOD: Organized Imports

Group imports by type (e.g., library, absolute path, relative path) and sort them alphabetically. Use path aliases for cleaner imports from deeply nested modules.

```javascript
// some-component.jsx
import React from 'react'; // Library imports
import { useSelector } from 'react-redux';

import { API_URL } from 'config/constants'; // Absolute imports (e.g., using path aliases)
import { selectUser } from 'store/selectors';

import { Button } from './components/Button'; // Relative imports
import { formatCurrency } from '../utils/formatters';
```

##