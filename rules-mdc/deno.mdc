---
description: This guide defines definitive Deno best practices for our team, focusing on secure, maintainable, and idiomatic code using Deno's built-in tooling and modern web standards.
globs: **/*.{js,jsx,ts,tsx}
---
# Deno Best Practices

Deno is our runtime of choice for its security, built-in tooling, and first-class TypeScript support. This guide outlines the definitive best practices for writing Deno applications, ensuring consistency, performance, and maintainability across our projects.

## 1. Project Configuration: `deno.json` is King

Always use `deno.json` (or `deno.jsonc`) as the single source of truth for project configuration. This file unifies TypeScript compiler options, Deno-specific flags, tasks, linting, and formatting. Avoid `tsconfig.json` for Deno-first projects.

### ✅ GOOD: Centralized `deno.json`

```json title="deno.json"
{
  "compilerOptions": {
    "target": "es2022",
    "lib": ["deno.window", "deno.unstable"],
    "strict": true,
    "checkJs": true,
    "jsx": "react-jsx"
  },
  "lint": {
    "rules": {
      "tags": ["recommended"]
    }
  },
  "fmt": {
    "options": {
      "useTabs": false,
      "lineWidth": 100,
      "indentWidth": 2,
      "singleQuote": true
    }
  },
  "imports": {
    "std/": "https://deno.land/std@0.224.0/",
    "oak": "jsr:@oak/oak@16"
  },
  "tasks": {
    "dev": "deno run --allow-net --watch main.ts",
    "test": "deno test --allow-read",
    "start": "deno run --allow-net main.ts"
  }
}
```

### ❌ BAD: Fragmented Configuration

```json title="tsconfig.json"
{
  "compilerOptions": {
    "target": "es2022",
    "strict": true
  }
}
```

```json title="package.json"
{
  "scripts": {
    "dev": "deno run --allow-net --watch main.ts"
  }
}
```

## 2. Module Organization: ES Modules & Explicit Imports

Deno embraces ES modules and URL-based dependencies. Always use explicit file extensions and prefer JSR or `deno.land/x` for external modules.

### 2.1. Explicit URL Imports

Import directly from URLs or use `deno.json`'s `imports` map for cleaner paths.

### ✅ GOOD: URL Imports with Import Map

```typescript title="main.ts"
import { serve } from "std/http/server.ts"; // Uses import map
import { Application } from "oak"; // Uses import map

const app = new Application();
app.listen({ port: 8000 });
console.log("Server running on http://localhost:8000");
```

### ❌ BAD: Missing Extensions or Node.js-style Imports

```typescript title="main.ts"
import { serve } from "https://deno.land/std/http/server"; // Missing .ts extension
import { Application } from "@oak/oak"; // Bare specifier without import map or npm:
```

### 2.2. Entry Point: `mod.ts`

For libraries or main application entry points, name the primary module `mod.ts`.

### ✅ GOOD: `mod.ts` as Entry

```typescript title="mod.ts"
export * from "./src/utils.ts";
export * from "./src/server.ts";
```

### ❌ BAD: `index.ts` for Deno-first projects

```typescript title="index.ts"
export * from "./src/utils.ts";
```

## 3. Security: Explicit Permissions

Deno is secure by default. Always declare necessary permissions explicitly using `--allow-*` flags. Grant the *minimal* required permissions.

### ✅ GOOD: Minimal Permissions

```bash
deno run --allow-net=:8080 --allow-read=. main.ts
```

### ❌ BAD: Overly Permissive or Missing Permissions

```bash
deno run --allow-all main.ts # Too broad
deno run main.ts # Will fail if network/file access is needed
```

## 4. Standard Library Preference

Leverage Deno's built-in standard library (`deno_std`) for common tasks like HTTP, file system, and testing. This reduces external dependencies and maintains consistency.

### ✅ GOOD: Using `deno_std`

```typescript title="file_reader.ts"
import { readAll } from "std/io/read_all.ts";

async function readFileContents(filePath: string): Promise<string> {
  const file = await Deno.open(filePath);
  const contents = await readAll(file);
  Deno.close(file.rid);
  return new TextDecoder().decode(contents);
}
```

### ❌ BAD: Relying on Third-Party for Basic Functionality

```typescript title="file_reader.ts"
// Assuming a third-party 'fs-extra' equivalent exists
import { readFile } from "https://deno.land/x/fs_extra/mod.ts";

async function readFileContents(filePath: string): Promise<string> {
  return await readFile(filePath, { encoding: "utf8" });
}
```

## 5. Type Safety: TypeScript First

Write code in TypeScript (`.ts` or `.tsx`). For JavaScript files, enable type checking with `@ts-check` and use JSDoc for type annotations. Ensure `strict: true` is set in `deno.json`.

### ✅ GOOD: TypeScript with Strict Checks

```typescript title="user.ts"
interface User {
  id: string;
  name: string;
  email?: string; // Optional property
}

function getUserName(user: User): string {
  return user.name;
}
```

### ✅ GOOD: JSDoc for JavaScript

```javascript title="utils.js"
// @ts-check

/**
 * @typedef {object} Product
 * @property {string} id
 * @property {string} name
 * @property {number} price
 */

/**
 * Calculates the total price of products.
 * @param {Product[]} products - List of products.
 * @returns {number} The total price.
 */
function calculateTotalPrice(products) {
  return products.reduce((total, p) => total + p.price, 0);
}
```

### ❌ BAD: Untyped JavaScript or Relaxed TypeScript

```javascript title="user.js"
// No @ts-check, no JSDoc
function getUserName(user) {
  return user.name;
}
```

## 6. Asynchronous Operations: Top-Level `await`

Embrace top-level `await` for cleaner asynchronous code, especially in entry files. Always `await` promises explicitly.

### ✅ GOOD: Top-Level `await`

```typescript title="main.ts"
import { serve } from "std/http/server.ts";

const handler = (req: Request): Response => {
  return new Response("Hello from Deno!");
};

console.log("Server starting...");
await serve(handler, { port: 8000 });
console.log("Server stopped.");
```

### ❌ BAD: IIFEs or Unhandled Promises

```typescript title="main.ts"
import { serve } from "std/http/server.ts";

const handler = (req: Request): Response => {
  return new Response("Hello from Deno!");
};

(async () => { // Unnecessary IIFE
  console.log("Server starting...");
  serve(handler, { port: 8000 }); // Unawaited promise
  console.log("Server stopped."); // Will log immediately
})();
```

## 7. Testing: Built-in `deno test`

Use Deno's built-in test runner (`deno test`) and `deno_std/assert` for writing unit and integration tests. Organize tests in `_test.ts` or `_test.js` files.

### ✅ GOOD: Idiomatic Deno Testing

```typescript title="math_test.ts"
import { assertEquals } from "std/assert/assert_equals.ts";
import { add } from "./math.ts";

Deno.test("add function sums two numbers correctly", () => {
  assertEquals(add(1, 2), 3);
  assertEquals(add(-1, 1), 0);
  assertEquals(add(0, 0), 0);
});
```

```typescript title="math.ts"
export function add(a: number, b: number): number {
  return a + b;
}
```

### ❌ BAD: Third-Party Test Runners or Inconsistent Naming

```typescript title="test.ts"
// Using a third-party test runner not integrated with Deno's ecosystem
import { describe, it, expect } from "https://deno.land/x/some_test_lib/mod.ts";
import { add } from "./math.ts";

describe("add", () => {
  it("should sum numbers", () => {
    expect(add(1, 2)).toBe(3);
  });
});
```