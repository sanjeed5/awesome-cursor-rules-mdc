---
description: This guide provides opinionated, actionable best practices for writing fast, reliable, and maintainable unit and integration tests using Vitest in modern JavaScript and TypeScript projects.
globs: **/*.{js,ts,jsx,tsx}
---
# Vitest Best Practices

Vitest is the definitive testing framework for our Vite-powered projects. It offers a fast, Jest-compatible API with deep integration into the Vite ecosystem. Adhering to these guidelines ensures our tests are robust, performant, and easy to maintain.

## 1. Code Organization & Naming

**Always co-locate test files with their source.** This improves discoverability and ensures tests are updated alongside their implementation.

*   **File Naming**: Use `*.test.{ts,tsx,js,jsx}`.
*   **Location**: Place test files directly next to the component or module they test.

❌ BAD:
```
// src/components/Button/Button.tsx
// tests/components/Button.test.tsx
```

✅ GOOD:
```typescript
// src/components/Button/Button.tsx
// src/components/Button/Button.test.tsx
```

## 2. Test Structure & Isolation

**Organize tests logically using `describe` and `it` (or `test`) blocks.** Ensure each test is isolated and deterministic.

*   **`describe`**: Group related tests into suites.
*   **`it` / `test`**: Define individual test cases. Prefer `it` for consistency with Jest.
*   **Hooks (`beforeEach`, `afterEach`)**: Use these for setup and teardown to ensure test isolation.

❌ BAD: (Shared state, no cleanup)
```typescript
let user;
test('creates user', () => {
  user = createUser();
  expect(user).toBeDefined();
});
test('updates user', () => { // Depends on previous test
  user.name = 'New Name';
  expect(user.name).toBe('New Name');
});
```

✅ GOOD: (Isolated tests with hooks)
```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createUser, deleteUser } from './user-service';

describe('User Service', () => {
  let user;

  beforeEach(() => {
    user = createUser(); // Create a fresh user for each test
  });

  afterEach(() => {
    deleteUser(user.id); // Clean up after each test
  });

  it('should create a user', () => {
    expect(user).toBeDefined();
    expect(user.id).toBeTypeOf('string');
  });

  it('should update a user', () => {
    user.name = 'Jane Doe';
    expect(user.name).toBe('Jane Doe');
  });
});
```

## 3. Asynchronous Testing with `vi.waitFor`

**Always use `vi.waitFor` for polling conditions in asynchronous tests.** Avoid arbitrary `setTimeout` calls or manual polling loops. `vi.waitFor` is designed for reliable synchronization.

❌ BAD: (Flaky, relies on arbitrary timeout)
```typescript
test('data loads after delay', async () => {
  let data = null;
  fetchData().then(res => (data = res));
  await new Promise(resolve => setTimeout(resolve, 100)); // Arbitrary wait
  expect(data).toEqual('some data');
});
```

✅ GOOD: (Reliable polling with `vi.waitFor`)
```typescript
import { it, expect, vi } from 'vitest';
import { fetchData } from './api'; // Assume fetchData returns a Promise

it('should load data after delay', async () => {
  let data = null;
  fetchData().then(res => (data = res));

  // Polls until data is not null, with a 2-second timeout
  await vi.waitFor(() => expect(data).not.toBeNull(), { timeout: 2000 });

  expect(data).toEqual('some data');
});
```

## 4. Mocking Strategies

**Leverage Vitest's `vi` API for all mocking.** This provides Jest-compatible syntax and seamless integration. Always clean up mocks after each test.

*   **`vi.fn()`**: Mock individual functions.
*   **`vi.spyOn()`**: Spy on existing object methods.
*   **`vi.mock()`**: Mock entire modules.

### Function Mocking

❌ BAD: (Manual mock, no easy reset)
```typescript
const originalFetch = global.fetch;
global.fetch = () => Promise.resolve({ json: () => ({ id: 1 }) });
// ... test ...
global.fetch = originalFetch; // Easy to forget cleanup
```

✅ GOOD: (Using `vi.fn` with `afterEach` cleanup)
```typescript
import { it, expect, vi, afterEach } from 'vitest';
import { getUser } from './user-api';

// Mock the module containing fetchUser
vi.mock('./user-api', async (importOriginal) => {
  const mod = await importOriginal();
  return {
    ...mod,
    fetchUser: vi.fn(), // Mock specific function within the module
  };
});

// Import the mocked function after vi.mock
import { fetchUser } from './user-api';

afterEach(() => {
  vi.clearAllMocks(); // Clear mock calls after each test to prevent state leakage
});

it('should fetch user data', async () => {
  fetchUser.mockResolvedValueOnce({ id: 1, name: 'Test User' });
  const user = await getUser(1);
  expect(fetchUser).toHaveBeenCalledWith(1);
  expect(user.name).toBe('Test User');
});
```

### Module Mocking

**Mock modules at the top of the file.** This ensures the mock is applied before the module under test imports it.

✅ GOOD: (Module mock before imports)
```typescript
import { vi, it, expect } from 'vitest';

// Mock the entire 'lodash' module to control its behavior
vi.mock('lodash', () => ({
  debounce: vi.fn((fn) => fn), // Mock debounce to execute immediately
}));

import { debounce } from 'lodash'; // Import the mocked debounce
import { saveInput } from './input-handler'; // Module using debounce

it('should call save function without debounce delay', () => {
  saveInput('test');
  expect(debounce).toHaveBeenCalledOnce();
});
```

## 5. DOM Environment & Component Testing

**Use `happy-dom` for lightweight DOM environments.** It's generally faster and sufficient for most component tests. Switch to `jsdom` only if specific browser APIs are missing in `happy-dom`.

*   Configure in `vite.config.ts` or `vitest.config.ts`.

✅ GOOD: (Configuring `happy-dom`)
```typescript
// vite.config.ts or vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'happy-dom', // Use happy-dom for faster DOM mocking
    globals: true, // Auto-import test APIs globally (e.g., describe, it, expect)
  },
});
```

## 6. Performance & Concurrent Tests

**Utilize `.concurrent` for tests that can run in parallel.** This significantly speeds up test suites where tests are independent.

*   Use `it.concurrent` for individual tests.
*   Use `describe.concurrent` for entire suites.
*   **Important**: When using `.concurrent`, always destructure `expect` from the test context to avoid issues with snapshot and assertion tracking.

❌ BAD: (Sequential tests, slow)
```typescript
describe('My Feature', () => {
  it('test A', async () => { /* ... */ });
  it('test B', async () => { /* ... */ });
});
```

✅ GOOD: (Concurrent tests, faster)
```typescript
import { describe, it } from 'vitest';

describe.concurrent('My Feature', () => {
  it('test A', async ({ expect }) => { // Destructure expect for concurrent tests
    expect(1).toBe(1);
  });

  it.concurrent('test B', async ({ expect }) => { // Destructure expect
    expect(2).toBe(2);
  });
});
```

## 7. Code Coverage

**Enable V8-based code coverage.** It offers near-zero overhead and integrates seamlessly.

*   Add `coverage` configuration to `vite.config.ts` or `vitest.config.ts`.
*   Run with `vitest run --coverage`.

✅ GOOD: (V8 coverage configuration)
```typescript
// vite.config.ts or vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'happy-dom',
    globals: true,
    coverage: {
      provider: 'v8', // Use V8 for native, fast coverage
      reporter: ['text', 'json', 'html'], // Output formats for reports
      exclude: ['node_modules/', 'dist/', '.eslintrc.cjs'], // Exclude common directories from coverage
    },
  },
});
```