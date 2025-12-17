---
description: Definitive guidelines for writing robust, maintainable, and performant Jest tests in JavaScript and TypeScript projects.
globs: **/*.{js,ts,jsx,tsx}
---
# jest Best Practices

This guide outlines our team's definitive Jest best practices. Adhere to these rules for consistent, reliable, and efficient testing.

## 1. Configuration (`jest.config.ts`)

Always use a `jest.config.ts` file for type safety and explicit configuration.

```typescript
// jest.config.ts
import type { Config } from 'jest';
import { defaults } from 'jest-config';

const config: Config = {
  // Use ts-jest for TypeScript files
  preset: 'ts-jest',
  // Automatically clear mock calls and instances between tests
  clearMocks: true,
  // Reset mocks between tests to prevent state leakage
  resetMocks: true,
  // Detailed test output
  verbose: true,
  // Collect coverage from specific files, exclude common boilerplate
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx', // Exclude entry points
    '!src/reportWebVitals.ts',
  ],
  // Ignore coverage for specific paths
  coveragePathIgnorePatterns: [
    '/node_modules/',
    'src/setupTests.ts',
  ],
  // Setup file for global test environment configurations (e.g., @testing-library/jest-dom)
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  // Map module paths for aliases or specific packages
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1', // Example: for absolute imports
  },
  // Test environment for browser-like DOM APIs
  testEnvironment: 'jsdom',
  // Match test files ending with .test.ts(x) or .spec.ts(x)
  testMatch: [...defaults.testMatch, '**/?(*.)+(spec|test).[tj]s?(x)'],
};

export default config;
```

## 2. Test Organization

Colocate test files with the code they test. This improves discoverability and maintainability.

❌ BAD: Separate `tests/` directory
```
src/
  components/
    Button.tsx
tests/
  components/
    Button.test.tsx
```

✅ GOOD: Colocated `__tests__/` or `.test.ts`
```
src/
  components/
    Button/
      index.tsx
      __tests__/
        Button.test.tsx
  utils/
    formatDate.ts
    formatDate.test.ts
```

## 3. Core Principles

### Test Behavior, Not Implementation

Focus on the public API and expected outcomes. Refactoring internal logic should not break tests.

❌ BAD: Testing internal state or private methods
```typescript
class UserService {
  private _users: User[] = []; // Internal state

  constructor() { /* ... */ }
  async fetchUsers() { /* ... */ }
}

test('UserService should have empty _users array initially', () => {
  const service = new UserService();
  expect(service._users).toHaveLength(0); // Accessing private property
});
```

✅ GOOD: Testing public behavior
```typescript
class UserService {
  private _users: User[] = [];

  constructor() { /* ... */ }
  async fetchUsers() { /* ... */ return this._users; }
}

test('UserService should return an empty array of users initially', async () => {
  const service = new UserService();
  const users = await service.fetchUsers();
  expect(users).toHaveLength(0); // Testing public method output
});
```

### Descriptive Test Names

Use clear, concise names that explain *what* is being tested and *what* the expected outcome is.

❌ BAD: Vague names
```typescript
test('adds', () => { /* ... */ });
it('should work', () => { /* ... */ });
```

✅ GOOD: Specific and expressive names
```typescript
describe('sum function', () => {
  test('should correctly add two positive numbers', () => { /* ... */ });
  test('should return zero when adding zero to a number', () => { /* ... */ });
});
```

### Keep Tests Isolated

Each test should run independently without relying on the state or side effects of other tests. Use `beforeEach` and `afterEach` for setup/teardown.

```typescript
let mockData: any[];

beforeEach(() => {
  mockData = [{ id: 1, name: 'Alice' }];
});

afterEach(() => {
  // Clean up any global mocks or side effects if not handled by config.resetMocks
  jest.clearAllMocks();
});

test('should add a new item to mockData', () => {
  mockData.push({ id: 2, name: 'Bob' });
  expect(mockData).toHaveLength(2);
});

test('should retrieve the initial mockData', () => {
  expect(mockData).toHaveLength(1); // mockData is reset by beforeEach
});
```

## 4. Asynchronous Testing

Always use `async/await` for asynchronous code.

❌ BAD: Using `.then()` or forgetting `await`
```typescript
test('should fetch user data', () => {
  fetchUser(1).then(user => {
    expect(user.id).toBe(1);
  });
});

test('should not fetch user data (missing await)', () => {
  const user = fetchUser(1); // Test finishes before promise resolves
  expect(user).toBeDefined(); // This will pass, but user is a Promise
});
```

✅ GOOD: Using `async/await`
```typescript
test('should fetch user data correctly', async () => {
  const user = await fetchUser(1);
  expect(user.id).toBe(1);
});

test('should handle fetch error', async () => {
  await expect(fetchUser(999)).rejects.toThrow('User not found');
});
```

## 5. Mocking Strategies

Mock external dependencies to isolate the unit under test and ensure deterministic, fast tests.

### Automatic Mocks (`jest.mock`)

Use `jest.mock` for entire modules.

```typescript
// api.ts
export const fetchData = async () => { /* ... */ };

// service.ts
import { fetchData } from './api';
export const getServiceData = async () => fetchData();

// service.test.ts
import { getServiceData } from './service';
import { fetchData } from './api';

jest.mock('./api'); // Mocks the entire module

test('getServiceData should call fetchData', async () => {
  (fetchData as jest.Mock).mockResolvedValue('mocked data'); // Cast to jest.Mock
  const data = await getServiceData();
  expect(data).toBe('mocked data');
  expect(fetchData).toHaveBeenCalledTimes(1);
});
```

### Spying on Functions (`jest.spyOn`)

Use `jest.spyOn` to observe calls to existing functions without replacing their original implementation (unless you explicitly mock it).

```typescript
// calculator.ts
export const add = (a: number, b: number) => a + b;
export const calculate = (a: number, b: number) => add(a, b) * 2;

// calculator.test.ts
import * as calculator from './calculator';

test('calculate should call add', () => {
  const addSpy = jest.spyOn(calculator, 'add');
  calculator.calculate(1, 2);
  expect(addSpy).toHaveBeenCalledWith(1, 2);
  addSpy.mockRestore(); // Clean up the spy
});

test('calculate should return mocked value if add is mocked', () => {
  const addSpy = jest.spyOn(calculator, 'add').mockReturnValue(100);
  const result = calculator.calculate(1, 2);
  expect(result).toBe(200); // 100 * 2
  addSpy.mockRestore();
});
```

### Fake Timers (`jest.useFakeTimers`)

Control `setTimeout`, `setInterval`, `Date` for predictable tests involving time.

```typescript
// timer.ts
export const runAfterDelay = (cb: () => void) => setTimeout(cb, 1000);

// timer.test.ts
import { runAfterDelay } from './timer';

jest.useFakeTimers();

test('runAfterDelay calls the callback after 1 second', () => {
  const callback = jest.fn();
  runAfterDelay(callback);

  expect(callback).not.toHaveBeenCalled();
  jest.advanceTimersByTime(500); // Advance by 500ms
  expect(callback).not.toHaveBeenCalled();
  jest.advanceTimersByTime(500); // Advance by another 500ms (total 1000ms)
  expect(callback).toHaveBeenCalledTimes(1);
});

jest.useRealTimers(); // Restore real timers after tests
```

### Avoid Over-Mocking

Mock only what's necessary. Over-mocking can lead to brittle tests that don't reflect real-world behavior.

## 6. UI Component Testing (with React Testing Library)

For React components, use React Testing Library with `@testing-library/jest-dom` matchers.

```typescript
// src/setupTests.ts (referenced in jest.config.ts)
import '@testing-library/jest-dom';

// src/components/Button/index.tsx
type ButtonProps = { onClick: () => void; children: React.ReactNode; };
export const Button = ({ onClick, children }: ButtonProps) => (
  <button onClick={onClick}>{children}</button>
);

// src/components/Button/__tests__/Button.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from '../';

test('Button renders children and handles click', async () => {
  const user = userEvent.setup();
  const handleClick = jest.fn();
  render(<Button onClick={handleClick}>Click Me</Button>);

  const button = screen.getByRole('button', { name: /click me/i });
  expect(button).toBeInTheDocument();

  await user.click(button);
  expect(handleClick).toHaveBeenCalledTimes(1);
});
```

### Avoid `screen.debug()` in Committed Code

`screen.debug()` is for debugging during development, not for production tests.

❌ BAD:
```typescript
test('renders component', () => {
  render(<MyComponent />);
  screen.debug(); // Should not be committed
  expect(screen.getByText('Hello')).toBeInTheDocument();
});
```

✅ GOOD:
```typescript
test('renders component', () => {
  render(<MyComponent />);
  expect(screen.getByText('Hello')).toBeInTheDocument();
});
```

## 7. Snapshot Testing

Use snapshot tests for UI components or schema stability, but review them carefully.

```typescript
// src/components/Card/__tests__/Card.test.tsx
import { render } from '@testing-library/react';
import { Card } from '../';

test('Card component matches snapshot', () => {
  const { asFragment } = render(<Card title="Test" content="Hello World" />);
  expect(asFragment()).toMatchSnapshot();
});

// For GraphQL schema stability (requires jest-serializer-graphql-schema)
import { buildSchema } from 'graphql';
import { lexicographicSortSchema } from 'graphql/utilities';

test('GraphQL schema is stable', () => {
  const schema = buildSchema(/* ... */);
  expect(lexicographicSortSchema(schema)).toMatchSnapshot();
});
```

## 8. Avoid Focused Tests in Committed Code

Never commit `.only` or `.skip` to the codebase. These are for local development only.

❌ BAD:
```typescript
describe.only('My Feature', () => { /* ... */ });
test.skip('should not run this test', () => { /* ... */ });
```

✅ GOOD:
```typescript
describe('My Feature', () => { /* ... */ });
test('should run this test', () => { /* ... */ });
```