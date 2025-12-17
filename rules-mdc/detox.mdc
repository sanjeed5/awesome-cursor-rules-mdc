---
description: This guide provides definitive, opinionated best practices for writing reliable, maintainable, and performant end-to-end tests with Detox in React Native applications.
globs: **/*.{js,jsx}
---
# detox Best Practices

Detox is the definitive choice for E2E testing in React Native. These rules ensure your tests are fast, stable, and easy to maintain, leveraging modern JavaScript and Detox's gray-box capabilities.

## 1. Test Organization with Page Object Model (POM)

Always structure your tests using the Page Object Model. This abstracts UI interactions and selectors, making tests readable and resilient to UI changes.

❌ **BAD: Direct selectors and repeated logic**
```javascript
// login.e2e.js
describe('Login Flow', () => {
  it('should log in successfully', async () => {
    await element(by.id('emailInput')).typeText('test@example.com');
    await element(by.id('passwordInput')).typeText('password123');
    await element(by.id('loginButton')).tap();
    await expect(element(by.id('homeScreen'))).toBeVisible();
  });
});
```

✅ **GOOD: Page Object Model**
```javascript
// pages/LoginPage.js
class LoginPage {
  constructor() {
    this.emailInput = element(by.id('emailInput'));
    this.passwordInput = element(by.id('passwordInput'));
    this.loginButton = element(by.id('loginButton'));
  }

  async login(email, password) {
    await this.emailInput.typeText(email);
    await this.passwordInput.typeText(password);
    await this.loginButton.tap();
  }

  async isVisible() {
    await expect(this.emailInput).toBeVisible();
  }
}
export default new LoginPage();

// tests/login.e2e.js
import LoginPage from '../pages/LoginPage';
import HomeScreen from '../pages/HomeScreen';

describe('Login Flow', () => {
  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('should log in successfully', async () => {
    await LoginPage.login('test@example.com', 'password123');
    await HomeScreen.isVisible();
  });
});
```

## 2. Stable Selectors are Paramount

Prioritize `by.id` for all interactive elements. Fallback to `by.text` only when `by.id` is not feasible (e.g., dynamic content). Avoid fragile selectors like `by.type` or `by.label` if a more stable alternative exists.

❌ **BAD: Fragile selector**
```javascript
// Relies on element type, which can change
await element(by.type('RCTTextView').withAncestor(by.id('welcomeMessage'))).toBeVisible();
```

✅ **GOOD: Stable selector with `testID`**
```javascript
// In your React Native component: <Text testID="welcomeMessageText">Welcome!</Text>
await expect(element(by.id('welcomeMessageText'))).toBeVisible();
```

## 3. Explicit Waits and Assertions

Never use arbitrary `sleep()` calls. Detox's gray-box synchronization handles most async operations, but for complex UI states or specific data loads, use `waitFor` with `expect` conditions.

❌ **BAD: Arbitrary `sleep`**
```javascript
await element(by.id('submitButton')).tap();
await sleep(2000); // 🚨 Flaky!
await expect(element(by.id('successMessage'))).toBeVisible();
```

✅ **GOOD: Explicit `waitFor` and `expect`**
```javascript
await element(by.id('submitButton')).tap();
await waitFor(element(by.id('successMessage')))
  .toBeVisible()
  .withTimeout(5000); // Max wait time
```

## 4. Reset App State Before Each Test

Ensure test isolation by resetting the app state before every `it` block. `device.reloadReactNative()` is the standard for a quick, clean slate. For a full app re-installation, use `device.launchApp({ delete: true })` in `beforeAll`.

```javascript
describe('User Profile', () => {
  beforeEach(async () => {
    // Resets React Native state, including navigation and Redux store
    await device.reloadReactNative();
    // Perform login or other common setup here
    await LoginPage.login('existing@example.com', 'password123');
  });

  it('should display user details', async () => {
    // ... test logic
  });
});
```

## 5. Embrace `async/await` Everywhere

All Detox interactions and assertions are asynchronous. Use `async/await` consistently for clear, sequential test logic.

❌ **BAD: Mixing Promises and `async/await` inconsistently**
```javascript
it('should navigate', () => { // Missing async
  element(by.id('navButton')).tap().then(() => {
    return expect(element(by.id('nextScreen'))).toBeVisible();
  });
});
```

✅ **GOOD: Consistent `async/await`**
```javascript
it('should navigate', async () => {
  await element(by.id('navButton')).tap();
  await expect(element(by.id('nextScreen'))).toBeVisible();
});
```

## 6. Mocking Strategies for External Dependencies

For true E2E tests, mock external APIs or services to control test data and eliminate flakiness from network instability. Use tools like `msw` (Mock Service Worker) for client-side mocking or `nock` for Node.js-based API mocks.

```javascript
// setup.js (run before tests)
import { setupServer } from 'msw/node';
import { rest } from 'msw';

const server = setupServer(
  rest.post('https://api.example.com/login', (req, res, ctx) => {
    return res(ctx.json({ token: 'mock-token', user: { id: '123' } }));
  }),
  rest.get('https://api.example.com/profile', (req, res, ctx) => {
    return res(ctx.json({ name: 'Test User', email: 'test@example.com' }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// tests/profile.e2e.js
describe('User Profile', () => {
  beforeEach(async () => {
    await device.reloadReactNative();
    // Login will now use the mocked API response
    await LoginPage.login('mock@example.com', 'password');
  });

  it('should display mocked user profile data', async () => {
    await expect(element(by.id('userName'))).toHaveText('Test User');
    await expect(element(by.id('userEmail'))).toHaveText('test@example.com');
  });
});
```