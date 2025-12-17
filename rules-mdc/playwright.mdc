---
description: Definitive guidelines for writing robust, maintainable, and high-quality end-to-end tests with Playwright in TypeScript.
globs: **/*.{js,ts}
---
# Playwright Best Practices

Playwright is the gold standard for reliable E2E testing. These rules ensure your tests are fast, stable, and easy to maintain, aligning with modern 2025 development standards for reliability, quality, and structure.

## 1. Always Use `@playwright/test`

Leverage the official test runner for built-in fixtures, isolation, and web-first assertions. Avoid the low-level `playwright` library for E2E tests.

❌ BAD: Using `playwright` directly
```typescript
import { chromium } from 'playwright';
// ... manual browser/context setup and teardown
const browser = await chromium.launch();
const page = await browser.newPage();
// ...
await browser.close();
```

✅ GOOD: Using `@playwright/test`
```typescript
import { test, expect } from '@playwright/test';
test('should navigate to home', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Home/);
});
```

## 2. Prioritize Robust Locators

Use Playwright's built-in Locators API, favoring user-facing attributes over brittle CSS selectors. This drastically improves test stability.

❌ BAD: Fragile, implementation-dependent selectors
```typescript
await page.locator('div.container > ul > li:nth-child(2) > button').click();
```

✅ GOOD: Semantic, user-facing locators
```typescript
await page.getByRole('button', { name: 'Add to Cart' }).click();
await page.getByLabel('Username').fill('testuser');
await page.getByTestId('product-item-123').click();
```

## 3. Embrace Web-First Assertions

Playwright's `expect` assertions automatically retry until conditions are met, eliminating manual waits and flakiness. Never use `page.waitForTimeout()`.

❌ BAD: Manual, flaky waits and generic assertions
```typescript
await page.waitForTimeout(2000); // 🚨 Flaky!
const title = await page.title();
assert.equal(title, 'My Page'); // 🚨 Not web-first
```

✅ GOOD: Reliable, auto-retrying assertions
```typescript
await expect(page).toHaveTitle(/My Page/);
await expect(page.getByText('Welcome')).toBeVisible();
await expect(page.getByRole('checkbox')).toBeChecked();
```

## 4. Implement the Page Object Model (POM)

Encapsulate selectors and actions within dedicated classes. This improves readability, reusability, and maintainability.

❌ BAD: Repeated selectors and logic across tests
```typescript
// test-login.spec.ts
await page.getByLabel('Username').fill('user');
await page.getByLabel('Password').fill('pass');
await page.getByRole('button', { name: 'Login' }).click();

// test-profile.spec.ts
// ... same login steps repeated ...
```

✅ GOOD: Centralized Page Object
```typescript
// pages/LoginPage.ts
import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly usernameInput: Locator;
  readonly passwordInput: Locator;
  readonly loginButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.usernameInput = page.getByLabel('Username');
    this.passwordInput = page.getByLabel('Password');
    this.loginButton = page.getByRole('button', { name: 'Login' });
  }

  async navigate() {
    await this.page.goto('/login');
  }

  async login(username: string, password: string) {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.loginButton.click();
  }
}

// tests/login.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

test('should successfully log in', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.navigate();
  await loginPage.login('testuser', 'password');
  await expect(page).toHaveURL(/dashboard/);
});
```

## 5. Optimize Performance with Auth State & Route Blocking

Reduce test execution time by reusing authenticated sessions and blocking unnecessary network requests.

❌ BAD: Logging in for every test and loading all assets
```typescript
test('view profile', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Username').fill('user');
  await page.getByLabel('Password').fill('pass');
  await page.getByRole('button', { name: 'Login' }).click();
  await page.goto('/profile'); // Loads all images, analytics, etc.
});
```

✅ GOOD: Reusing auth state and blocking requests
```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';
export default defineConfig({
  use: {
    storageState: 'playwright-auth.json', // Path to save/load auth state
  },
});

// global-setup.ts (run once before all tests)
import { chromium, expect } from '@playwright/test';
export default async function globalSetup() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('/login');
  await page.getByLabel('Username').fill('testuser');
  await page.getByLabel('Password').fill('password');
  await page.getByRole('button', { name: 'Login' }).click();
  await expect(page).toHaveURL(/dashboard/);
  await page.context().storageState({ path: 'playwright-auth.json' });
  await browser.close();
}

// tests/profile.spec.ts
import { test, expect } from '@playwright/test';
test('should display user profile', async ({ page, context }) => {
  // Block unnecessary resources for faster tests
  await context.route('**/*.{png,jpg,jpeg,gif,webp,svg,css}', route => route.abort());
  await page.goto('/profile'); // Automatically uses saved auth state
  await expect(page.getByText('Welcome, testuser!')).toBeVisible();
});
```

## 6. Mock APIs for Deterministic Tests

Isolate your UI tests from backend flakiness by intercepting and mocking API responses.

❌ BAD: Relying on a live, potentially unstable backend
```typescript
test('display products', async ({ page }) => {
  await page.goto('/products'); // Fetches from real API
  await expect(page.getByText('Product A')).toBeVisible();
});
```

✅ GOOD: Mocking API responses
```typescript
test('display mocked products', async ({ page }) => {
  await page.route('**/api/products', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([{ id: 1, name: 'Mock Product' }]),
    });
  });
  await page.goto('/products');
  await expect(page.getByText('Mock Product')).toBeVisible();
});
```

## 7. Leverage CI/CD Features for Debugging

Configure tracing, screenshots, and video recording in your `playwright.config.ts` to instantly diagnose failures in CI.

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';
export default defineConfig({
  reporter: [['html'], ['list']],
  use: {
    trace: 'on-first-retry', // Record trace only on first retry
    screenshot: 'on',       // Always take a screenshot on failure
    video: 'on-first-retry',// Record video on first retry
  },
});
```

## 8. Maintain Code Quality with Linters & Formatters

Integrate ESLint (with Playwright plugin) and Prettier into your workflow via pre-commit hooks to enforce consistent code style and catch errors early.

```json
// package.json (example scripts)
{
  "scripts": {
    "lint": "eslint . --ext .ts",
    "format": "prettier --write .",
    "test:e2e": "playwright test"
  },
  "devDependencies": {
    "@playwright/test": "^1.x.x",
    "@typescript-eslint/eslint-plugin": "^7.x.x",
    "@typescript-eslint/parser": "^7.x.x",
    "eslint": "^8.x.x",
    "eslint-plugin-playwright": "^1.x.x",
    "prettier": "^3.x.x",
    "husky": "^9.x.x",
    "lint-staged": "^15.x.x"
  }
}
```
```javascript
// .eslintrc.js
module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'playwright'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:playwright/recommended',
  ],
  rules: {
    // Add custom rules or override defaults here
  },
  env: {
    node: true,
    browser: true,
  },
};
```
```json
// .prettierrc
{
  "singleQuote": true,
  "semi": true,
  "trailingComma": "all"
}
```
```json
// .husky/pre-commit
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

npx lint-staged
```
```json
// .lintstagedrc.js
module.exports = {
  '*.{ts,js}': ['eslint --fix', 'prettier --write'],
};
```