---
description: This rule enforces best practices and coding standards for Playwright tests, including stable selectors, test isolation, user-centric testing, and performance considerations.
globs: *.spec.ts
---
- **General Principles**
  - **Test User-Visible Behavior:** Focus tests on how users interact with your application, not on internal implementation details.
  - **Isolate Tests:** Ensure tests are independent of each other to prevent cascading failures and ensure predictable results.
  - **Avoid Testing Third-Party Dependencies:** Mock or stub external services and APIs to isolate your application's behavior.

- **Code Organization and Structure**
  - **Directory Structure:**
    - `tests/`: Contains all test files.
    - `tests/e2e/`: End-to-end tests.
    - `tests/unit/`: Unit tests (if applicable, though Playwright is primarily for E2E).
    - `tests/utils/`: Helper functions and page object models.
  - **File Naming Conventions:**
    - Use `.spec.ts` or `.spec.js` for test files (e.g., `login.spec.ts`).
    - Group related tests in the same file.
  - **Module Organization:**
    - Employ Page Object Model (POM) to encapsulate UI elements and interactions.
  - **Component Architecture:**
    - Structure tests around components or features of your application.
  - **Code Splitting Strategies:**
    - Not directly applicable to tests, but keep test files concise and focused.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - **Page Object Model (POM):** A common pattern where each page is represented as a class, with methods for interacting with the page's elements.  This improves reusability and maintainability. Example:
      typescript
      class LoginPage {
        constructor(private readonly page: Page) {}

        async goto() {
          await this.page.goto('/login');
        }

        async login(username: string, password: string) {
          await this.page.fill('#username', username);
          await this.page.fill('#password', password);
          await this.page.click('#login-button');
        }

        async getErrorMessage() {
          return await this.page.textContent('#error-message');
        }
      }
      
    - **Fixture pattern:** Use Playwright's built-in fixtures to manage test setup and teardown. This ensures each test starts in a clean state.
  - **Recommended Approaches:**
    - Use `baseURL` in `playwright.config.ts` to avoid hardcoding URLs in tests.
    - Utilize `expect` matchers for assertions (e.g., `expect(page.locator('#success')).toBeVisible()`).
    - Use auto-waiting features for improved stability.
  - **Anti-patterns:**
    - Hardcoding URLs.
    - Using brittle selectors (e.g., XPath based on DOM structure).
    - Writing tests that depend on each other.
  - **State Management:**
    - Keep tests stateless. Reset the application state before each test.
    - Use database transactions or API calls to seed data for tests.
  - **Error Handling:**
    - Use `try...catch` blocks to handle expected errors.
    - Log errors and failures with descriptive messages.
    - Use `expect.soft()` for non-critical assertions that shouldn't fail the test immediately.

- **Performance Considerations**
  - **Optimization Techniques:**
    - Run tests in parallel to reduce overall test execution time.
    - Use `reuseExistingServer: true` in `playwright.config.ts` during development to speed up debugging.
    - Use `codegen` to generate selectors automatically.
  - **Memory Management:**
    - Close pages and browsers after each test or group of tests to release resources.
  - **Rendering Optimization:**
    - Not directly applicable but optimize your application's rendering for faster testing.
  - **Bundle Size Optimization:**
    - Not directly applicable, but optimize your application's bundle size for faster loading.
  - **Lazy Loading Strategies:**
    - Not directly applicable to tests.

- **Security Best Practices**
  - **Common Vulnerabilities:**
    - Avoid exposing sensitive data (e.g., passwords, API keys) in test code or logs.
  - **Input Validation:**
    - Test input validation to ensure your application handles invalid data correctly.
  - **Authentication and Authorization:**
    - Test different user roles and permissions.
  - **Data Protection:**
    - Ensure sensitive data is encrypted in the database.
  - **Secure API Communication:**
    - Test that API calls are made over HTTPS.

- **Testing Approaches**
  - **Unit Testing:**
    - While Playwright primarily focuses on E2E testing, unit tests can be written for utility functions or components.
  - **Integration Testing:**
    - Test the interaction between different parts of your application.
  - **End-to-End Testing:**
    - Simulate user flows to test the entire application.
  - **Test Organization:**
    - Group tests by feature or functionality.
    - Use `describe` blocks to organize tests.
  - **Mocking and Stubbing:**
    - Use Playwright's `route` API to mock API responses.
    - Use `locator.evaluate` to stub JavaScript functions.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - Using XPath instead of CSS selectors.
    - Not using auto-waiting features.
    - Writing flaky tests.
  - **Edge Cases:**
    - Handling different screen sizes and devices.
    - Testing error conditions and edge cases.
  - **Version-Specific Issues:**
    - Stay up-to-date with Playwright's release notes and upgrade guides.
  - **Compatibility Concerns:**
    - Test on different browsers and operating systems.
  - **Debugging Strategies:**
    - Use Playwright Inspector to debug tests visually.
    - Use `console.log` statements to log information during test execution.
    - Use `pause()` to halt test execution and inspect the page.

- **Tooling and Environment**
  - **Recommended Development Tools:**
    - VS Code with the Playwright extension.
  - **Build Configuration:**
    - Use TypeScript for type safety and autocompletion.
  - **Linting and Formatting:**
    - Use ESLint and Prettier to enforce code style.
  - **Deployment Best Practices:**
    - Run tests in CI/CD pipeline before deploying to production.
  - **CI/CD Integration:**
    - Integrate Playwright with CI/CD tools like GitHub Actions, Jenkins, or GitLab CI.

- **Specific Best Practices & Details**
    - **Stable Selectors:** Prefer CSS selectors based on attributes like `data-testid` or `data-test-id` over XPath or fragile CSS classnames.
    - **Leverage Auto-waiting:** Playwright automatically waits for elements to be actionable before performing actions.  Avoid explicit waits where possible. However, use explicit waits (e.g. `waitForSelector`) when necessary.
    - **Web-First Assertions:** Use `expect` assertions, which retry and wait for conditions to be met. They help to avoid flakiness.
    - **Configure Debugging Highlights:**  Configure `playwright.config.ts` to highlight actions performed by playwright in the browser during debugging to see what's happening step by step. Example:
        typescript
        use: {
            /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
            trace: 'on-first-retry',
            video: 'on',
            screenshot: 'only-on-failure',
        }
        

- **Additional Notes**
    - Regularly review and update your test suite to reflect changes in your application.
    - Document your tests to make them easier to understand and maintain.
    - Use a consistent naming convention for your tests.