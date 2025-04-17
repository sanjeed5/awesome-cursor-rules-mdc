---
description: This rule file provides guidelines for writing stable and maintainable end-to-end tests using Detox, covering code structure, testing strategies, and performance considerations. It includes best practices for test ID usage, dealing with flakiness, and integrating with CI/CD pipelines.
globs: *.e2e.js,*.e2e.ts,*.spec.js,*.spec.ts
---
# Detox E2E Testing Best Practices for React Native

This document outlines best practices for writing stable, maintainable, and performant end-to-end (E2E) tests for React Native applications using Detox.

## Ground Rules

- **Isolate Tests:** Each test should start fresh and not depend on the execution of previous tests.  Restart the app before each test.
- **Consistent Input, Consistent Output:** Ensure identical app behavior across test iterations by managing varying server responses and app states. Mock server responses to prevent external variations.
- **End every test with an expectation:** Ensures a validation phase that confirms the intended outcome of the test.

## 1. Code Organization and Structure

### Directory Structure


<project_root>/
  e2e/
    config.js          # Detox configuration file
    utils.js           # Helper functions for tests
    firstTest.spec.js   # Example test file
    ...              # More test files
  src/
    ...              # Your application source code


- **`e2e` directory:**  Dedicated directory for all Detox-related files.
- **`config.js`:** Contains Detox configuration, including device settings, build paths, and test runner configuration.
- **`utils.js` (or similar):**  Houses reusable helper functions, such as custom matchers, navigation helpers, and data generation functions.
- **`*.spec.js` or `*.e2e.js`:** Test files containing the actual test cases. Choose a consistent naming convention.

### File Naming Conventions

- **Test files:** `[ComponentName].spec.js`, `[FeatureName].e2e.js`.
- **Helper functions:** `helpers.js`, `utils.js`, `testData.js`.
- **Configuration:** `detox.config.js` or `e2e.config.js`.

### Module Organization

- **Separate test logic from application code:** Tests reside in the `e2e` directory, separate from the `src` directory containing the app's source code.
- **Group related tests into modules:** For example, put all tests related to user authentication in an `e2e/auth/` directory.
- **Use helper modules:**  Extract common test logic, like logging in a user or navigating to a specific screen, into reusable helper functions.

### Component Architecture (Page Object Model)

- **Implement the Page Object Model (POM):** Create classes or modules that represent UI screens or components.  Each page object should encapsulate the selectors and actions for that screen.

javascript
// e2e/pageObjects/LoginPage.js
const { element, by, expect } = require('detox');

class LoginPage {
  constructor() {
    this.usernameField = element(by.id('username_input'));
    this.passwordField = element(by.id('password_input'));
    this.loginButton = element(by.id('login_button'));
  }

  async enterUsername(username) {
    await this.usernameField.typeText(username);
  }

  async enterPassword(password) {
    await this.passwordField.typeText(password);
  }

  async tapLoginButton() {
    await this.loginButton.tap();
  }

  async isLoggedIn() {
    await expect(element(by.id('home_screen'))).toBeVisible();
  }
}

module.exports = new LoginPage();


- **Benefits of POM:**
    - Improved test readability and maintainability.
    - Reduced code duplication.
    - Easier refactoring of UI elements.

### Code Splitting Strategies (Not directly applicable to Detox, but consider app bundle size)

- **Dynamic imports:** Utilize dynamic imports in the app code to load modules only when needed, improving initial load time. This is a React Native concern, not Detox itself, but Detox tests will execute faster on a faster-loading app.
- **Route-based splitting:** Split your app into separate bundles based on routes or features.

## 2. Common Patterns and Anti-patterns

### Design Patterns Specific to Detox

- **Wait for element to be visible:**

javascript
await waitFor(element(by.id('my_element'))).toBeVisible().withTimeout(5000);


- **Retry flaky tests:** Implement a retry mechanism for tests that occasionally fail due to timing issues or other unpredictable factors. Use Jest's retryTimes:

javascript
jest.retryTimes(3); // Retry the test up to 3 times


- **Use `beforeAll` and `beforeEach` wisely:** `beforeAll` is for setup that only needs to happen once for the entire test suite, while `beforeEach` is for setup that must be done before each test.

### Recommended Approaches for Common Tasks

- **Simulating user interactions:**

javascript
await element(by.id('my_button')).tap();
await element(by.id('my_input')).typeText('Hello, Detox!');
await element(by.id('my_scroll_view')).scroll(50, 'down');


- **Asserting element visibility and text content:**

javascript
await expect(element(by.id('my_element'))).toBeVisible();
await expect(element(by.text('Expected Text'))).toBeVisible();
await expect(element(by.id('my_element'))).toHaveText('Actual Text');


- **Handling alerts and dialogs:**

javascript
await expect(element(by.text('Alert Title'))).toBeVisible();
await element(by.text('OK')).tap(); // Dismiss the alert


### Anti-patterns and Code Smells to Avoid

- **Hardcoded timeouts:** Avoid using `setTimeout` directly. Rely on Detox's built-in synchronization and waiting mechanisms.
- **Incomplete Test Coverage:** Focus on critical paths and user flows, ensuring comprehensive coverage of the core functionalities.
- **Over-reliance on Thread.sleep() (or equivalent):** Try to use `waitFor` or Detox's synchronization primitives instead.  Sleeping is almost always a sign that Detox's sync isn't working right.
- **Ignoring console.log statements:** Keep console log statements clean and helpful for debugging purposes.

### State Management Best Practices (App-level, impacts test environment)

- **Mock external dependencies:** Mock API calls, database interactions, and other external dependencies to ensure consistent test results and isolate the app's logic.
- **Control app state before each test:** Use `beforeEach` to reset the app's state to a known value before running each test. This can involve clearing data, logging out users, or navigating to a specific screen.

### Error Handling Patterns

- **Use `try...catch` blocks:** Wrap test actions in `try...catch` blocks to handle potential errors gracefully.
- **Log errors:** Log any errors that occur during test execution to help with debugging.
- **Fail tests on unexpected errors:** If a test encounters an unexpected error, mark the test as failed to prevent false positives.

## 3. Performance Considerations

### Optimization Techniques (Mainly React Native app optimization)

- **Minimize UI updates:**  Reduce the number of UI updates during animations or interactions to improve rendering performance.
- **Use virtualization for long lists:** Virtualize long lists to render only the visible items, reducing memory consumption and improving scrolling performance. This improves the app's performance generally, leading to faster test execution.
- **Optimize image loading:** Optimize image sizes and use caching mechanisms to reduce image loading times.

### Memory Management (React Native app concern)

- **Release resources:** Properly release resources, such as event listeners and timers, when they are no longer needed to prevent memory leaks.
- **Use memoization:** Use memoization techniques to avoid recomputing expensive values when they haven't changed.

### Rendering Optimization (React Native app concern)

- **Use `shouldComponentUpdate` or `React.memo`:** Implement `shouldComponentUpdate` or `React.memo` to prevent unnecessary re-renders of components.
- **Avoid inline styles:**  Avoid using inline styles, as they can cause performance issues.

### Bundle Size Optimization (React Native app concern)

- **Use code splitting:** Split your app into smaller chunks to reduce the initial bundle size.
- **Remove unused code:** Remove any unused code, such as dead code or unused libraries.
- **Optimize images:** Optimize image sizes and use appropriate image formats.

### Lazy Loading Strategies (React Native app concern)

- **Load components on demand:**  Load components only when they are needed, such as when a user navigates to a specific screen.
- **Use lazy-loaded images:**  Load images only when they are visible in the viewport.

## 4. Security Best Practices (App Security - Impacts Test Scope)

### Common Vulnerabilities and How to Prevent Them

- **Data injection:** Protect against data injection attacks by validating all user input and using parameterized queries.
- **Cross-site scripting (XSS):**  Prevent XSS attacks by sanitizing all user-generated content before rendering it.
- **Man-in-the-middle (MITM) attacks:**  Implement SSL/TLS to encrypt communication between the app and the server.

### Input Validation

- **Validate all user input:** Validate all user input on both the client and server sides to prevent invalid or malicious data from being processed.
- **Use appropriate validation libraries:** Use established validation libraries to simplify the validation process and ensure consistency.

### Authentication and Authorization Patterns

- **Use secure authentication protocols:** Use secure authentication protocols, such as OAuth 2.0 or OpenID Connect.
- **Implement proper authorization checks:** Implement proper authorization checks to ensure that users can only access the resources they are authorized to access.

### Data Protection Strategies

- **Encrypt sensitive data:** Encrypt sensitive data, such as passwords and credit card numbers, both in transit and at rest.
- **Store data securely:** Store data in secure locations, such as encrypted databases or keychains.

### Secure API Communication

- **Use HTTPS:** Always use HTTPS to encrypt communication between the app and the server.
- **Validate API responses:** Validate API responses to ensure that they are not tampered with.

## 5. Testing Approaches

### Unit Testing Strategies (Generally not used with Detox)

- **Test individual components or functions in isolation:** Use unit tests to verify the behavior of individual components or functions in isolation.
- **Use mocking to isolate dependencies:** Use mocking to isolate dependencies and ensure that the unit tests are not affected by external factors.

### Integration Testing

- **Test interactions between multiple components or modules:** Use integration tests to verify the interactions between multiple components or modules.
- **Use stubs to simulate external dependencies:** Use stubs to simulate external dependencies and ensure that the integration tests are not affected by real-world conditions.

### End-to-End Testing

- **Test the entire application from end to end:** Use end-to-end tests to verify the entire application flow from the user's perspective.
- **Simulate real user interactions:** Simulate real user interactions as closely as possible to ensure that the tests accurately reflect the user experience.

### Test Organization

- **Organize tests by feature or functionality:** Group tests by feature or functionality to improve test discoverability and maintainability.
- **Use descriptive test names:** Use descriptive test names to clearly indicate what each test is verifying.
- **Keep tests small and focused:** Keep tests small and focused on verifying a single aspect of the application.

### Mocking and Stubbing (Used sparingly with Detox. More common is configuring the backend)

- **Mock external dependencies:**  Mock external dependencies, such as API calls or database interactions, to ensure consistent test results.
- **Use stubs to simulate specific scenarios:**  Use stubs to simulate specific scenarios, such as error conditions or edge cases.

## 6. Common Pitfalls and Gotchas

### Frequent Mistakes Developers Make

- **Using brittle selectors:** Avoid using selectors that are based on text or position, as these can easily break when the UI changes. Use `testID` whenever possible.
- **Not handling asynchronous operations correctly:** Ensure that all asynchronous operations are properly handled to prevent race conditions and timing issues.
- **Ignoring error messages:** Pay attention to error messages and logs to identify and fix problems quickly.

### Edge Cases to Be Aware Of

- **Handling different screen sizes and orientations:**  Test the app on different screen sizes and orientations to ensure that the UI adapts correctly.
- **Handling network connectivity issues:**  Test the app's behavior when the network connection is slow or unavailable.
- **Handling different languages and locales:**  Test the app with different languages and locales to ensure that it is properly localized.

### Version-Specific Issues

- **Keep Detox up to date:**  Stay up to date with the latest version of Detox to take advantage of bug fixes and performance improvements.
- **Be aware of compatibility issues:**  Be aware of compatibility issues between different versions of Detox, React Native, and other dependencies.

### Compatibility Concerns

- **Test on multiple devices and operating systems:**  Test the app on a variety of devices and operating systems to ensure that it is compatible with the target audience.
- **Use a device farm:**  Use a device farm, such as HeadSpin, to test the app on a wide range of real devices.

### Debugging Strategies

- **Use the Detox CLI:**  Use the Detox CLI to run tests, inspect the app's state, and debug issues.
- **Use logging statements:**  Use logging statements to track the execution flow and identify potential problems.
- **Use breakpoints:**  Use breakpoints in your test code to pause execution and inspect variables.

## 7. Tooling and Environment

### Recommended Development Tools

- **Visual Studio Code:** A popular code editor with excellent support for JavaScript and React Native.
- **Detox CLI:** The Detox command-line interface for running tests and managing the Detox environment.
- **HeadSpin:** For running tests on real devices and analyzing performance metrics.

### Build Configuration

- **Configure Detox in `package.json`:** Configure Detox in the `package.json` file, including device settings, build paths, and test runner configuration.
- **Use different configurations for different environments:** Use different configurations for development, staging, and production environments.

### Linting and Formatting

- **Use ESLint:** Use ESLint to enforce code style and detect potential errors.
- **Use Prettier:** Use Prettier to automatically format code.

### Deployment Best Practices

- **Automate the deployment process:** Automate the deployment process using tools like Fastlane or Bitrise.
- **Use continuous integration and continuous delivery (CI/CD):** Use CI/CD to automatically build, test, and deploy the app whenever changes are made.

### CI/CD Integration

- **Integrate Detox tests into your CI/CD pipeline:** Integrate Detox tests into your CI/CD pipeline to automatically run tests whenever changes are made.
- **Use a CI/CD service like Jenkins, CircleCI, or Travis CI:** Use a CI/CD service like Jenkins, CircleCI, or Travis CI to automate the build, test, and deployment process.

## Working with Test IDs

### Benefits of Test IDs

*   Stable matchers that don't depend on UI text.
*   Locale-agnostic testing.
*   Clear code navigability.

### Assigning Test IDs

*   **React Native:** Use the `testID` prop on View components.

    jsx
    <View>
      <TouchableOpacity testID="Onboarding.Next_button">
        <Text>Next</Text>
      </TouchableOpacity>
    </View>
    

*   **iOS Native:** Use the `accessibilityIdentifier` property.
*   **Android Native:** Use the `viewTag` property.

### Best Practices for Naming Test IDs

*   Use a consistent naming system (e.g., `ITEM_NAME_ALL_CAPS` or `ItemNameUpperCamelCase`).
*   Use notations for special items (e.g., `_ROOT` for screen roots, `_BTN` for buttons).
*   Apply consistent prefixes as categories (e.g., `EDIT_PROFILE_SCREEN.DONE_BTN`).
*   Drill down to details of elements via chains of contexts (e.g., `SITE_LIST_ITEM1.OPTIONS`).
*   In large projects, use module identifiers (e.g., `AUTH.LOGIN_SCREEN.EDIT_PASSWORD`).

#### Examples of good test ID usage:


EDIT_PROFILE_SCREEN.DONE_BTN
SITE_LIST_ROOT
SITE_LIST_ITEM1.OPTIONS


### Rules of Thumb for test IDs

*   Use unique, simple, and concise names.
*   Dissociate test ID names from element text/labels.
*   Do not utilize the element's text / label in the naming of a test ID!

### Finding Test IDs

*   **MacOS Accessibility Inspector:** Use the built-in accessibility inspector to verify test IDs on iOS.
*   **Detox Layout Inspector:** Set up and use the Detox Layout Inspector for Android.
* Incorrect or absent testID is a common cause for test failure. If your test can't find your testID and you can't see it either using tools described above, that usually means you haven't passed it down to this component.
Make sure you keep forwarding it down until it reaches a native component.

By following these best practices, you can create a robust and maintainable suite of Detox E2E tests for your React Native application.