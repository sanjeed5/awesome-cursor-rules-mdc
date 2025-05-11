---
description: This rule file provides comprehensive best practices for using Vitest, covering code organization, testing strategies, performance, and security within Vitest projects. These guidelines ensure clean, maintainable, and reliable test suites.
globs: *.js,*.jsx,*.ts,*.tsx,*.vue,*.svelte,*.spec,*.test.*
---
# Vitest Best Practices and Coding Standards

This document outlines best practices for using Vitest to create reliable, maintainable, and performant test suites. It covers various aspects of testing, from code organization to performance considerations and security measures.

## 1. Code Organization and Structure

### 1.1 Directory Structure

*   **Keep tests close to the source code:** Place test files in the same directory as the components or modules they test. This improves discoverability and maintainability.

    
    src/
      components/
        MyComponent.vue
        MyComponent.spec.ts
        MyComponent.test.ts  # Alternative naming
      utils/
        math.ts
        math.test.ts
      App.vue
      App.spec.ts
    

*   **Use a dedicated `tests` directory for end-to-end tests or shared utilities:**  For larger projects, a `tests` directory at the root level can house end-to-end tests, integration tests that require a specific environment, or shared testing utilities.

    
    tests/
      e2e/
        specs/
          home.spec.ts
        support/
          commands.ts
      unit/
        utils.test.ts # Tests for general utilities
      integration/
        db.setup.ts # Setup for integration tests 
    src/ 
      ...
    

### 1.2 File Naming Conventions

*   **Use consistent naming:** Adopt a consistent naming scheme for test files. Common conventions include:
    *   `[component/module].spec.ts`
    *   `[component/module].test.ts`
    *   `[component/module].e2e.ts` (for end-to-end tests)

*   **Be descriptive:** Name test files to clearly indicate what they are testing. For example, `MyComponent.props.spec.ts` might test specific props of `MyComponent`.

### 1.3 Module Organization

*   **Group related tests:** Organize tests into modules using `describe` blocks. This improves readability and helps structure test output.

    typescript
    import { describe, it, expect } from 'vitest';
    import { add } from './math';

    describe('Math functions', () => {
      describe('add', () => {
        it('should add two numbers correctly', () => {
          expect(add(2, 3)).toBe(5);
        });

        it('should handle negative numbers', () => {
          expect(add(-1, 1)).toBe(0);
        });
      });
    });
    

### 1.4 Component Architecture

*   **Test component logic separately:** Extract complex logic from components into separate, testable functions or modules. This promotes reusability and simplifies component testing.

*   **Focus on component interactions:**  When testing components, concentrate on verifying that the component renders correctly given certain props or state and that it emits the correct events in response to user interactions.

### 1.5 Code Splitting Strategies

*   **Test code splitting:**  If your application uses code splitting, ensure your tests cover different code chunks and lazy-loaded modules.

*   **Mock dynamic imports:**  Use Vitest's mocking capabilities to simulate dynamic imports during testing.

    typescript
    import { describe, it, expect, vi } from 'vitest';

    describe('Dynamic import', () => {
      it('should mock dynamic import', async () => {
        const mockModule = { value: 'mocked' };
        vi.mock('./dynamic-module', () => ({
          default: mockModule,
        }));

        const dynamicModule = await import('./dynamic-module');
        expect(dynamicModule.default).toBe(mockModule);
      });
    });
    

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to Vitest

*   **AAA (Arrange, Act, Assert):** Structure each test case following the AAA pattern for clarity and maintainability.

    typescript
    it('should add two numbers correctly', () => {
      // Arrange
      const a = 2;
      const b = 3;

      // Act
      const result = add(a, b);

      // Assert
      expect(result).toBe(5);
    });
    

*   **Page Object Model (POM):** For end-to-end tests, use the Page Object Model to abstract away the details of the user interface and make tests more resilient to UI changes. Define dedicated classes representing different pages or components.

### 2.2 Recommended Approaches for Common Tasks

*   **Mocking external dependencies:**  Use Vitest's mocking capabilities to isolate units of code during testing. Use `vi.mock()` to mock modules and `vi.spyOn()` to spy on specific methods or properties.

*   **Testing asynchronous code:** Utilize `async/await` and Vitest's `expect.resolves` and `expect.rejects` matchers for testing asynchronous functions.

    typescript
    it('should resolve with the correct value', async () => {
      await expect(Promise.resolve(42)).resolves.toBe(42);
    });

    it('should reject with an error', async () => {
      await expect(Promise.reject(new Error('Something went wrong'))).rejects.toThrow('Something went wrong');
    });
    

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Over-mocking:** Avoid mocking everything.  Mock only external dependencies or components that are not under test.  Testing against mocks rather than real implementations reduces test confidence.

*   **Flaky tests:**  Tests that pass or fail intermittently are a sign of underlying issues.  Investigate flaky tests to identify and fix the root cause, such as race conditions or reliance on external resources.

*   **Ignoring edge cases:** Ensure your tests cover all possible scenarios, including edge cases, error conditions, and boundary values. Don't only test the "happy path".

*   **Long test functions:** Break down overly complex test functions into smaller, more focused tests. This improves readability and makes it easier to identify the cause of failures.

### 2.4 State Management Best Practices

*   **Isolate state:**  When testing code that relies on state management libraries (e.g., Vuex, Pinia, Redux), isolate the state and actions being tested.

*   **Mock store actions/getters:** Mock actions and getters to control the state during testing and verify that they are called correctly.

    typescript
    import { describe, it, expect, vi } from 'vitest';
    import { useStore } from './store';

    describe('Store actions', () => {
      it('should dispatch the correct action', () => {
        const store = useStore();
        const mockAction = vi.fn();
        store.dispatch = mockAction;

        store.commit('increment');
        expect(mockAction).toHaveBeenCalledWith('increment');
      });
    });
    

### 2.5 Error Handling Patterns

*   **Test error handling:** Ensure your tests cover error handling scenarios.  Use `try...catch` blocks or `expect.rejects` to verify that errors are thrown and handled correctly.

*   **Mock error responses:** Mock API responses to simulate error conditions and test how your code handles them.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Run tests in parallel:** Vitest supports running tests in parallel. Enable this feature to speed up test execution.

    json
    // vitest.config.ts
    import { defineConfig } from 'vitest/config'

    export default defineConfig({
      test: {
        threads: true,  // Enable parallel execution
      },
    })
    

*   **Use `--changed` and `--related` flags:**  When running tests, use the `--changed` flag to only run tests that have changed since the last commit or the `--related` flag to run tests related to specific files.

    bash
    vitest --changed
    vitest --related src/components/MyComponent.vue
    

*   **Optimize test setup:**  Minimize the amount of setup required for each test.  Use `beforeAll` and `afterAll` hooks to perform setup and teardown operations once for each test suite, rather than for each test case.

### 3.2 Memory Management

*   **Clean up after tests:**  Ensure that your tests do not leak memory.  Use `afterEach` hooks to clean up any resources created during the test, such as mocks or temporary files.

*   **Avoid creating large objects in tests:**  Minimize the size of objects created in tests to reduce memory consumption.

### 3.3 Rendering Optimization

*   **Use shallow rendering:**  When testing components, use shallow rendering to avoid rendering the entire component tree. This can significantly improve test performance.

    typescript
    import { shallowMount } from '@vue/test-utils';
    import MyComponent from './MyComponent.vue';

    it('should render correctly', () => {
      const wrapper = shallowMount(MyComponent);
      expect(wrapper.exists()).toBe(true);
    });
    

### 3.4 Bundle Size Optimization

*   **Keep tests small:**  Avoid including unnecessary dependencies in your test files.  This can help reduce the bundle size of your tests and improve startup time.

### 3.5 Lazy Loading Strategies

*   **Mock lazy-loaded modules:**  When testing code that uses lazy loading, mock the lazy-loaded modules to avoid loading them during testing.  This can improve test performance and reduce dependencies.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Prevent XSS vulnerabilities by sanitizing user input and encoding output. Ensure the testing framework is not vulnerable either. Check versions of plugins.

*   **Injection Attacks:**  Prevent injection attacks by validating user input and using parameterized queries.

*   **Sensitive Data Exposure:** Avoid storing sensitive data in test files. Use environment variables or secure configuration files to manage sensitive data.

### 4.2 Input Validation

*   **Test input validation:**  Ensure your tests cover input validation scenarios.  Verify that your code correctly validates user input and handles invalid input gracefully.

### 4.3 Authentication and Authorization Patterns

*   **Mock authentication:**  When testing code that requires authentication, mock the authentication service to avoid making actual API calls.  Verify that your code correctly handles authenticated and unauthenticated states.

*   **Test authorization:**  Ensure your tests cover authorization scenarios.  Verify that your code correctly enforces access control and prevents unauthorized access to resources.

### 4.4 Data Protection Strategies

*   **Protect sensitive data in tests:**  Avoid including sensitive data in your test files.  Use mock data or anonymized data for testing.

### 4.5 Secure API Communication

*   **Mock API responses:**  Mock API responses to avoid making actual API calls during testing.  Use HTTPS for secure communication.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Focus on individual units:**  Unit tests should focus on testing individual functions, classes, or modules in isolation. Avoid testing multiple units of code in a single test.

*   **Test all code paths:** Ensure your unit tests cover all possible code paths, including normal execution paths, error conditions, and edge cases.

*   **Use mocks and stubs:** Use mocks and stubs to isolate units of code and control their behavior during testing. 

### 5.2 Integration Testing

*   **Test interactions between units:**  Integration tests should focus on testing the interactions between different units of code. Verify that units of code work together correctly.

*   **Use real dependencies:**  Use real dependencies in integration tests whenever possible.  This can help ensure that your code works correctly in a real-world environment.

### 5.3 End-to-End Testing

*   **Test the entire application:** End-to-end tests should focus on testing the entire application, from the user interface to the backend services. Verify that the application works correctly from the user's perspective.

*   **Use a real browser:**  Use a real browser for end-to-end testing. This can help ensure that your application works correctly in different browsers and environments.

### 5.4 Test Organization

*   **Group related tests:**  Organize tests into modules using `describe` blocks. This improves readability and helps structure test output. (See 1.3 Module Organization)

*   **Use meaningful test names:** Use descriptive test names that clearly indicate what the test is verifying. This makes it easier to understand the purpose of the test and to identify the cause of failures.

*   **Keep tests short and focused:** Keep tests short and focused on a single aspect of the code. This improves readability and makes it easier to maintain the tests.

### 5.5 Mocking and Stubbing

*   **Use mocks to verify interactions:**  Use mocks to verify that functions are called with the correct arguments and that they return the correct values.

*   **Use stubs to control behavior:** Use stubs to control the behavior of functions during testing. This allows you to simulate different scenarios and test how your code handles them.

*   **Avoid over-mocking:** Only mock dependencies that are not under test.  Over-mocking can lead to tests that are brittle and do not accurately reflect the behavior of the code.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Not writing enough tests:** Ensure that you have sufficient test coverage to catch bugs and prevent regressions.

*   **Writing brittle tests:** Avoid writing tests that are too tightly coupled to the implementation details of the code. This makes the tests brittle and difficult to maintain.

*   **Ignoring test failures:** Address test failures promptly.  Ignoring test failures can lead to regressions and make it more difficult to maintain the code.

### 6.2 Edge Cases to Be Aware Of

*   **Null and undefined values:** Ensure your tests cover scenarios where values are null or undefined.

*   **Empty strings and arrays:** Ensure your tests cover scenarios where strings or arrays are empty.

*   **Boundary values:** Ensure your tests cover boundary values, such as the minimum and maximum values of numeric types.

### 6.3 Version-Specific Issues

*   **Keep Vitest up to date:** Stay up-to-date with the latest version of Vitest to benefit from bug fixes, performance improvements, and new features.

### 6.4 Compatibility Concerns

*   **Test on different browsers and environments:** Ensure your tests cover different browsers and environments to catch compatibility issues.

### 6.5 Debugging Strategies

*   **Use debugging tools:** Utilize debugging tools to step through your tests and identify the cause of failures.  Vitest integrates with popular debuggers.

*   **Write clear and concise tests:** Write clear and concise tests to make it easier to understand the purpose of the test and to identify the cause of failures.

*   **Use logging:**  Add logging statements to your tests to help track the flow of execution and identify the source of problems.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **VS Code with the Vitest extension:** VS Code with the Vitest extension provides a rich development experience, including test discovery, execution, and debugging.

*   **Other IDEs with Vitest support:** Many other IDEs, such as WebStorm and IntelliJ IDEA, also offer support for Vitest.

### 7.2 Build Configuration

*   **Use a build tool:** Use a build tool, such as Vite or esbuild, to bundle your code and optimize it for testing.

*   **Configure Vitest:** Configure Vitest to suit your project's needs.  Use the `vitest.config.ts` file to customize Vitest's behavior.

### 7.3 Linting and Formatting

*   **Use ESLint and Prettier:**  Use ESLint and Prettier to enforce consistent coding styles and catch potential errors. Integrate these tools into your development workflow.

### 7.4 Deployment Best Practices

*   **Run tests before deployment:** Always run your tests before deploying your code to ensure that it is working correctly.  Automate this process as part of your CI/CD pipeline.

### 7.5 CI/CD Integration

*   **Integrate Vitest with your CI/CD pipeline:**  Integrate Vitest with your CI/CD pipeline to automatically run tests on every commit. This helps catch bugs early and prevent regressions.

*   **Use a CI/CD service:** Use a CI/CD service, such as GitHub Actions or GitLab CI, to automate your build, test, and deployment processes.

## Conclusion

By following these best practices, you can create robust and maintainable test suites with Vitest that ensure the quality and reliability of your code.