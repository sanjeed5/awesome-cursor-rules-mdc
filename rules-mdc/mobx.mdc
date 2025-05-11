---
description: This rule provides comprehensive guidance for using MobX effectively, covering best practices for code organization, performance, testing, and common pitfalls. It aims to ensure efficient and maintainable state management in React and other JavaScript applications using MobX.
globs: *.js,*.jsx,*.ts,*.tsx
---
# MobX Best Practices and Coding Standards

This document outlines the best practices for using MobX in your projects. Following these guidelines will help you write more maintainable, performant, and robust code.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

*   **Feature-Based Organization:** Organize your code by feature rather than by file type (e.g., components, stores, utils). This promotes better modularity and easier navigation.

    
    src/
    ├── features/
    │   ├── user-profile/
    │   │   ├── components/
    │   │   │   ├── UserProfile.jsx
    │   │   │   └── UserDetails.jsx
    │   │   ├── stores/
    │   │   │   ├── userStore.js
    │   │   ├── api/
    │   │   │   └── userApi.js
    │   │   └── utils/
    │   │       └── userUtils.js
    │   ├── product-listing/
    │   │   └── ...
    ├── app.js
    └── ...
    

*   **Dedicated `stores` Directory:** Place all your MobX stores in a dedicated `stores` directory to clearly separate state management logic from the rest of your application.

*   **Shared Utilities:** Create a `utils` directory for reusable utility functions.

### 1.2 File Naming Conventions

*   **Descriptive Names:** Use descriptive names for files and modules that clearly indicate their purpose.
*   **Consistent Case:** Maintain a consistent naming convention (e.g., camelCase for JavaScript files, PascalCase for React components).

### 1.3 Module Organization

*   **Single Responsibility Principle:** Each module should have a single, well-defined responsibility.
*   **Clear Exports:** Clearly define the exports of each module (e.g., named exports for individual functions, default export for the main component or store).
*   **Avoid Circular Dependencies:** Ensure that modules do not have circular dependencies to prevent runtime errors and improve code maintainability. Use dependency injection if necessary.

### 1.4 Component Architecture

*   **Presentational and Container Components:** Separate presentational (UI-focused) components from container (data-fetching and logic-handling) components.  Presentational components receive data via props, while container components connect to MobX stores and pass data down.

    jsx
    // Container component
    import React from 'react';
    import { observer } from 'mobx-react-lite';
    import { useUserStore } from './stores/userStore';
    import UserProfile from './UserProfile';

    const UserProfileContainer = observer(() => {
      const userStore = useUserStore();

      return <UserProfile user={userStore.user} />; // Pass data as props
    });

    export default UserProfileContainer;

    // Presentational component
    import React from 'react';

    const UserProfile = ({ user }) => {
      return (
        <div>
          <h1>{user.name}</h1>
          <p>{user.email}</p>
        </div>
      );
    };

    export default UserProfile;
    

*   **Functional Components with Hooks:** Use functional components with the `useObserver` hook (or `observer` from `mobx-react-lite`) for better performance and readability.

    jsx
    import React from 'react';
    import { observer } from 'mobx-react-lite';
    import { useUserStore } from './stores/userStore';

    const UserProfile = observer(() => {
      const userStore = useUserStore();

      return (
        <div>
          <h1>{userStore.user.name}</h1>
          <p>{userStore.user.email}</p>
        </div>
      );
    });

    export default UserProfile;
    

*   **Component Composition:** Favor component composition over deep inheritance to create reusable and flexible components.

### 1.5 Code Splitting Strategies

*   **Route-Based Splitting:** Split your application into chunks based on routes or pages. This allows users to download only the code they need for the current view.
*   **Component-Based Splitting:** Split large components into smaller chunks that can be loaded on demand.  Use `React.lazy` and `Suspense` for lazy loading components.
*   **Vendor Splitting:** Separate your vendor dependencies (e.g., libraries) into a separate chunk. This allows browsers to cache vendor code separately from your application code.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to MobX

*   **Observable State:** Use `@observable` to define the state that MobX should track for changes.  Ensure that only necessary data is made observable to optimize performance.
*   **Computed Properties:** Use `@computed` to derive values from observable state. Computed properties are automatically updated when their dependencies change and are cached for performance.

    javascript
    import { makeObservable, observable, computed } from 'mobx';

    class Cart {
      items = [];

      constructor() {
        makeObservable(this, {
          items: observable,
          totalPrice: computed
        });
      }

      get totalPrice() {
        return this.items.reduce((sum, item) => sum + item.price, 0);
      }
    }
    

*   **Actions:** Use `@action` to modify the state. Actions ensure that state changes are batched and tracked by MobX.  All state modifications should happen within actions to maintain predictability.

    javascript
    import { makeObservable, observable, computed, action } from 'mobx';

    class Cart {
      items = [];

      constructor() {
        makeObservable(this, {
          items: observable,
          totalPrice: computed,
          addItem: action
        });
      }

      get totalPrice() {
        return this.items.reduce((sum, item) => sum + item.price, 0);
      }

      addItem(item) {
        this.items.push(item);
      }
    }
    

*   **Reactions:** Use `reaction`, `autorun`, and `when` to react to state changes.  Use `reaction` for side effects that depend on specific observable values, `autorun` for side effects that depend on any observable value, and `when` for one-time side effects.

### 2.2 Recommended Approaches for Common Tasks

*   **Form Handling:** Use MobX to manage form state. Create observable properties for each form field and use actions to update them.

    jsx
    import React from 'react';
    import { observer } from 'mobx-react-lite';
    import { makeObservable, observable, action } from 'mobx';

    class FormStore {
      name = '';
      email = '';

      constructor() {
        makeObservable(this, {
          name: observable,
          email: observable,
          setName: action,
          setEmail: action
        });
      }

      setName(value) {
        this.name = value;
      }

      setEmail(value) {
        this.email = value;
      }
    }

    const formStore = new FormStore();

    const Form = observer(() => {
      return (
        <form>
          <input type="text" value={formStore.name} onChange={e => formStore.setName(e.target.value)} />
          <input type="email" value={formStore.email} onChange={e => formStore.setEmail(e.target.value)} />
        </form>
      );
    });

    export default Form;
    

*   **Asynchronous Operations:** Use actions to handle asynchronous operations such as API calls. Use `async/await` syntax to simplify asynchronous code.

    javascript
    import { makeObservable, observable, action } from 'mobx';

    class UserStore {
      user = null;
      loading = false;

      constructor() {
        makeObservable(this, {
          user: observable,
          loading: observable,
          fetchUser: action
        });
      }

      async fetchUser(id) {
        this.loading = true;
        try {
          const response = await fetch(`/api/users/${id}`);
          this.user = await response.json();
        } finally {
          this.loading = false;
        }
      }
    }
    

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Mutating Observables Directly:** Avoid directly mutating observable values outside of actions. This can lead to unexpected behavior and make it difficult to track state changes. Always use actions to modify observable state.
*   **Over-Observing:** Avoid making everything observable. Only observe the data that needs to be tracked for changes. Over-observing can lead to performance issues.
*   **Complex Computed Properties:** Keep computed properties simple and focused. Avoid complex logic in computed properties, as this can make your code harder to understand and debug.
*   **Using `autorun` Excessively:** Be careful when using `autorun`, as it can easily lead to performance issues if not used correctly. Prefer `reaction` when you need to react to specific observable values.
*   **Forgetting to Dispose Reactions:** Always dispose of reactions when they are no longer needed to prevent memory leaks. Use the `dispose` function returned by `autorun` and `reaction`.

### 2.4 State Management Best Practices

*   **Single Source of Truth:** Maintain a single source of truth for your application's state. Avoid duplicating state across multiple stores.
*   **Normalized State:** Normalize your state to reduce redundancy and improve performance. Store data in a flat, relational structure.
*   **Immutability (with MobX's Mutability):** While MobX embraces mutability for performance, try to treat your data as immutable as possible, especially when working with arrays and objects.  Instead of directly modifying arrays, use methods like `concat`, `slice`, and `filter` to create new arrays.
*   **Centralized State Management:** Use MobX to manage all your application's state in a centralized location. This makes it easier to reason about and debug your code.

### 2.5 Error Handling Patterns

*   **Try-Catch Blocks:** Use `try-catch` blocks to handle errors in asynchronous operations and other code that might throw exceptions.
*   **Error Stores:** Create dedicated error stores to manage application-wide errors.  This allows you to display error messages to the user and log errors for debugging.
*   **Global Error Handling:** Implement global error handling to catch unhandled exceptions and prevent your application from crashing.  Use `window.onerror` or `React Error Boundaries`.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Use `mobx-react-lite`:** Use `mobx-react-lite` instead of `mobx-react` for smaller bundle size and improved performance. `mobx-react-lite` provides hooks-based integration with React.
*   **`useMemo` and `useCallback`:** Use `useMemo` and `useCallback` to optimize rendering performance by memoizing expensive calculations and preventing unnecessary re-renders.

    jsx
    import React, { useMemo } from 'react';
    import { observer } from 'mobx-react-lite';

    const MyComponent = observer(() => {
      const expensiveValue = useMemo(() => {
        // Perform expensive calculation
        return computeExpensiveValue();
      }, []);

      return <div>{expensiveValue}</div>;
    });

    export default MyComponent;
    

*   **`shouldComponentUpdate` (Class Components):**  If you are using class components, implement `shouldComponentUpdate` to prevent unnecessary re-renders.  Compare the previous and current props and state to determine if a re-render is necessary. Consider using `PureComponent`.
*   **Minimize Re-renders:** Minimize the number of re-renders by optimizing your component structure and using techniques like `useMemo` and `useCallback`.

### 3.2 Memory Management

*   **Dispose Reactions:** Always dispose of reactions when they are no longer needed to prevent memory leaks. Use the `dispose` function returned by `autorun` and `reaction`.
*   **Avoid Creating Unnecessary Objects:** Avoid creating unnecessary objects, especially in computed properties and reactions. This can lead to memory leaks and performance issues.
*   **Garbage Collection:** Be aware of JavaScript's garbage collection mechanism and avoid creating circular references that can prevent garbage collection.

### 3.3 Rendering Optimization

*   **Virtualization:** Use virtualization techniques to render large lists efficiently. Virtualization renders only the visible items in the list, which can significantly improve performance.
*   **Debouncing and Throttling:** Use debouncing and throttling to limit the frequency of updates to the UI. This can improve performance by preventing excessive re-renders.

### 3.4 Bundle Size Optimization

*   **Code Splitting:** Use code splitting to reduce the initial bundle size of your application. This allows users to download only the code they need for the current view.
*   **Tree Shaking:** Use tree shaking to remove dead code from your bundle. Tree shaking is a technique that removes unused code from your bundle, which can significantly reduce its size.
*   **Minification:** Use minification to reduce the size of your code. Minification removes whitespace and comments from your code, which can reduce its size.

### 3.5 Lazy Loading Strategies

*   **Lazy Loading Components:** Use `React.lazy` and `Suspense` to lazy load components. This allows you to load components on demand, which can improve the initial load time of your application.
*   **Lazy Loading Images:** Use lazy loading for images to improve the initial load time of your application. This can be done using the `loading` attribute on the `img` element or using a library like `react-lazyload`.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):** Prevent XSS attacks by sanitizing user input and escaping output. Use libraries like `DOMPurify` to sanitize HTML.
*   **Cross-Site Request Forgery (CSRF):** Prevent CSRF attacks by using CSRF tokens. CSRF tokens are unique, secret values that are included in requests to prevent attackers from forging requests on behalf of users.
*   **SQL Injection:** Prevent SQL injection attacks by using parameterized queries or an ORM (Object-Relational Mapper). Parameterized queries escape user input, which prevents attackers from injecting malicious SQL code.

### 4.2 Input Validation

*   **Server-Side Validation:** Validate user input on the server-side to prevent malicious data from being stored in your database.
*   **Client-Side Validation:** Validate user input on the client-side to provide immediate feedback to the user and improve the user experience. However, always validate on the server-side as well, since client-side validation can be bypassed.
*   **Regular Expressions:** Use regular expressions to validate user input. Regular expressions are a powerful tool for validating data against specific patterns.

### 4.3 Authentication and Authorization Patterns

*   **Authentication:** Use a secure authentication mechanism to verify the identity of users. Use libraries like `Passport.js` or `Auth0` to simplify the authentication process.
*   **Authorization:** Implement authorization to control access to resources based on the user's role. Use role-based access control (RBAC) or attribute-based access control (ABAC) to manage access permissions.
*   **JSON Web Tokens (JWT):** Use JWTs to securely transmit user information between the client and the server. JWTs are digitally signed, which makes them tamper-proof.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit. Use HTTPS to encrypt data in transit and libraries like `bcrypt` to encrypt passwords.
*   **Data Masking:** Mask sensitive data to protect it from unauthorized access. Data masking replaces sensitive data with fictitious data, which allows developers to work with the data without exposing the actual sensitive information.
*   **Data Anonymization:** Anonymize data to remove personally identifiable information (PII). Data anonymization is a technique that removes or modifies PII to prevent it from being linked to a specific individual.

### 4.5 Secure API Communication

*   **HTTPS:** Always use HTTPS to encrypt communication between the client and the server. HTTPS uses TLS/SSL to encrypt data in transit, which prevents eavesdropping.
*   **API Keys:** Use API keys to authenticate requests to your API. API keys are unique, secret values that are used to identify the client making the request.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your API. Rate limiting limits the number of requests that a client can make within a given time period.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test Stores in Isolation:** Unit test MobX stores in isolation to verify that their state and actions behave as expected. Use mocking and stubbing to isolate the stores from external dependencies.
*   **Test Computed Properties:** Test computed properties to ensure that they correctly derive values from observable state.
*   **Test Actions:** Test actions to ensure that they correctly modify the state.

### 5.2 Integration Testing

*   **Test Components with Stores:** Integration test components with MobX stores to verify that they interact correctly. Use a testing library like `React Testing Library` to render the components and simulate user interactions.
*   **Test API Interactions:** Test API interactions to ensure that data is correctly fetched from and sent to the server. Use mocking to isolate the components from the actual API.

### 5.3 End-to-End Testing

*   **Automated Browser Tests:** Use end-to-end testing frameworks like Cypress or Selenium to automate browser tests. End-to-end tests verify that the entire application works correctly from the user's perspective.

### 5.4 Test Organization

*   **Separate Test Files:** Create separate test files for each module or component. This makes it easier to find and run the tests.
*   **Descriptive Test Names:** Use descriptive names for your tests that clearly indicate what they are testing.
*   **Test Suites:** Organize your tests into test suites based on functionality or module.

### 5.5 Mocking and Stubbing

*   **Mock Dependencies:** Use mocking to replace external dependencies with mock objects. This allows you to isolate the code being tested and control its behavior.
*   **Stub Functions:** Use stubbing to replace functions with predefined return values or behavior. This allows you to control the behavior of the code being tested without actually executing it.
*   **Mock API Calls:** Mock API calls to avoid making real API requests during testing. This makes your tests faster and more reliable.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Forgetting to Wrap Components with `observer`:** Forgetting to wrap React components with `observer` (or using `useObserver`) prevents them from reacting to changes in the MobX store.
*   **Directly Modifying Observable Arrays/Objects:** Directly modifying observable arrays or objects (e.g., `myArray[0] = 'new value'`) won't trigger reactivity. Always use the methods provided by MobX (e.g., `myArray.splice(0, 1, 'new value')` or `myObject.set('key', 'value')`).
*   **Not Using `useLocalObservable` in Components:**  Using `useLocalObservable` is crucial for creating isolated, component-specific stores, preventing unintended state sharing.

### 6.2 Edge Cases to Be Aware Of

*   **React Strict Mode:** Be aware that React Strict Mode can cause MobX to re-run reactions multiple times. This can be useful for debugging, but it can also lead to performance issues.
*   **Large Datasets:** Be careful when working with large datasets in MobX. Consider using virtualization techniques to improve performance.

### 6.3 Version-Specific Issues

*   **MobX 5 vs. MobX 6:** Be aware of the differences between MobX 5 and MobX 6. MobX 6 introduced several breaking changes, including the removal of implicit observability. Make sure your code is compatible with the version of MobX you are using.
*   **React Compatibility:** Ensure that your version of `mobx-react` or `mobx-react-lite` is compatible with your version of React.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:** Ensure that your code is compatible with the browsers you are targeting. Use polyfills to support older browsers.
*   **Node.js Compatibility:** Ensure that your code is compatible with the version of Node.js you are using. Use a tool like `nvm` to manage multiple Node.js versions.

### 6.5 Debugging Strategies

*   **MobX DevTools:** Use the MobX DevTools to inspect your application's state and track changes. The MobX DevTools is a browser extension that allows you to visualize your MobX stores and track changes in real-time.
*   **Logging:** Use logging to track the execution of your code and identify errors. Use a logging library like `debug` to simplify the logging process.
*   **Breakpoints:** Use breakpoints to pause the execution of your code and inspect its state. Breakpoints are a powerful tool for debugging complex code.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **VS Code:** Use VS Code as your IDE. VS Code has excellent support for JavaScript, TypeScript, and React, and it has a wide range of extensions that can improve your development workflow.
*   **ESLint:** Use ESLint to enforce coding standards and identify potential errors. ESLint is a linter that can be configured to enforce a wide range of coding standards.
*   **Prettier:** Use Prettier to automatically format your code. Prettier is a code formatter that can be configured to automatically format your code according to a set of rules.

### 7.2 Build Configuration

*   **Webpack:** Use Webpack to bundle your code. Webpack is a module bundler that can be used to bundle your code and its dependencies into a single file.
*   **Babel:** Use Babel to transpile your code to older versions of JavaScript. Babel is a transpiler that can be used to convert your code to older versions of JavaScript, which allows it to run on older browsers.
*   **TypeScript:** Use TypeScript to add static typing to your code. TypeScript is a superset of JavaScript that adds static typing to the language. This can help you catch errors early and improve the maintainability of your code.

### 7.3 Linting and Formatting

*   **ESLint:** Use ESLint to enforce coding standards and identify potential errors. Configure ESLint to use the recommended rules for React and MobX.
*   **Prettier:** Use Prettier to automatically format your code. Configure Prettier to use a consistent code style.
*   **Husky:** Use Husky to run linters and formatters before committing code. This ensures that all code committed to the repository meets the required standards.

### 7.4 Deployment Best Practices

*   **Continuous Integration/Continuous Deployment (CI/CD):** Implement a CI/CD pipeline to automate the deployment process. This ensures that your code is automatically tested and deployed whenever changes are made.
*   **Caching:** Use caching to improve the performance of your application. Cache static assets like images and JavaScript files to reduce the load on your server.
*   **Content Delivery Network (CDN):** Use a CDN to distribute your static assets across multiple servers. This improves the performance of your application by serving assets from the server closest to the user.

### 7.5 CI/CD Integration

*   **GitHub Actions:** Use GitHub Actions to automate your CI/CD pipeline. GitHub Actions is a CI/CD service that is integrated with GitHub. Use Jenkins, CircleCI, or other CI/CD tools.
*   **Automated Testing:** Automate your testing process to ensure that your code is thoroughly tested before it is deployed. Use a testing framework like Jest or Mocha to write automated tests.
*   **Automated Deployment:** Automate your deployment process to ensure that your code is deployed quickly and reliably. Use a deployment tool like `Capistrano` or `Deployer` to automate the deployment process.