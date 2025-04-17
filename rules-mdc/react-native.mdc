---
description: This rule provides comprehensive best practices and coding standards for React Native development, covering code organization, performance, security, testing, and common pitfalls.
globs: *.js,*.jsx,*.ts,*.tsx
---
- Use TypeScript for type safety and improved code maintainability.
- Prefer functional components with hooks over class components for simplicity and reusability.
- Maintain a clear and consistent project structure for scalability and maintainability.

## 1. Code Organization and Structure:

- **Directory Structure:**
    - Adopt a feature-based or component-based structure. For example:
        
        src/
        ├── components/
        │   ├── Button/
        │   │   ├── Button.tsx
        │   │   ├── Button.styles.ts
        │   │   ├── Button.test.tsx
        │   ├── Input/
        │   └── ...
        ├── screens/
        │   ├── Home/
        │   │   ├── HomeScreen.tsx
        │   │   ├── HomeScreen.styles.ts
        │   │   ├── HomeScreen.test.tsx
        │   ├── Profile/
        │   └── ...
        ├── navigation/
        ├── services/
        ├── utils/
        ├── types/
        └── App.tsx
        
    - Separate concerns into distinct directories (e.g., components, screens, navigation, services, utils, types).

- **File Naming Conventions:**
    - Use descriptive names for files and components (e.g., `HomeScreen.tsx`, `useUserData.ts`).
    - Follow a consistent naming convention (e.g., PascalCase for components, camelCase for functions and variables).
    - Use `.styles.ts` for style files and `.test.tsx` for test files to keep components readable and organized.

- **Module Organization:**
    - Group related components and modules into logical units.
    - Use absolute imports and module aliases to avoid long relative paths (e.g., `@components/Button` instead of `../../../components/Button`).
    - Configure module aliases in `tsconfig.json` or `jsconfig.json`.

- **Component Architecture:**
    - Favor small, reusable components with a single responsibility (Single Responsibility Principle).
    - Use composition over inheritance to create complex components.
    - Consider using a UI library like React Native Paper or NativeBase for pre-built components.

- **Code Splitting Strategies:**
    - Implement lazy loading for screens or components that are not immediately needed.
    - Use `React.lazy` and `Suspense` to load components on demand.
    - Utilize dynamic imports for conditional loading of modules.

## 2. Common Patterns and Anti-patterns:

- **Design Patterns:**
    - **Higher-Order Components (HOCs):** Use HOCs for cross-cutting concerns like authentication or logging.
    - **Render Props:** Use render props to share code between React components.
    - **Hooks:** Use custom hooks to encapsulate logic and stateful behavior.

- **Recommended Approaches:**
    - **State Management:**
        - Use React Context for simple state management.
        - Use Redux, Zustand, or Jotai for complex state management.
        - Consider using Recoil for fine-grained state management.
    - **API Calls:**
        - Use `axios` or `fetch` for making API requests.
        - Create a service layer to handle API calls and data transformations.
    - **Navigation:**
        - Use React Navigation for managing app navigation.
        - Define navigation stacks and routes in a separate module.

- **Anti-patterns and Code Smells:**
    - **Long Component Files:** Break down large components into smaller, more manageable pieces.
    - **Deeply Nested Components:** Avoid excessive nesting, which can impact performance.
    - **Mutating State Directly:** Always use `setState` or a state management library to update state.
    - **Unnecessary Re-renders:** Optimize components to prevent unnecessary re-renders.
    - **Global Styles:** Avoid using global styles, as they can lead to conflicts and make it difficult to maintain the application.

- **State Management Best Practices:**
    - Choose a state management solution that fits the complexity of your application.
    - Keep state minimal and derive values when possible.
    - Use selectors to access state and memoize computed values.

- **Error Handling Patterns:**
    - Use `try...catch` blocks to handle errors gracefully.
    - Implement a global error handler to catch unhandled exceptions.
    - Log errors to a remote monitoring service.

## 3. Performance Considerations:

- **Optimization Techniques:**
    - **Memoization:** Use `React.memo` to memoize components and prevent unnecessary re-renders.
    - **Pure Components:** Extend `React.PureComponent` for components that only depend on props.
    - **Debouncing and Throttling:** Use debouncing and throttling to limit the frequency of function calls.
    - **Virtualization:** Use `FlatList` or `SectionList` for rendering large lists of data.

- **Memory Management:**
    - Avoid memory leaks by properly cleaning up event listeners and timers.
    - Use `useCallback` and `useMemo` to prevent creating new functions and objects on every render.

- **Rendering Optimization:**
    - Minimize the number of re-renders by optimizing component updates.
    - Use `shouldComponentUpdate` (for class components) or `React.memo` to control re-renders.
    - Avoid using inline styles, as they are re-created on every render.

- **Bundle Size Optimization:**
    - Use code splitting to reduce the initial bundle size.
    - Remove unused code and dependencies.
    - Use a bundler like Metro or Webpack with tree shaking enabled.
    - Compress images and other assets.

- **Lazy Loading Strategies:**
    - Implement lazy loading for images and other assets using `React.lazy` and `Suspense`.
    - Use dynamic imports to load modules on demand.

## 4. Security Best Practices:

- **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
    - **SQL Injection:** Use parameterized queries to prevent SQL injection attacks.
    - **Cross-Site Request Forgery (CSRF):** Implement CSRF protection tokens.
    - **Man-in-the-Middle (MITM) Attacks:** Use HTTPS to encrypt communication.

- **Input Validation:**
    - Validate user input on both the client and server sides.
    - Use regular expressions or validation libraries to enforce input constraints.

- **Authentication and Authorization Patterns:**
    - Use a secure authentication protocol like OAuth 2.0 or JWT.
    - Implement role-based access control (RBAC) to restrict access to sensitive resources.

- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure storage mechanisms for storing API keys and other secrets.

- **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement API rate limiting to prevent abuse.
    - Validate API responses to prevent data injection attacks.

## 5. Testing Approaches:

- **Unit Testing Strategies:**
    - Write unit tests for individual components and modules.
    - Use a testing framework like Jest or Mocha.
    - Mock dependencies to isolate the component being tested.

- **Integration Testing:**
    - Write integration tests to verify the interaction between components and modules.
    - Test the integration with external APIs and services.

- **End-to-End Testing:**
    - Write end-to-end tests to verify the entire application flow.
    - Use a testing framework like Detox or Appium.

- **Test Organization:**
    - Organize tests into separate directories based on component or module.
    - Use descriptive names for test files and test cases.

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components and control their behavior during testing.
    - Use a mocking library like Jest or Sinon.

## 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes:**
    - **Directly Mutating State:** Always use `setState` or a state management library to update state.
    - **Ignoring Platform Differences:** Test your application on both iOS and Android devices.
    - **Over-Optimizing:** Optimize only when necessary, as premature optimization can lead to complex code.
    - **Not Using a Debugger:** Utilize the React Native debugger for efficient debugging.

- **Edge Cases:**
    - **Handling Device Orientation Changes:** Implement logic to handle device orientation changes gracefully.
    - **Handling Network Connectivity Issues:** Implement error handling for network connectivity issues.
    - **Handling Different Screen Sizes and Densities:** Design your UI to adapt to different screen sizes and densities.

- **Version-Specific Issues:**
    - Be aware of breaking changes in React Native and its dependencies.
    - Test your application with different versions of React Native.

- **Compatibility Concerns:**
    - Ensure that your application is compatible with the target operating systems and devices.
    - Use polyfills to support older browsers and devices.

- **Debugging Strategies:**
    - Use the React Native debugger to inspect the component tree and state.
    - Use the console to log messages and debug code.
    - Use a remote debugging tool to debug on a physical device.

## 7. Tooling and Environment:

- **Recommended Development Tools:**
    - **VS Code:** Use VS Code with the React Native Tools extension for debugging and code completion.
    - **React Native CLI:** Use the React Native CLI for creating and managing React Native projects.
    - **Expo CLI:** Use the Expo CLI for developing and testing React Native applications without native code.
    - **Android Studio:** Use Android Studio for building and debugging Android applications.
    - **Xcode:** Use Xcode for building and debugging iOS applications.

- **Build Configuration:**
    - Use a build system like Gradle (Android) or CocoaPods (iOS) to manage dependencies.
    - Configure build variants for different environments (e.g., development, staging, production).

- **Linting and Formatting:**
    - Use ESLint and Prettier to enforce code style and catch potential errors.
    - Configure ESLint and Prettier to automatically format code on save.

- **Deployment Best Practices:**
    - Use a continuous integration and continuous deployment (CI/CD) pipeline to automate the deployment process.
    - Use a service like App Center or Bitrise for building and deploying React Native applications.

- **CI/CD Integration:**
    - Integrate your code repository with a CI/CD service like GitHub Actions or CircleCI.
    - Configure the CI/CD pipeline to run tests and build the application on every commit.