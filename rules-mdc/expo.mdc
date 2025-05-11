---
description: This rule provides comprehensive best practices and coding standards for Expo projects, covering code organization, performance, security, testing, and common pitfalls to ensure maintainable and high-quality applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
- **Code Organization and Structure**
  - **Directory Structure:**
    - Adopt a feature-based directory structure, grouping related files (components, hooks, styles, tests) within feature folders.
    - Examples:
      
      src/
      ├── components/
      │   ├── Button/
      │   │   ├── Button.tsx
      │   │   ├── Button.styles.ts
      │   │   └── Button.test.tsx
      │   └── ...
      ├── screens/
      │   ├── Home/
      │   │   ├── HomeScreen.tsx
      │   │   ├── HomeScreen.styles.ts
      │   │   └── HomeScreen.test.tsx
      │   └── ...
      ├── navigation/
      │   ├── AppNavigator.tsx
      │   └── ...
      ├── services/
      │   ├── api.ts
      │   └── ...
      ├── utils/
      │   ├── helpers.ts
      │   └── ...
      ├── App.tsx
      └── index.tsx
      
  - **File Naming Conventions:**
    - Use descriptive and consistent names for files and directories.
    - Component files: `ComponentName.tsx` or `ComponentName.js`
    - Style files: `ComponentName.styles.ts` or `ComponentName.styles.js`
    - Hook files: `useHookName.ts` or `useHookName.js`
    - Test files: `ComponentName.test.tsx` or `ComponentName.test.js`
  - **Module Organization:**
    - Group related components, hooks, and utilities into modules.
    - Use `index.ts` (or `index.js`) files to export members from a module for easier importing.
    - Example:
      
      // src/components/Button/index.ts
      export { default as Button } from './Button';
      
  - **Component Architecture:**
    - Favor functional components with hooks for managing state and side effects.
    - Separate concerns by creating presentational (UI) and container (logic) components.
    - Use component composition to build complex UIs from smaller, reusable components.
  - **Code Splitting Strategies:**
    - Implement lazy loading for screens or components that are not immediately needed.
    - Utilize `React.lazy` and `Suspense` for dynamic imports.
    - Expo supports dynamic imports; leverage them to reduce initial bundle size.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - **Higher-Order Components (HOCs):** Use cautiously; prefer hooks or render props for better composability.
    - **Render Props:** A pattern for sharing code between React components using a prop whose value is a function.
    - **Compound Components:** Create components that implicitly share state (e.g., Tabs and Tab).
  - **Recommended Approaches:**
    - Use Expo's APIs for accessing device features (camera, location, notifications) instead of relying on native modules directly where possible.
    - Leverage Expo's managed workflow for a smoother development experience.
  - **Anti-patterns:**
    - Directly manipulating the DOM (use React's state and props instead).
    - Writing complex logic directly within components (extract into hooks or utility functions).
    - Neglecting error handling and edge cases.
  - **State Management:**
    - Use React Context for simple state management.
    - Consider libraries like Zustand, Redux, or Jotai for more complex state management needs.
    - Leverage `useReducer` for managing complex state transitions.
  - **Error Handling:**
    - Implement try-catch blocks to handle potential errors.
    - Use error boundary components to catch errors during rendering.
    - Log errors using a centralized logging service (e.g., Sentry).

- **Performance Considerations**
  - **Optimization Techniques:**
    - Use `React.memo` to prevent unnecessary re-renders of components.
    - Implement `shouldComponentUpdate` (for class components) or `useMemo` and `useCallback` (for functional components) to optimize rendering.
    - Debounce or throttle event handlers to reduce the frequency of updates.
  - **Memory Management:**
    - Avoid creating large arrays or objects in component state if not necessary.
    - Clean up event listeners and timers when components unmount.
    - Release resources when they are no longer needed.
  - **Rendering Optimization:**
    - Virtualize long lists using `FlatList` or `SectionList` to improve scrolling performance.
    - Optimize image sizes and formats to reduce loading times.
    - Use the `useNativeDriver: true` prop in animations for better performance.
  - **Bundle Size Optimization:**
    - Use code splitting to reduce the initial bundle size.
    - Remove unused code and dependencies.
    - Optimize images and other assets.
  - **Lazy Loading:**
    - Implement lazy loading for images and other resources that are not immediately visible.
    - Use the `<Image>` component from `react-native` or `expo-image` with placeholder and loading indicators.

- **Security Best Practices**
  - **Common Vulnerabilities:**
    - Cross-Site Scripting (XSS) - Not typically a risk in React Native/Expo, but be mindful when rendering user-provided content via WebView.
    - Data injection - Protect against SQL or command injection if interacting with backend systems.
    - Insecure data storage - Never store sensitive information (API keys, credentials) directly in the app code.
  - **Input Validation:**
    - Validate user input on both the client and server sides.
    - Use appropriate data types and formats.
    - Sanitize user input to prevent injection attacks.
  - **Authentication and Authorization:**
    - Use secure authentication mechanisms (e.g., OAuth 2.0, JWT).
    - Implement proper authorization to restrict access to sensitive data and functionality.
    - Store authentication tokens securely (e.g., using Expo SecureStore).
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all API communication.
    - Avoid storing sensitive information in logs.
  - **Secure API Communication:**
    - Use HTTPS for all API requests.
    - Validate API responses.
    - Implement rate limiting to prevent abuse.

- **Testing Approaches**
  - **Unit Testing:**
    - Write unit tests for individual components, hooks, and utility functions.
    - Use testing libraries like Jest and React Testing Library.
    - Mock dependencies to isolate units of code.
  - **Integration Testing:**
    - Test the interactions between multiple components or modules.
    - Use tools like Detox or Appium for end-to-end testing on real devices or emulators.
  - **End-to-end Testing:**
    - Simulate user interactions to test the entire application flow.
    - Use tools like Detox or Appium for end-to-end testing on real devices or emulators.
  - **Test Organization:**
    - Organize tests in a clear and maintainable structure.
    - Group tests by component or module.
    - Use descriptive names for test cases.
  - **Mocking and Stubbing:**
    - Use mocking to isolate units of code and simulate dependencies.
    - Use stubbing to replace complex or external dependencies with simplified versions.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - Forgetting to clean up event listeners or timers.
    - Mutating state directly instead of using `setState` or state update functions.
    - Over-rendering components due to unnecessary state updates.
  - **Edge Cases:**
    - Handling different screen sizes and orientations.
    - Dealing with slow network connections or API failures.
    - Managing application state during interruptions (e.g., phone calls).
  - **Version-specific Issues:**
    - Be aware of breaking changes in React Native and Expo SDK updates.
    - Test your application thoroughly after upgrading dependencies.
  - **Compatibility Concerns:**
    - Test your application on different devices and operating systems.
    - Use platform-specific code only when necessary.
  - **Debugging Strategies:**
    - Use the React Native debugger or Chrome DevTools for debugging.
    - Use console.log statements or debugging tools to inspect variables and application state.
    - Use error boundary components to catch and log errors.

- **Tooling and Environment**
  - **Recommended Tools:**
    - VS Code with Expo and ESLint extensions
    - Expo CLI for managing Expo projects.
    - React Native Debugger or Chrome DevTools for debugging.
  - **Build Configuration:**
    - Use environment variables to configure different build environments (development, staging, production).
    - Configure your build process to optimize assets and reduce bundle size.
  - **Linting and Formatting:**
    - Use ESLint with the Airbnb or Standard style guide.
    - Use Prettier to automatically format your code.
    - Configure your editor to automatically run ESLint and Prettier on save.
  - **Deployment:**
    - Use Expo Application Services (EAS) for building and deploying your application.
    - Follow Expo's deployment documentation for best practices.
  - **CI/CD Integration:**
    - Integrate your application with a CI/CD pipeline (e.g., GitHub Actions, CircleCI).
    - Automate testing, linting, and deployment processes.

- **Expo Specific Recommendations**
  - **Use Expo Modules:** Prefer Expo modules for accessing native features over community libraries when available, as they are guaranteed to be compatible and well-maintained within the Expo ecosystem.
  - **EAS Build:**  Use Expo Application Services (EAS) for building native binaries.  This simplifies the build process and handles much of the complexity of native builds.
  - **Expo Updates:** Implement Expo Updates for over-the-air updates.  This allows for quicker iteration and bug fixes without requiring users to download a new version from the app store.
  - **Managed Workflow:** Stick to the managed workflow whenever possible.  It handles much of the native configuration and simplifies development.
  - **Expo SecureStore:** Use `expo-secure-store` to securely store sensitive data like API keys and tokens.
  - **Environment Variables:** Use `.env` files and `expo-constants` to manage environment-specific configurations.
  - **Asset Management:**  Place assets (images, fonts) in the `assets` folder and use the `Image` and `Font` components to load them.

- **TypeScript Best Practices**
  - **Strict Mode:** Enable TypeScript's strict mode (`strict: true` in `tsconfig.json`) for enhanced type safety.
  - **Explicit Types:**  Explicitly define types for function parameters, return values, and variables.
  - **Interfaces vs. Types:** Use interfaces for defining contract between objects, and types for defining data structures.
  - **Utility Types:** Leverage TypeScript's utility types (e.g., `Partial`, `Required`, `Readonly`, `Pick`, `Omit`) for more concise and maintainable code.
  - **Type Guards:** Use type guards to narrow down types within conditional blocks.
  - **Async/Await:** Use `async/await` for asynchronous operations to improve code readability and avoid callback hell.

- **Resources**
  - Expo Documentation: [https://docs.expo.dev/](https://docs.expo.dev/)
  - React Native Documentation: [https://reactnative.dev/](https://reactnative.dev/)
  - Expo JavaScript Style Guide: [https://github.com/expo/expo/blob/main/guides/Expo%20JavaScript%20Style%20Guide.md](https://github.com/expo/expo/blob/main/guides/Expo%20JavaScript%20Style%20Guide.md)