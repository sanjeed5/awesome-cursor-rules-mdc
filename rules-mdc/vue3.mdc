---
description: This rule provides best practices and coding standards for Vue 3 projects, covering code organization, performance, security, testing, tooling, and common pitfalls to ensure maintainable and efficient applications. It aims to guide developers in writing high-quality Vue 3 code.
globs: *.vue
---
- **Code Organization and Structure**:
  - **Directory Structure**: Adopt a feature-based directory structure. Group related files (components, stores, utilities) within feature-specific directories rather than separating by file type. This enhances maintainability and discoverability.
    - Example:
      
      src/
        components/
          MyComponent.vue
          ...
        views/
          MyView.vue
          ...
        features/
          user-profile/
            components/
              UserProfileCard.vue
            composables/
              useUserProfileData.js
            store/
              userProfile.js
          ...
      
  - **File Naming Conventions**: Use PascalCase for component file names (e.g., `MyComponent.vue`). Use camelCase for variable and function names (e.g., `myVariable`, `myFunction`). Use kebab-case for component selectors in templates (e.g., `<my-component>`).
  - **Module Organization**: Utilize ES modules (`import`/`export`) for modularity and code reusability. Group related functions and components into modules.
  - **Component Architecture**: Favor a component-based architecture. Design components to be small, reusable, and composable. Use props for data input and events for data output. Consider using a component library (e.g., Vuetify, Element Plus) for pre-built components.
  - **Code Splitting Strategies**: Implement lazy loading for components and routes to reduce initial bundle size. Use dynamic imports for on-demand loading of modules.
    - Example:
      javascript
      // Route-based code splitting
      const routes = [
        {
          path: '/about',
          component: () => import('./views/About.vue')
        }
      ]
      

- **Common Patterns and Anti-patterns**:
  - **Design Patterns**: Apply common design patterns such as composition API, provider/inject, and observer pattern where applicable.
    - **Composition API**: Organize component logic into composable functions for reusability and maintainability.
    - **Provider/Inject**: Use `provide` and `inject` to share data between components without prop drilling.
  - **Recommended Approaches**: Utilize `v-model` for two-way data binding, computed properties for derived state, and watchers for side effects. Use the Composition API for enhanced code organization and reusability.
  - **Anti-patterns and Code Smells**: Avoid directly mutating props. Avoid excessive use of global variables. Avoid complex logic within templates. Avoid tight coupling between components. Avoid over-engineering solutions.
  - **State Management**: Choose a state management solution (e.g., Vuex, Pinia) for complex applications.  Favor Pinia for Vue 3 due to its simpler API and improved TypeScript support. Decouple components from state management logic using actions and mutations.
  - **Error Handling**: Implement global error handling using `app.config.errorHandler`. Use `try...catch` blocks for handling synchronous errors. Utilize `Promise.catch` for handling asynchronous errors. Provide user-friendly error messages.
    - Example:
      javascript
      // Global error handler
      app.config.errorHandler = (err, vm, info) => {
        console.error('Global error:', err, info);
        // Report error to server or display user-friendly message
      }
      

- **Performance Considerations**:
  - **Optimization Techniques**: Use `v-once` for static content. Use `v-memo` to memoize parts of the template. Use `key` attribute for `v-for` loops to improve rendering performance.
  - **Memory Management**: Avoid creating memory leaks by properly cleaning up event listeners and timers. Use `onBeforeUnmount` lifecycle hook to release resources.
  - **Rendering Optimization**: Use virtual DOM efficiently. Minimize unnecessary re-renders by using `ref` and `reactive` appropriately. Use `shouldUpdate` hook in functional components to control updates.
  - **Bundle Size Optimization**: Use code splitting, tree shaking, and minification to reduce bundle size. Remove unused dependencies. Use smaller alternative libraries where possible.
  - **Lazy Loading**: Implement lazy loading for images, components, and routes. Use `IntersectionObserver` API for lazy loading images.

- **Security Best Practices**:
  - **Common Vulnerabilities**: Prevent Cross-Site Scripting (XSS) attacks by sanitizing user input. Prevent Cross-Site Request Forgery (CSRF) attacks by using CSRF tokens. Prevent SQL injection attacks by using parameterized queries.
  - **Input Validation**: Validate user input on both client-side and server-side. Use appropriate data types and formats. Escape special characters.
  - **Authentication and Authorization**: Implement secure authentication and authorization mechanisms. Use HTTPS to encrypt communication. Store passwords securely using hashing and salting.
  - **Data Protection**: Protect sensitive data using encryption. Avoid storing sensitive data in client-side storage. Follow privacy best practices.
  - **Secure API Communication**: Use HTTPS for API communication. Validate API responses. Implement rate limiting to prevent abuse.

- **Testing Approaches**:
  - **Unit Testing**: Write unit tests for individual components, functions, and modules. Use Jest or Vitest as a test runner. Mock dependencies to isolate units of code.
  - **Integration Testing**: Write integration tests to verify the interaction between components and modules. Use Vue Test Utils for component testing.
  - **End-to-End Testing**: Write end-to-end tests to simulate user interactions and verify the application's overall functionality. Use Cypress or Playwright for end-to-end testing.
  - **Test Organization**: Organize tests into separate directories based on the component or module being tested. Use descriptive test names.
  - **Mocking and Stubbing**: Use mocks and stubs to isolate units of code and simulate dependencies. Use `jest.mock` or `vi.mock` for mocking modules.

- **Common Pitfalls and Gotchas**:
  - **Frequent Mistakes**: Forgetting to register components. Incorrectly using `v-if` and `v-show`. Mutating props directly. Not handling asynchronous operations correctly. Ignoring error messages.
  - **Edge Cases**: Handling empty arrays or objects. Dealing with browser compatibility issues. Managing state in complex components.
  - **Version-Specific Issues**: Being aware of breaking changes between Vue 2 and Vue 3. Using deprecated APIs.
  - **Compatibility Concerns**: Ensuring compatibility with different browsers and devices. Testing on different screen sizes and resolutions.
  - **Debugging Strategies**: Using Vue Devtools for debugging. Using `console.log` statements for inspecting variables. Using a debugger for stepping through code.

- **Tooling and Environment**:
  - **Recommended Development Tools**: Use VS Code with the Volar extension for Vue 3 development. Use Vue CLI or Vite for project scaffolding. Use Vue Devtools for debugging.
  - **Build Configuration**: Configure Webpack or Rollup for building the application. Optimize build settings for production. Use environment variables for configuration.
  - **Linting and Formatting**: Use ESLint with the `eslint-plugin-vue` plugin for linting Vue code. Use Prettier for code formatting. Configure linting and formatting rules to enforce code style.
  - **Deployment Best Practices**: Use a CDN for serving static assets. Use server-side rendering (SSR) or pre-rendering for improved SEO and performance. Deploy to a reliable hosting platform.
  - **CI/CD Integration**: Integrate linting, testing, and building into the CI/CD pipeline. Use automated deployment tools. Monitor application performance and errors.

- **Additional Best Practices**: 
  - **Accessibility (A11y)**: Ensure components are accessible by using semantic HTML, providing ARIA attributes where necessary, and testing with screen readers. 
  - **Internationalization (i18n)**: Implement i18n from the start if multilingual support is required. Use a library like `vue-i18n` to manage translations. 
  - **Documentation**: Document components and composables using JSDoc or similar tools. Generate documentation automatically using tools like Storybook. 

- **Vue 3 Specific Recommendations**:
    - **TypeScript**: Use TypeScript for improved type safety and code maintainability. Define component props and emits with type annotations.
    - **Teleport**: Use the `Teleport` component to render content outside the component's DOM hierarchy, useful for modals and tooltips.
    - **Suspense**: Use the `Suspense` component to handle asynchronous dependencies gracefully, providing fallback content while waiting for data to load.

- **Naming Conventions**:
    - Components: PascalCase (e.g., `MyComponent.vue`)
    - Variables/Functions: camelCase (e.g., `myVariable`, `myFunction`)
    - Props/Events: camelCase (e.g., `myProp`, `myEvent`)
    - Directives: kebab-case (e.g., `v-my-directive`)

- **Composition API Best Practices**:
  - **Reactive Refs**: Use `ref` for primitive values and `reactive` for objects. 
  - **Readonly Refs**: Use `readonly` to prevent accidental mutations of reactive data.
  - **Computed Properties**: Use `computed` for derived state and avoid complex logic within templates.
  - **Lifecycle Hooks**: Use `onMounted`, `onUpdated`, `onUnmounted`, etc., to manage component lifecycle events.
  - **Watchers**: Use `watch` for reacting to reactive data changes and performing side effects.