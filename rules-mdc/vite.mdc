---
description: This rule provides comprehensive best practices, coding standards, and guidelines for developing applications using Vite, covering aspects from code organization and performance to security and testing.
globs: *.js,*.jsx,*.ts,*.tsx,*.vue,*.svelte
---
- **Introduction:**
  - This document outlines best practices for developing applications using Vite, a fast and opinionated build tool that aims to provide a better development experience.

- **Prerequisites:**
  - Ensure Node.js and npm/yarn/pnpm are installed.
  - Familiarity with JavaScript/TypeScript, HTML, and CSS.

- **Code Organization and Structure:**
  - **Directory Structure:**
    - Adopt a modular structure based on features or components.
    
    src/
    ├── components/
    │   ├── Button/
    │   │   ├── Button.tsx
    │   │   ├── Button.module.css
    │   │   └── Button.test.tsx
    │   ├── Input/
    │   │   └── ...
    ├── pages/
    │   ├── Home.tsx
    │   ├── About.tsx
    │   └── ...
    ├── services/
    │   ├── api.ts
    │   └── ...
    ├── utils/
    │   ├── helpers.ts
    │   └── ...
    ├── App.tsx
    ├── main.tsx
    └── vite-env.d.ts
    
  - **File Naming Conventions:**
    - Use descriptive and consistent names.
    - Component files: `ComponentName.tsx` or `component-name.tsx`.
    - Style files: `ComponentName.module.css` or `component-name.module.css`.
    - Test files: `ComponentName.test.tsx` or `component-name.test.tsx`.
  - **Module Organization:**
    - Group related files into modules or folders.
    - Use `index.ts` (barrel files) to simplify imports.
    typescript
    // src/components/Button/index.ts
    export { default as Button } from './Button';
    
  - **Component Architecture:**
    - Favor small, reusable components.
    - Utilize functional components and hooks in React (or equivalent in Vue/Svelte).
    - Separate concerns: presentational vs. container components.
  - **Code Splitting Strategies:**
    - Use dynamic imports (`import()`) for lazy loading.
    - Split routes using `React.lazy` or Vue's dynamic component feature.
    - Configure Vite's `rollupOptions.output.manualChunks` for fine-grained control.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Higher-Order Components (HOCs):**  Carefully consider alternatives like render props or hooks for better composability.
    - **Render Props:**  Useful for sharing logic between components, but can lead to deeply nested structures.
    - **Hooks:**  Promote code reuse and simplify component logic.
  - **Recommended Approaches:**
    - Use environment variables for configuration.
    - Implement a consistent API client for data fetching.
    - Centralize state management using libraries like Redux, Zustand, or Vuex.
  - **Anti-patterns:**
    - Avoid deeply nested component trees without proper optimization.
    - Don't mutate state directly; use setState or Vue's reactivity system.
    - Overusing global styles; prefer CSS modules or styled components.
  - **State Management:**
    - Choose a state management solution based on application complexity.
    - Use Redux for complex state management with predictable state transitions and time travel debugging.
    - Consider Zustand or Jotai for simpler state management with a smaller bundle size.
    - For Vue, Vuex or Pinia are popular choices.
  - **Error Handling:**
    - Implement global error boundaries to catch unhandled exceptions.
    - Use try-catch blocks for local error handling.
    - Log errors to a central error tracking service (e.g., Sentry, Rollbar).

- **Performance Considerations:**
  - **Optimization Techniques:**
    - Use production-ready code minification and bundling.
    - Optimize images and other assets using tools like `imagemin` or Vite plugins.
  - **Memory Management:**
    - Avoid memory leaks by properly cleaning up event listeners and subscriptions.
    - Use `useEffect` with a cleanup function in React (or `onUnmounted` in Vue).
  - **Rendering Optimization:**
    - Use memoization techniques (`React.memo`, `useMemo`, `shouldComponentUpdate`) to prevent unnecessary re-renders.
    - Virtualize large lists using libraries like `react-window` or `react-virtualized`.
  - **Bundle Size Optimization:**
    - Analyze bundle size using `rollup-plugin-visualizer` or similar tools.
    - Remove unused code using tree shaking.
    - Use code splitting to load only necessary code.
  - **Lazy Loading:**
    - Lazy load components and images that are not immediately visible.
    - Use `IntersectionObserver` to trigger loading when elements enter the viewport.

- **Security Best Practices:**
  - **Common Vulnerabilities:**
    - Cross-Site Scripting (XSS): Sanitize user input to prevent XSS attacks.
    - Cross-Site Request Forgery (CSRF): Use CSRF tokens to protect against CSRF attacks.
    - Injection Attacks: Validate and sanitize input to prevent SQL injection and other injection attacks.
  - **Input Validation:**
    - Validate all user input on both the client and server side.
    - Use a library like `yup` or `joi` for schema validation.
  - **Authentication and Authorization:**
    - Use a secure authentication and authorization mechanism (e.g., OAuth 2.0, JWT).
    - Store passwords securely using bcrypt or Argon2.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all communication.
  - **Secure API Communication:**
    - Implement proper CORS configuration to prevent unauthorized access to your API.
    - Rate limit API requests to prevent abuse.

- **Testing Approaches:**
  - **Unit Testing:**
    - Write unit tests for individual components and functions.
    - Use testing libraries like Jest, Mocha, or Vitest.
    - Mock dependencies to isolate units under test.
  - **Integration Testing:**
    - Test the interaction between different parts of your application.
    - Use testing libraries like React Testing Library or Vue Test Utils.
  - **End-to-End Testing:**
    - Test the entire application from the user's perspective.
    - Use tools like Cypress or Playwright.
  - **Test Organization:**
    - Organize tests into folders based on features or components.
    - Use descriptive test names.
  - **Mocking and Stubbing:**
    - Use mocks and stubs to isolate units under test.
    - Avoid over-mocking; test the actual implementation whenever possible.

- **Common Pitfalls and Gotchas:**
  - **Frequent Mistakes:**
    - Improperly handling asynchronous operations.
    - Neglecting accessibility considerations.
    - Using outdated dependencies.
  - **Edge Cases:**
    - Handling different screen sizes and devices.
    - Supporting internationalization and localization.
    - Dealing with slow network connections.
  - **Version-Specific Issues:**
    - Be aware of breaking changes in Vite and its plugins.
    - Pin dependencies to specific versions to avoid unexpected issues.
  - **Compatibility Concerns:**
    - Test your application in different browsers and devices.
    - Use polyfills to support older browsers.
  - **Debugging Strategies:**
    - Use browser developer tools to inspect the DOM, network requests, and console output.
    - Use debugging tools like `debugger` or `console.log`.

- **Tooling and Environment:**
  - **Recommended Tools:**
    - VS Code with extensions like ESLint, Prettier, and TypeScript.
    - Chrome DevTools or Firefox Developer Tools.
    - npm/yarn/pnpm for package management.
  - **Build Configuration:**
    - Configure Vite using `vite.config.ts` or `vite.config.js`.
    - Customize build options like `outDir`, `assetsDir`, and `rollupOptions`.
  - **Linting and Formatting:**
    - Use ESLint with recommended rulesets (e.g., `eslint:recommended`, `plugin:react/recommended`).
    - Use Prettier for code formatting.
    - Configure ESLint and Prettier to work together.
  - **Deployment Best Practices:**
    - Deploy to a CDN for optimal performance.
    - Use environment variables for configuration.
    - Set up proper caching headers.
  - **CI/CD Integration:**
    - Integrate with a CI/CD pipeline for automated testing and deployment.
    - Use tools like GitHub Actions, GitLab CI, or CircleCI.

- **TypeScript Best Practices (when using TypeScript):**
  - **Strict Type-Checking:**
    - Enable strict type-checking options in `tsconfig.json` (e.g., `strict: true`, `noImplicitAny: true`, `strictNullChecks: true`).
  - **Typing Props and State:**
    - Use interfaces or types to define the shape of props and state.
    typescript
    interface ButtonProps {
      label: string;
      onClick: () => void;
    }

    const Button: React.FC<ButtonProps> = ({ label, onClick }) => {
      return <button onClick={onClick}>{label}</button>;
    };
    

- **ESLint Configuration (Example):**
  javascript
  module.exports = {
    env: {
      browser: true,
      es2021: true,
      node: true,
    },
    extends: [
      'eslint:recommended',
      'plugin:react/recommended',
      'plugin:@typescript-eslint/recommended',
      'prettier',
    ],
    parser: '@typescript-eslint/parser',
    parserOptions: {
      ecmaFeatures: {
        jsx: true,
      },
      ecmaVersion: 12,
      sourceType: 'module',
    },
    plugins: ['react', '@typescript-eslint', 'prettier'],
    rules: {
      'prettier/prettier': 'error',
      'react/react-in-jsx-scope': 'off',
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  };
  

- **Conclusion:**
  - Following these best practices will help you build efficient, maintainable, and secure applications with Vite.  Continuously review and update your practices as the library and ecosystem evolve.