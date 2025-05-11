---
description: This rule enforces best practices and coding standards for projects using the Ant Design (antd) UI library within React applications. It covers code organization, performance, security, testing, and common pitfalls to ensure maintainable and efficient applications.
globs: *.js,*.jsx,*.ts,*.tsx
---
# Ant Design (antd) Best Practices and Coding Standards

This document outlines the recommended best practices for developing React applications using the Ant Design (antd) UI library. Following these guidelines will lead to more maintainable, performant, and secure applications.

## 1. Code Organization and Structure

### 1.1. Directory Structure

-   **`src/components/`**: Contains reusable React components, including those utilizing antd components.
-   **`src/pages/`**: Contains components representing different application routes or pages.
-   **`src/layouts/`**: Contains layout components that provide a consistent structure across pages.
-   **`src/services/`**: Contains modules for interacting with APIs and handling data fetching.
-   **`src/utils/`**: Contains utility functions and helper modules.
-   **`src/styles/`**: Contains global styles, theme customizations, and component-specific stylesheets.
-   **`src/assets/`**: Contains static assets such as images, fonts, and icons.
-   **`src/context/`**: (Optional) If using context API, store all context definition files here.

Example:


my-app/
├── src/
│   ├── components/
│   │   ├── Button.jsx
│   │   ├── Input.jsx
│   │   └── ...
│   ├── pages/
│   │   ├── HomePage.jsx
│   │   ├── LoginPage.jsx
│   │   └── ...
│   ├── layouts/
│   │   ├── MainLayout.jsx
│   │   └── ...
│   ├── services/
│   │   ├── api.js
│   │   └── ...
│   ├── utils/
│   │   ├── date-formatter.js
│   │   └── ...
│   ├── styles/
│   │   ├── global.css
│   │   ├── theme.js
│   │   └── ...
│   ├── App.jsx
│   └── index.js
└── ...


### 1.2. File Naming Conventions

-   **Components**: Use PascalCase for component file names (e.g., `MyComponent.jsx`, `UserProfile.tsx`).
-   **Styles**: Use kebab-case for style file names (e.g., `my-component.css`, `user-profile.module.scss`).
-   **Modules**: Use camelCase for module file names (e.g., `api.js`, `dateFormatter.ts`).

### 1.3. Module Organization

-   Group related components, styles, and assets within the same directory.
-   Create separate modules for API interactions, data transformations, and utility functions.

### 1.4. Component Architecture

-   **Presentational Components**: Focus on rendering UI elements and receiving data via props.
-   **Container Components**: Handle data fetching, state management, and logic, passing data to presentational components.
-   Use functional components with hooks whenever possible for simplicity and reusability.

### 1.5. Code Splitting

-   Utilize React.lazy and Suspense to load components on demand, improving initial load time.
-   Split routes into separate chunks to minimize the initial bundle size.
-   Consider using dynamic imports for less frequently used components or modules.

Example:

jsx
import React, { Suspense } from 'react';

const MyComponent = React.lazy(() => import('./MyComponent'));

function MyPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <MyComponent />
    </Suspense>
  );
}


## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Higher-Order Components (HOCs)**: Use for cross-cutting concerns like authentication or data fetching.  Prefer hooks where possible.
-   **Render Props**:  An alternative to HOCs for sharing code between components.  Prefer hooks where possible.
-   **Compound Components**: Create reusable components with implicit state sharing (e.g., `Tabs` and `Tab` components).

### 2.2. Recommended Approaches

-   **Form Handling**: Use `antd`'s `Form` component for managing form state, validation, and submission.
-   **Data Display**: Leverage `antd`'s `Table`, `List`, and `Card` components for structured data presentation.
-   **Navigation**: Use `antd`'s `Menu` and `Breadcrumb` components for creating intuitive navigation.

### 2.3. Anti-patterns and Code Smells

-   **Direct DOM Manipulation**: Avoid directly manipulating the DOM; let React manage updates.
-   **Over-reliance on `any` type**: Using `any` in TypeScript defeats the purpose of static typing.  Provide explicit types.
-   **Mutating Props**: Treat props as read-only and avoid modifying them directly.
-   **Inline Styles**:  Keep styles in CSS files or use styled-components for better organization and maintainability. Prefer CSS Modules or Styled Components for component specific styles.

### 2.4. State Management

-   **Component State**: Use `useState` for simple, local component state.
-   **Context API**: Use for sharing state across multiple components without prop drilling.
-   **Redux/Zustand**: Consider for complex applications with global state and predictable state transitions.
-   **MobX**: Consider for complex applications where you want to observe changes in your data and derive calculations from that data.

### 2.5. Error Handling

-   **Try-Catch Blocks**: Use for handling synchronous errors.
-   **Error Boundaries**: Use to catch errors during rendering and prevent the entire application from crashing.
-   **Global Error Handling**: Implement a global error handler to log errors and provide user feedback.

Example (Error Boundary):

jsx
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Caught error: ', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

// Usage:
<ErrorBoundary>
  <MyComponent />
</ErrorBoundary>


## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Memoization**: Use `React.memo` to prevent unnecessary re-renders of components with the same props.
-   **Pure Components**: Extend `React.PureComponent` for class components to perform shallow prop comparisons.
-   **Virtualization**: Use `antd`'s `Table` and `List` components with virtualization for large datasets.
-   **Debouncing/Throttling**: Use for event handlers that trigger frequent updates (e.g., search input).

### 3.2. Memory Management

-   **Avoid Memory Leaks**: Properly clean up event listeners and timers in `useEffect` hooks.
-   **Release Resources**: Release unused objects and data structures to free up memory.

### 3.3. Rendering Optimization

-   **ShouldComponentUpdate**: Implement `shouldComponentUpdate` (for class components) or use `React.memo` (for functional components) to prevent unnecessary re-renders.
-   **Immutable Data**: Use immutable data structures to simplify change detection.

### 3.4. Bundle Size Optimization

-   **Modular Imports**: Import only the necessary components from `antd` to reduce bundle size (e.g., `import { Button } from 'antd';`).  Use `babel-plugin-import` for automatic modular imports.
-   **Tree Shaking**: Ensure your build process supports tree shaking to remove unused code.
-   **Code Splitting**: As mentioned earlier, split your code into smaller chunks to reduce the initial bundle size.

### 3.5. Lazy Loading

-   Use `React.lazy` and `Suspense` to load components on demand.
-   Implement lazy loading for images and other assets using libraries like `react-lazyload`.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

-   **Cross-Site Scripting (XSS)**: Prevent XSS by sanitizing user input and encoding output.
-   **Cross-Site Request Forgery (CSRF)**: Protect against CSRF attacks by implementing CSRF tokens.
-   **SQL Injection**: Avoid directly embedding user input in SQL queries; use parameterized queries or ORMs.

### 4.2. Input Validation

-   **Server-Side Validation**: Always validate user input on the server-side.
-   **Client-Side Validation**: Use `antd`'s `Form` component for client-side validation to provide immediate feedback to the user.
-   **Sanitize Input**: Sanitize user input to remove potentially harmful characters or code.

### 4.3. Authentication and Authorization

-   **Secure Authentication**: Use secure authentication mechanisms like JWT (JSON Web Tokens) or OAuth.
-   **Role-Based Access Control (RBAC)**: Implement RBAC to control access to different parts of the application based on user roles.

### 4.4. Data Protection

-   **Encryption**: Encrypt sensitive data both in transit and at rest.
-   **Data Masking**: Mask sensitive data in the UI to prevent unauthorized access.

### 4.5. Secure API Communication

-   **HTTPS**: Use HTTPS to encrypt communication between the client and the server.
-   **API Rate Limiting**: Implement rate limiting to prevent abuse and denial-of-service attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

-   Test individual components in isolation to ensure they function correctly.
-   Use testing libraries like Jest and React Testing Library.
-   Mock dependencies to isolate the component being tested.

### 5.2. Integration Testing

-   Test the interaction between multiple components or modules.
-   Use testing libraries like React Testing Library and Cypress.

### 5.3. End-to-End Testing

-   Test the entire application from the user's perspective.
-   Use testing frameworks like Cypress or Playwright.

### 5.4. Test Organization

-   Create a `tests/` directory at the root of your project.
-   Place test files alongside the components or modules they test (e.g., `MyComponent.test.jsx`).
-   Use descriptive test names to clearly indicate what is being tested.

### 5.5. Mocking and Stubbing

-   Use mocking libraries like Jest's `jest.mock()` to mock external dependencies.
-   Use stubbing to replace functions or methods with predefined behavior.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Not using modular imports**:  Importing the entire `antd` library can significantly increase bundle size.
-   **Ignoring TypeScript errors**: Failing to address TypeScript errors can lead to runtime issues.
-   **Not handling asynchronous operations correctly**:  Failing to handle promises or async/await can lead to unhandled rejections and unexpected behavior.
-   **Not localizing strings correctly**: Hardcoding strings instead of using `antd` i18n features. 

### 6.2. Edge Cases

-   **Handling different screen sizes and devices**: Ensuring responsive design using `antd` grid system.
-   **Accessibility**: Consider accessibility when using components, making sure to include `aria` attributes.
-   **Browser compatibility**: Test the app on various browsers (Chrome, Firefox, Safari, Edge, etc).

### 6.3. Version-Specific Issues

-   **Breaking changes**: Be aware of breaking changes when upgrading `antd` versions.
-   **Deprecated APIs**:  Avoid using deprecated APIs and migrate to the recommended alternatives.
-   **CSS class conflicts:** Potential issues with CSS specificity or conflicts with global styles. Use CSS Modules or Styled Components for more robust style isolation.

### 6.4. Compatibility Concerns

-   **React version**:  Ensure compatibility between `antd` and your React version.
-   **Other UI libraries**: Avoid conflicts with other UI libraries by using consistent styling and naming conventions.

### 6.5. Debugging

-   **Use browser developer tools**: Inspect the DOM, network requests, and console output.
-   **Use React DevTools**: Inspect the component tree, props, and state.
-   **Use logging and debugging statements**: Add `console.log` statements to trace the execution flow and inspect variable values.

## 7. Tooling and Environment

### 7.1. Recommended Tools

-   **IDE**: VS Code, WebStorm.
-   **Build Tool**: Webpack, Parcel, Rollup, esbuild.
-   **Testing Libraries**: Jest, React Testing Library, Cypress.
-   **Linting**: ESLint, Prettier.

### 7.2. Build Configuration

-   **Optimize for production**: Use production-specific build configurations to minimize bundle size and improve performance.
-   **Configure code splitting**: Set up code splitting to load components on demand.
-   **Enable tree shaking**: Ensure your build process supports tree shaking to remove unused code.

### 7.3. Linting and Formatting

-   **ESLint**: Use ESLint with recommended React and `antd` plugins to enforce coding standards and detect potential errors.
-   **Prettier**: Use Prettier to automatically format your code for consistency.
-   **Stylelint:** Use Stylelint to enforce consistent style practices.

### 7.4. Deployment

-   **Choose a hosting platform**:  Netlify, Vercel, AWS, Google Cloud, Azure.
-   **Configure environment variables**:  Set up environment variables for API keys, database credentials, and other sensitive information.
-   **Use a CDN**: Use a Content Delivery Network (CDN) to cache static assets and improve loading times.

### 7.5. CI/CD Integration

-   **Set up a CI/CD pipeline**: Use tools like Jenkins, Travis CI, CircleCI, or GitHub Actions to automate testing, building, and deployment.
-   **Automate testing**: Run unit, integration, and end-to-end tests in your CI/CD pipeline.
-   **Automate deployment**: Automate the deployment process to reduce manual effort and errors.