---
description: This rule provides comprehensive best practices and coding standards for developing Astro projects. It covers code organization, performance, security, testing, and common pitfalls.
globs: *.astro
---
# Astro Library Best Practices and Coding Standards

This document outlines the recommended best practices and coding standards for developing Astro projects to ensure maintainability, performance, and security.

## 1. Code Organization and Structure

### 1.1 Project Structure

-   **`src/` Directory:**  This directory contains the core source code of your Astro project.
    -   **`src/pages/`:**  This directory is crucial for routing. Every `.astro`, `.md`, or `.mdx` file in this directory automatically becomes a route. Use a clear and consistent naming convention for your pages (e.g., `about.astro`, `blog/index.astro`, `blog/[slug].astro`).
    -   **`src/components/`:**  Store reusable UI components in this directory. Organize components into subdirectories based on their functionality or feature area (e.g., `src/components/Header/`, `src/components/Card/`).
    -   **`src/layouts/`:** Layouts define the overall structure of your pages (e.g., header, footer, navigation).  Use layouts to maintain a consistent look and feel across your website.
    -   **`src/content/`:**  (Recommended for content-driven sites) Use the Content Collections feature to manage structured content like blog posts, documentation, or product listings.  Define schemas for your content types to ensure data consistency.
    -   **`src/styles/`:** Store global styles and CSS variables in this directory. Consider using CSS Modules or a CSS-in-JS solution for component-specific styling.
    -   **`src/scripts/`:**  Place client-side JavaScript files in this directory. Use modules and avoid polluting the global scope.
    -   **`src/assets/`:**  Store static assets like images, fonts, and other media files in this directory.
-   **`public/` Directory:** This directory contains static assets that don't need processing, such as `robots.txt`, `favicon.ico`, and other files that should be served directly.  Avoid placing images that require optimization in this directory; use the `src/assets/` directory instead.
-   **`astro.config.mjs`:** This file contains the Astro configuration options, including integrations, build settings, and more.  Keep this file well-organized and documented.
-   **`.env`:** Store environment variables in this file.  Use a library like `dotenv` to load environment variables into your application.

### 1.2 Component Design

-   **Atomic Design Principles:** Consider using Atomic Design principles to structure your components into atoms, molecules, organisms, templates, and pages.  This promotes reusability and maintainability.
-   **Single Responsibility Principle:** Each component should have a single, well-defined responsibility.  Avoid creating large, complex components that do too much.
-   **Props and Slots:** Use props to pass data to components and slots to allow components to accept children.  Define prop types using TypeScript to ensure type safety.

## 2. Common Patterns and Anti-patterns

### 2.1 UI Framework Integration

-   **Island Architecture:** Embrace Astro's island architecture to minimize JavaScript and improve performance.  Only hydrate interactive components.
-   **Client Directives:** Use `client:load`, `client:idle`, `client:visible`, `client:media`, and `client:only` directives appropriately to control when components are hydrated.  Avoid over-hydrating components.
-   **Framework Component Composition:**  Compose components from different frameworks (React, Vue, Svelte, etc.) within the same Astro page.  This allows you to leverage the strengths of different frameworks.
-   **Passing Props and Children:**  Pass props and children to framework components from Astro components.  Use named slots to organize children.
-   **Anti-pattern: Mixing Astro and Framework Components:** Do not import `.astro` components directly into UI framework components (e.g., `.jsx` or `.svelte`). Use slots to pass static content from Astro components to framework components.

### 2.2 Data Fetching

-   **`fetch` API:** Use the `fetch` API to fetch data from external sources or internal APIs.  Handle errors appropriately.
-   **Content Collections API:** Use the Content Collections API to manage structured content like blog posts or documentation.  Define schemas for your content types.
-   **CMS Integration:** Integrate with a headless CMS like Hygraph, Contentful, or Strapi to manage content.  Use the CMS's API to fetch data and display it on your website.
-   **GraphQL:** When using a CMS that supports GraphQL (like Hygraph), use GraphQL queries to fetch only the data you need.  This can improve performance and reduce data transfer.

### 2.3 Routing

-   **File-Based Routing:**  Use Astro's file-based routing to create routes automatically.  Follow a consistent naming convention for your pages.
-   **Dynamic Routes:** Use dynamic routes (e.g., `[slug].astro`) to handle variable segments in the URL.  Access the route parameters using `Astro.params`.
-   **Nested Routes:** Use nested folders within `src/pages/` to create nested routes.
-   **Anti-pattern: Overly Complex Routing:**  Avoid creating overly complex routing structures.  Keep your routes simple and intuitive.

### 2.4 Styling

-   **Scoped CSS:**  Use scoped CSS within Astro components to avoid style conflicts.  This ensures that component styles are isolated.
-   **Global Styles:**  Use global styles for site-wide styling.  Store global styles in `src/styles/global.css`.
-   **CSS Frameworks:** Use CSS frameworks like Tailwind CSS, Bootstrap, or Materialize to speed up development and maintain a consistent look and feel. Install the appropriate integration for your chosen CSS framework.
-   **CSS Modules:** Consider using CSS Modules for component-specific styling.  CSS Modules generate unique class names to avoid naming collisions.

## 3. Performance Considerations

### 3.1 Minimizing JavaScript

-   **Static-First Architecture:**  Embrace Astro's static-first architecture to minimize JavaScript and improve performance.  Render as much as possible as static HTML.
-   **Island Architecture:**  Only hydrate interactive components.  Avoid over-hydrating components.
-   **Code Splitting:**  Use code splitting to break up your JavaScript into smaller chunks.  This can improve initial load times.
-   **Lazy Loading:**  Use lazy loading for images and other assets that are not immediately visible.  This can improve initial load times.

### 3.2 Image Optimization

-   **Astro's `<Image />` Component:**  Use Astro's built-in `<Image />` component to optimize images.  The `<Image />` component automatically optimizes images, resizes them, and generates responsive versions.
-   **Image Formats:**  Use modern image formats like WebP to reduce file sizes. Use AVIF for even better compression, where supported.
-   **Compression:**  Compress images to reduce file sizes.  Use tools like ImageOptim or TinyPNG.
-   **Lazy Loading:**  Use lazy loading for images that are not immediately visible.
-   **Responsive Images:**  Use responsive images to serve different image sizes based on the user's screen size.

### 3.3 Font Optimization

-   **Web Font Formats:** Use modern web font formats like WOFF2.  These formats offer better compression and performance.
-   **Font Loading:**  Use font loading strategies to avoid blocking rendering.  Consider using `font-display: swap`.
-   **Preloading:**  Preload important fonts to improve loading times.

### 3.4 Caching

-   **Browser Caching:**  Configure browser caching to cache static assets like images, fonts, and JavaScript files.  This can improve performance for returning users.
-   **CDN:**  Use a Content Delivery Network (CDN) to serve static assets from geographically distributed servers.  This can improve performance for users around the world.

## 4. Security Best Practices

### 4.1 Input Validation

-   **Validate User Input:**  Validate all user input to prevent injection attacks and other security vulnerabilities.  Use server-side validation in addition to client-side validation.
-   **Sanitize User Input:**  Sanitize user input to remove potentially malicious code.  Use a library like DOMPurify to sanitize HTML.

### 4.2 Cross-Site Scripting (XSS)

-   **Escape Output:**  Escape all output to prevent XSS attacks.  Use Astro's built-in escaping mechanisms or a library like Handlebars.js.
-   **Content Security Policy (CSP):**  Implement a Content Security Policy (CSP) to control the resources that the browser is allowed to load.  This can help prevent XSS attacks.

### 4.3 Cross-Site Request Forgery (CSRF)

-   **CSRF Tokens:**  Use CSRF tokens to protect against CSRF attacks.  Generate a unique CSRF token for each user session and include it in all forms.

### 4.4 Authentication and Authorization

-   **Secure Authentication:**  Use secure authentication mechanisms to protect user accounts.  Use a library like Passport.js or Auth0.
-   **Authorization:**  Implement authorization to control access to resources.  Use roles and permissions to define what users are allowed to do.

### 4.5 Dependency Management

-   **Keep Dependencies Up-to-Date:**  Keep your dependencies up-to-date to patch security vulnerabilities.  Use a tool like `npm audit` or `yarn audit` to identify and fix vulnerabilities.
-   **Lock Dependencies:**  Lock your dependencies to prevent unexpected changes.  Use `package-lock.json` or `yarn.lock`.

### 4.6 Environment Variables

-   **Store Secrets Securely:**  Store secrets like API keys and database passwords in environment variables.  Do not hardcode secrets in your code.
-   **Avoid Committing `.env`:**  Never commit your `.env` file to version control.  Use a `.gitignore` file to exclude it.

## 5. Testing Approaches

### 5.1 Unit Testing

-   **Test Individual Components:**  Write unit tests to test individual components in isolation.  Use a testing framework like Jest or Mocha.
-   **Mock Dependencies:**  Mock dependencies to isolate components during testing.

### 5.2 Integration Testing

-   **Test Component Interactions:**  Write integration tests to test how components interact with each other.  Use a testing framework like Cypress or Playwright.

### 5.3 End-to-End Testing

-   **Test User Flows:**  Write end-to-end tests to test complete user flows.  Use a testing framework like Cypress or Playwright.

### 5.4 Accessibility Testing

-   **Automated Accessibility Testing:**  Use automated accessibility testing tools to identify accessibility issues.  Use a tool like axe-core.
-   **Manual Accessibility Testing:**  Perform manual accessibility testing to ensure that your website is accessible to users with disabilities.  Use a screen reader to test your website.

## 6. Common Pitfalls and Gotchas

-   **Over-Hydration:**  Avoid over-hydrating components.  Only hydrate interactive components.
-   **Incorrect Client Directives:**  Use the correct client directives for your components.  Using the wrong client directive can lead to performance issues or unexpected behavior.
-   **Global Scope Pollution:**  Avoid polluting the global scope with JavaScript variables.  Use modules to encapsulate your code.
-   **Missing `alt` Attributes:**  Always include `alt` attributes for images.  The `alt` attribute is important for accessibility and SEO.
-   **Hardcoded Secrets:**  Never hardcode secrets in your code.  Store secrets in environment variables.
-   **Insecure Dependencies:**  Keep your dependencies up-to-date to patch security vulnerabilities.
-   **Not handling errors:** Always handle errors gracefully, especially when fetching data or interacting with external APIs.

## 7. Tooling and Environment

### 7.1 Code Editor

-   **Visual Studio Code:**  Use Visual Studio Code as your code editor.  VS Code has excellent support for Astro and other web development technologies.
-   **Astro VS Code Extension:**  Install the Astro VS Code extension for syntax highlighting, code completion, and other features.
-   **ESLint and Prettier:**  Install ESLint and Prettier for code linting and formatting.

### 7.2 Package Manager

-   **npm, Yarn, or pnpm:**  Use npm, Yarn, or pnpm as your package manager.  Choose the package manager that best suits your needs.

### 7.3 Build Tool

-   **Astro's Built-in Build Tool:**  Use Astro's built-in build tool to build your website.  The build tool automatically optimizes your code and assets.

### 7.4 Version Control

-   **Git:**  Use Git for version control.  Git allows you to track changes to your code and collaborate with other developers.
-   **GitHub, GitLab, or Bitbucket:**  Use GitHub, GitLab, or Bitbucket to host your Git repository.  These platforms provide a centralized location for managing and collaborating on code.

### 7.5 Deployment

-   **Netlify, Vercel, or Cloudflare Pages:**  Use Netlify, Vercel, or Cloudflare Pages to deploy your website.  These platforms provide easy-to-use deployment workflows and CDN integration.

By following these best practices and coding standards, you can ensure that your Astro projects are maintainable, performant, and secure.