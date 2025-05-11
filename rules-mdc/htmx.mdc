---
description: This rule provides comprehensive best practices for htmx development, covering code organization, security, performance, testing, and common pitfalls. It aims to guide developers in building robust, maintainable, and secure htmx applications.
globs: *.html
---
# htmx Best Practices and Coding Standards

This document outlines the recommended best practices and coding standards for developing htmx applications. Adhering to these guidelines will help ensure code quality, maintainability, security, and performance.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

While htmx doesn't enforce a specific directory structure, a well-organized project is crucial for maintainability. Consider the following structure as a starting point:


project-root/
├── assets/             # Static assets (CSS, JavaScript, images, fonts)
│   ├── css/
│   ├── js/
│   ├── img/
│   └── fonts/
├── components/       # Reusable HTML snippets/components
│   ├── button.html
│   ├── form.html
│   └── ...
├── layouts/          # Page layouts (e.g., base.html, default.html)
│   ├── base.html
│   └── ...
├── pages/            # Individual pages (e.g., index.html, about.html)
│   ├── index.html
│   └── ...
├── scripts/          # Server-side scripts (Python, Node.js, etc.)
│   ├── app.py          # Example using Flask
│   └── ...
├── templates/        # Server-side templates (e.g., Jinja2, Django Templates)
│   ├── ...
├── .env              # Environment variables
├── requirements.txt  # Python dependencies (if applicable)
└── package.json       # Node.js dependencies (if applicable)


*   **`assets/`**:  Contains all static assets like CSS, JavaScript (including htmx extensions), images, and fonts.  Organize further into subdirectories for each asset type.
*   **`components/`**: Stores reusable HTML snippets. These components can be included in various pages or other components to avoid code duplication.
*   **`layouts/`**: Defines the overall structure of pages.  Common elements like headers, footers, and navigation are placed here, allowing pages to inherit a consistent look and feel.
*   **`pages/`**: Contains the HTML files for individual pages of your application. These pages usually include content and may use components and layouts.
*   **`scripts/`**: Holds server-side scripts responsible for handling requests, processing data, and rendering HTML responses.  These scripts interact with htmx via the HTTP protocol.
*   **`templates/`**: Used in conjunction with server-side scripting, these are templates processed by the backend to generate HTML dynamically. (Only relevant if you are using a template engine on the backend).

### 1.2 File Naming Conventions

*   **HTML files:** Use descriptive names in lowercase with hyphens (e.g., `product-details.html`, `user-profile.html`).
*   **CSS files:** Similar to HTML, use lowercase with hyphens (e.g., `main.css`, `style.css`, `components.css`).
*   **JavaScript files:**  Use camelCase for file names (e.g., `utils.js`, `htmxConfig.js`).
*   **Component files:**  Reflect the component's purpose (e.g., `product-card.html`, `search-form.html`).
*   **Image files:** Use descriptive names related to the image content (e.g., `product-image-1.jpg`, `user-avatar.png`).

### 1.3 Module Organization

For larger htmx projects, consider modularizing your JavaScript code. This promotes code reuse and maintainability. CommonJS or ES modules can be used, depending on your build setup.

javascript
// utils.js
export function formatDate(date) {
  // ...
}

export function truncateText(text, limit) {
  // ...
}

// main.js
import { formatDate, truncateText } from './utils.js';

// ...


### 1.4 Component Architecture

Break down your user interface into reusable components. This makes your code easier to understand, test, and maintain. For example, a product card, a search form, or a navigation menu can be created as separate components.

html
<!-- product-card.html -->
<div class="product-card">
  <h3>{{ product.name }}</h3>
  <img src="{{ product.image }}" alt="{{ product.name }}">
  <p>{{ product.description }}</p>
  <button hx-post="/add-to-cart/{{ product.id }}">Add to Cart</button>
</div>


### 1.5 Code Splitting Strategies

In htmx, code splitting can refer to different aspects. While htmx itself doesn't require complex JavaScript bundles, consider these:

*   **HTML Components:** Split large pages into smaller, more manageable components. Load these components on demand using htmx requests to improve initial page load time.
*   **CSS Files:** Separate your CSS into logical modules (e.g., base styles, component styles, page-specific styles) to avoid loading unnecessary styles on every page.
*   **htmx Extensions:** Only include the htmx extensions that you are actively using. Do not include all extensions if only a few are needed.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to htmx

*   **Progressive Enhancement:** Design your application to work without JavaScript first, then enhance it with htmx. This ensures accessibility and a baseline level of functionality for all users.
*   **Hypermedia as the Engine of Application State (HATEOAS):**  The core philosophy of htmx. The server should provide all the necessary links and forms for the client to interact with the application.  The client simply follows these links and submits the forms; it doesn't need to know the specific URLs or request methods.
*   **Server-Side Rendering (SSR):**  Focus on rendering HTML on the server.  This improves initial page load time, SEO, and accessibility.  htmx enhances this by enabling dynamic updates without full page reloads.
*   **Component-Based UI:** Design your UI as a collection of reusable components.  Each component manages its own state and behavior, making the application easier to understand and maintain.

### 2.2 Recommended Approaches for Common Tasks

*   **Form Handling:** Use htmx to submit forms and update the UI based on the server's response. Validate user input on both the client and server sides.
*   **Pagination:** Implement pagination using htmx to load content in chunks. The server provides links to the next and previous pages, and htmx handles the loading and rendering of the new content.
*   **Real-Time Updates:** Use WebSockets or Server-Sent Events (SSE) with htmx to display real-time data. The server pushes updates to the client, and htmx updates the UI accordingly.
*   **Dynamic Search:**  Implement live search functionality by sending AJAX requests to the server as the user types.  Display the search results in real-time using htmx.
*   **Modal Dialogs:** Use htmx to load the content of modal dialogs on demand. This avoids loading unnecessary content on initial page load.

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Over-reliance on JavaScript:** htmx aims to reduce the amount of JavaScript required for web development. Avoid using JavaScript for tasks that can be easily accomplished with htmx attributes.
*   **Complex Client-Side Logic:**  Keep your client-side logic as simple as possible. Move complex logic to the server-side, where it can be more easily tested and maintained.
*   **Mixing htmx with other Front-end frameworks (React, Angular, Vue):** While technically possible, this defeats the purpose of using htmx. Using htmx with other front-end frameworks will likely increase complexity.
*   **Ignoring Security Best Practices:**  htmx does not automatically protect against security vulnerabilities.  Always follow security best practices, such as input validation and output escaping, to prevent XSS attacks.
*   **Creating API Endpoints that return JSON to only convert it to HTML:** htmx shines when your endpoints return HTML. Avoid the extra steps needed for JSON parsing on the client side.

### 2.4 State Management Best Practices

htmx promotes a server-centric approach to state management. The server maintains the application's state, and the client simply displays it. This simplifies client-side code and reduces the risk of inconsistencies between the client and server.

*   **Server-Side Sessions:** Store user-specific data in server-side sessions. This ensures that sensitive data is not exposed to the client.
*   **Hidden Input Fields:** Use hidden input fields to store data that needs to be persisted across requests. This is useful for maintaining state in forms.
*   **Cookies:**  Use cookies to store small amounts of data on the client-side. Be mindful of cookie size limits and security implications.
*   **URL Parameters:**  Use URL parameters to represent state that can be shared or bookmarked. This is useful for implementing features like filtering and sorting.
*    **`hx-vals`:** Use `hx-vals` for temporary client-side state that is not critical and doesn't need to be persisted on the server.  This is suitable for things like the current step in a multi-step form or the visibility state of a component.

### 2.5 Error Handling Patterns

*   **HTTP Status Codes:**  Use appropriate HTTP status codes to indicate the success or failure of a request.  For example, use `200 OK` for successful requests, `400 Bad Request` for invalid input, and `500 Internal Server Error` for server-side errors.
*   **htmx Error Handling:**  htmx provides the `htmx:responseError` event for handling errors. Use this event to display error messages to the user or take other appropriate actions.
*   **Server-Side Error Pages:**  Configure your server to display custom error pages for different HTTP status codes.  This provides a user-friendly experience even when errors occur.
*   **Client-Side Error Messages:**  Display informative error messages to the user when a request fails.  Avoid displaying technical details that could expose sensitive information.
*   **Error Logging:**  Log errors on the server-side to help diagnose and fix problems.  Include relevant information such as the request URL, user ID, and error message.
*   **Graceful Degradation:** Even when errors occur, ensure that your application degrades gracefully. Provide alternative functionality or a clear explanation of the problem.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Minimize DOM Updates:**  htmx is efficient at updating the DOM, but excessive updates can still impact performance.  Use the `hx-swap` attribute to control how content is swapped into the DOM and minimize unnecessary changes.
*   **Optimize Server-Side Rendering:**  Ensure that your server-side rendering is as efficient as possible.  Use caching, compression, and other optimization techniques to reduce the time it takes to generate HTML.
*   **Compress Static Assets:**  Compress your CSS, JavaScript, and image files to reduce their size and improve loading times.  Use tools like Gzip or Brotli for compression.
*   **Use a CDN:**  Serve your static assets from a Content Delivery Network (CDN) to improve loading times for users around the world.
*   **Lazy Load Images:**  Load images only when they are visible in the viewport. This reduces initial page load time and improves performance.
*   **Throttle and Debounce Events:** When using events like `keyup` or `mousemove` with `hx-trigger`, use the `throttle` and `debounce` modifiers to limit the number of requests sent to the server.

### 3.2 Memory Management

*   **Avoid Memory Leaks:**  Be careful to avoid memory leaks in your JavaScript code.  Remove event listeners when they are no longer needed and avoid creating circular references.
*   **Minimize DOM Elements:**  Reduce the number of DOM elements on your page as much as possible.  Complex layouts with many nested elements can impact performance.
*   **Use Event Delegation:**  Use event delegation to attach event listeners to a parent element instead of attaching them to individual child elements. This reduces the number of event listeners and improves performance.

### 3.3 Rendering Optimization

*   **Use Efficient CSS Selectors:**  Use efficient CSS selectors to minimize the time it takes to apply styles to your elements.  Avoid complex selectors that traverse the DOM unnecessarily.
*   **Avoid Reflows and Repaints:**  Avoid triggering reflows and repaints in your JavaScript code.  These operations can be expensive and impact performance.  Read properties before making changes to the DOM, and batch your DOM updates.
*   **Use Hardware Acceleration:**  Take advantage of hardware acceleration to improve rendering performance.  Use CSS properties like `transform` and `opacity` to trigger hardware acceleration.

### 3.4 Bundle Size Optimization

*   **Minimize Dependencies:**  Use only the dependencies that you absolutely need.  Avoid including large libraries that you only use for a small number of features.
*   **Tree Shaking:**  Use a bundler that supports tree shaking to remove unused code from your JavaScript bundles.
*   **Code Splitting:**  Split your code into smaller chunks that can be loaded on demand.  This reduces the initial download size and improves performance.

### 3.5 Lazy Loading Strategies

*   **Lazy Load Images:** Load images only when they are visible in the viewport. This reduces initial page load time and improves performance. Use the `loading="lazy"` attribute on `<img>` elements.
*   **Lazy Load Components:**  Load components only when they are needed.  Use htmx to load the component's HTML when it is first requested.
*   **Intersection Observer API:**  Use the Intersection Observer API to detect when an element enters the viewport. This can be used to trigger the lazy loading of images or components.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Cross-Site Scripting (XSS):**  Prevent XSS attacks by escaping user input and avoiding the use of `innerHTML`. Use template engines that automatically escape output.
*   **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by using anti-CSRF tokens in your forms. Most server-side frameworks provide built-in CSRF protection.
*   **SQL Injection:**  Prevent SQL injection attacks by using parameterized queries or ORMs.  Never construct SQL queries by concatenating user input.
*   **Authentication and Authorization:**  Implement proper authentication and authorization mechanisms to protect your application from unauthorized access.
*   **Sensitive Data Exposure:**  Avoid exposing sensitive data in URLs or client-side code. Store sensitive data securely on the server-side.

### 4.2 Input Validation

*   **Client-Side Validation:**  Use client-side validation to provide immediate feedback to the user and prevent invalid data from being submitted to the server. However, never rely solely on client-side validation.
*   **Server-Side Validation:**  Always validate user input on the server-side. This is the only way to ensure that your application is protected from invalid or malicious data.
*   **Sanitize User Input:**  Sanitize user input to remove or escape potentially harmful characters. This can help prevent XSS and other attacks.

### 4.3 Authentication and Authorization Patterns

*   **Cookie-Based Authentication:**  Use cookies to store authentication tokens. Set the `Secure`, `HttpOnly`, and `SameSite` attributes to protect your cookies from unauthorized access.
*   **Session Management:**  Use server-side sessions to store user-specific data. This prevents sensitive data from being exposed to the client.
*   **Role-Based Access Control (RBAC):**  Implement RBAC to control access to different parts of your application based on user roles. This ensures that users can only access the resources that they are authorized to access.

### 4.4 Data Protection Strategies

*   **Encryption:**  Encrypt sensitive data both in transit and at rest. Use HTTPS to encrypt data in transit, and use strong encryption algorithms to encrypt data at rest.
*   **Hashing:**  Use hashing to store passwords and other sensitive data. Never store passwords in plaintext.
*   **Data Masking:**  Mask sensitive data in your application's UI to prevent it from being exposed to unauthorized users.  For example, you can mask credit card numbers or social security numbers.

### 4.5 Secure API Communication

*   **HTTPS:**  Always use HTTPS to encrypt communication between the client and server.
*   **API Authentication:**  Require API clients to authenticate themselves before accessing your API.  Use API keys, OAuth, or other authentication mechanisms.
*   **Rate Limiting:**  Implement rate limiting to prevent API abuse. This limits the number of requests that a client can make in a given period of time.
*   **Input Validation:**  Validate all input to your API to prevent injection attacks and other security vulnerabilities.
*   **Output Sanitization:**  Sanitize all output from your API to prevent XSS attacks.
*   **CORS (Cross-Origin Resource Sharing):** Configure CORS carefully to allow only trusted domains to access your API.

### 4.6 The Golden Rules of htmx Security (reiterated):

*   **Only call routes you control**: Use relative URLs to avoid executing untrusted code.
*   **Always use an auto-escaping template engine**: Prevent XSS attacks by ensuring dynamic content is escaped.
*   **Only serve user-generated content inside HTML tags**: Avoid inserting user content in dangerous contexts like script or style tags.
*   **Secure your cookies**: Use attributes like Secure, HttpOnly, and SameSite=Lax for authentication cookies.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Test Server-Side Logic:**  Write unit tests for your server-side logic to ensure that it is working correctly.  Use a testing framework like pytest (Python), Jest (Node.js), or JUnit (Java).
*   **Test htmx Endpoints:** Write tests for your htmx endpoints. This could involve making requests to your endpoints and checking the HTML responses.
*   **Mock External Dependencies:**  Mock external dependencies, such as databases or APIs, to isolate your code and make your tests more reliable.

### 5.2 Integration Testing

*   **Test Interactions Between Components:**  Write integration tests to ensure that your components are working together correctly. This includes testing the interactions between htmx attributes and JavaScript code.
*   **Test Data Flow:**  Test the flow of data through your application to ensure that data is being processed and displayed correctly.

### 5.3 End-to-End Testing

*   **Simulate User Interactions:**  Write end-to-end tests to simulate user interactions with your application. This includes testing the entire workflow from start to finish.
*   **Test UI Rendering:**  Test the rendering of your UI to ensure that it is displaying correctly. This includes testing the layout, styles, and content of your pages.

### 5.4 Test Organization

*   **Organize Tests by Feature:**  Organize your tests by feature to make it easier to find and run tests for specific parts of your application.
*   **Use a Consistent Naming Convention:**  Use a consistent naming convention for your tests to make them easier to understand and maintain.
*   **Keep Tests Independent:**  Keep your tests independent of each other to avoid cascading failures.

### 5.5 Mocking and Stubbing

*   **Use Mocking to Isolate Code:**  Use mocking to isolate your code and make your tests more reliable. This involves replacing external dependencies with mock objects that simulate their behavior.
*   **Use Stubbing to Control Behavior:**  Use stubbing to control the behavior of your mock objects. This allows you to test different scenarios and edge cases.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

*   **Incorrect `hx-target` and `hx-swap` Usage:**  Pay close attention to the `hx-target` and `hx-swap` attributes.  Incorrect usage can lead to unexpected results and broken UI.
*   **Forgetting Server-Side Validation:**  Always validate user input on the server-side. Client-side validation is not sufficient for security.
*   **Over-complicating Client-Side Logic:**  htmx is designed to simplify client-side code.  Avoid adding unnecessary complexity.
*   **Ignoring Accessibility:**  Make sure your htmx application is accessible to all users. Use semantic HTML, provide alternative text for images, and ensure that your application is keyboard-navigable.
*   **Not handling edge cases and error conditions**: Ensure you have covered the most common error conditions, such as server errors, network issues, and no results.

### 6.2 Edge Cases to Be Aware Of

*   **Handling Empty Responses:**  Consider how your application should behave when the server returns an empty response.  Should the target element be cleared, or should a default message be displayed?
*   **Handling Concurrent Requests:**  Be aware of the potential for concurrent requests.  Use the `hx-sync` attribute to manage synchronization between requests.
*   **Handling Browser History:**  htmx does not automatically manage browser history.  If you need to support browser history, you will need to use JavaScript or a dedicated htmx extension.
*   **Unexpected interactions with browser extensions**: Some browser extensions, especially those that modify HTTP headers, can interfere with htmx's behavior.  Test your application with common extensions enabled to identify and address any conflicts.

### 6.3 Version-Specific Issues

*   **Refer to the Official Documentation:**  Always refer to the official htmx documentation for the latest information and best practices.
*   **Check Release Notes:** Review the release notes for each new version of htmx to identify any breaking changes or known issues.

### 6.4 Compatibility Concerns

*   **Browser Compatibility:**  htmx is compatible with modern browsers.  However, older browsers may require polyfills.
*   **Server-Side Framework Compatibility:**  htmx can be used with any server-side framework that can generate HTML.  However, some frameworks may require specific configuration.

### 6.5 Debugging Strategies

*   **Use Browser Developer Tools:**  Use your browser's developer tools to inspect network requests, examine the DOM, and debug JavaScript code.
*   **htmx Events:**  Listen for htmx events, such as `htmx:beforeRequest`, `htmx:afterRequest`, and `htmx:responseError`, to gain insight into what is happening in your application.
*   **Server-Side Logging:**  Add logging to your server-side code to help diagnose problems.
*   **Simplify the Problem:** Break down complex htmx interactions into smaller, more manageable steps to isolate the source of the problem.
*   **Inspect HTTP Headers:** Pay attention to the HTTP headers in both the request and response. Check the `Content-Type`, `HX-Request`, and other relevant headers.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Text Editor or IDE:**  Use a text editor or IDE with support for HTML, CSS, and JavaScript. Popular options include VS Code, Sublime Text, and Atom.
*   **Browser Developer Tools:**  Use your browser's developer tools to inspect network requests, examine the DOM, and debug JavaScript code.
*   **htmx Extension for Chrome:**  Install the htmx extension for Chrome to enable features like attribute highlighting and event inspection.
*   **Server-Side Development Tools:** Use the development tools appropriate for your chosen server-side framework (e.g., Python debugger, Node.js inspector).

### 7.2 Build Configuration

*   **Use a Task Runner:**  Use a task runner like Gulp or Grunt to automate common development tasks, such as minifying CSS and JavaScript, optimizing images, and running tests.
*   **Use a Module Bundler:**  Use a module bundler like Webpack or Parcel to bundle your JavaScript modules into a single file.
*   **Configure Code Linting and Formatting:** Integrate linters (like ESLint, stylelint) and formatters (like Prettier) to automatically enforce consistent coding styles and catch common errors.

### 7.3 Linting and Formatting

*   **Use a Code Linter:**  Use a code linter to automatically detect errors and enforce coding style guidelines. Popular linters include ESLint (JavaScript) and stylelint (CSS).
*   **Use a Code Formatter:**  Use a code formatter to automatically format your code according to a consistent style. Popular formatters include Prettier and js-beautify.

### 7.4 Deployment Best Practices

*   **Use a Version Control System:**  Use a version control system like Git to track changes to your code. This makes it easier to collaborate with others and revert to previous versions if necessary.
*   **Automate Deployment:**  Automate your deployment process using a tool like Jenkins, Travis CI, or CircleCI. This makes it easier to deploy your application to production quickly and reliably.
*   **Use a Continuous Integration/Continuous Deployment (CI/CD) Pipeline:**  Implement a CI/CD pipeline to automatically build, test, and deploy your application whenever changes are made to your codebase.
*   **Monitor Your Application:**  Monitor your application in production to detect and resolve problems quickly.

### 7.5 CI/CD Integration

*   **Set Up a CI/CD Pipeline:**  Set up a CI/CD pipeline to automatically build, test, and deploy your application. This makes it easier to release new features and bug fixes quickly and reliably.
*   **Run Tests Automatically:**  Configure your CI/CD pipeline to run your tests automatically whenever changes are made to your codebase. This helps to ensure that your code is working correctly before it is deployed to production.
*   **Automate Deployment:**  Automate your deployment process using a tool like Jenkins, Travis CI, or CircleCI. This makes it easier to deploy your application to production quickly and reliably.

By adhering to these best practices and coding standards, you can build htmx applications that are robust, maintainable, secure, and performant. Remember to stay up-to-date with the latest htmx documentation and community resources to ensure that you are using the most effective techniques.