---
description: This rule file outlines best practices for Puppeteer, covering code organization, performance, security, testing, and common pitfalls. It aims to guide developers in building robust and maintainable Puppeteer applications.
globs: *.js
---
# Puppeteer Best Practices

This document outlines best practices for using Puppeteer, covering code organization, common patterns, performance, security, testing, and common pitfalls. Following these guidelines will help you write more robust, maintainable, and efficient Puppeteer applications.

## 1. Code Organization and Structure

### Directory Structure

Consistent directory structure is crucial for maintainability. Here's a recommended structure:


project-root/
├── src/
│   ├── scrapers/       # Directory for specific scraping logic
│   │   ├── product_scraper.js
│   │   ├── news_scraper.js
│   │   └── ...
│   ├── utils/          # Utility functions (e.g., CSV export, error handling)
│   │   ├── csv_utils.js
│   │   ├── error_handler.js
│   │   └── ...
│   ├── config/         # Configuration files (e.g., URLs, selectors)
│   │   ├── urls.js
│   │   ├── selectors.js
│   │   └── ...
│   ├── models/         # Data models (e.g., Product, Article)
│   │   ├── product.js
│   │   ├── article.js
│   │   └── ...
│   ├── core/            # Core puppeteer setup and helper functions
│   │   ├── browser_manager.js # Browser launch and closing
│   │   ├── page_actions.js   # Reusable page interaction functions
│   │   └── ...
│   └── index.js        # Main entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .eslintrc.js        # ESLint configuration
├── .prettierrc.js      # Prettier configuration
├── package.json
└── README.md


### File Naming Conventions

*   **Scrapers:** `[target_site]_scraper.js` (e.g., `amazon_scraper.js`, `twitter_scraper.js`)
*   **Utility functions:** `[utility_name]_utils.js` (e.g., `csv_utils.js`, `string_utils.js`)
*   **Configuration files:**  Use descriptive names like `urls.js`, `selectors.js`, `api_keys.js`.
*   **Data models:** Singular nouns in PascalCase (e.g., `Product.js`, `Article.js`).
*   **Core Files:**  `browser_manager.js`, `page_actions.js`, `request_interceptor.js`

### Module Organization

*   **Separate concerns:** Each module should have a single responsibility.  For example, one module handles browser launching, another handles navigation, and another handles data extraction.
*   **Use ES modules:** Use `import` and `export` for better readability and maintainability.
*   **Avoid circular dependencies:** Ensure modules don't depend on each other in a circular manner.

### Component Architecture

*   **Reusable components:** Create reusable components for common tasks like navigating to a page, filling out a form, or waiting for an element to load.  These can be organized into functions within the `core` directory and reused across multiple scrapers.
*   **Configuration-driven:** Design components to be configurable via parameters or configuration files. This allows for easy adaptation to different websites or scraping scenarios.

### Code Splitting

*   **Scraper-specific bundles:** If you have multiple scrapers with significant code differences, consider creating separate bundles for each scraper to reduce initial load time.
*   **Dynamic imports:** Use dynamic imports (`import()`) to load modules only when they are needed.  This can be useful for loading scraper-specific configurations or utility functions.

## 2. Common Patterns and Anti-patterns

### Design Patterns

*   **Strategy Pattern:** Use the strategy pattern to handle different scraping strategies for different websites or content types. Create an interface for scraping, and implement concrete strategies for each website.  This promotes code reusability and flexibility.
*   **Factory Pattern:** Employ the factory pattern to create browser instances with specific configurations. This is especially useful when dealing with different browser profiles or proxy settings.
*   **Page Object Model (POM):** Encapsulate page elements and interactions into separate page objects. This makes tests more readable and maintainable.
*   **Observer Pattern:** Implement an observer pattern to monitor browser events (e.g., network requests, console messages) and react accordingly. This can be helpful for debugging and error handling.

### Recommended Approaches

*   **Resource Blocking:** Block unnecessary resources (images, CSS, fonts) to improve scraping speed and reduce bandwidth usage. Use `page.route()` to intercept requests and abort those that are not needed.
*   **Dynamic Content Handling:** Use `page.waitForSelector()`, `page.waitForResponse()`, or `page.waitForFunction()` instead of fixed delays to ensure elements are loaded before interacting with them.  Always use try/catch blocks around these to gracefully handle timeouts.
*   **Error Handling:** Implement robust error handling with `try...catch` blocks and logging.  Ensure browser instances are properly closed even if errors occur.
*   **User-Agent Spoofing:** Set the `User-Agent` header to mimic real user behavior and avoid detection by anti-bot systems.
*   **Proxy Rotation:** Use a proxy server and rotate it periodically to avoid IP bans.
*   **Headless Mode vs. Headed Mode:** Use headless mode for faster execution but switch to headed mode for debugging when necessary. Consider `slowMo` to slow down execution in headed mode for better visibility.

### Anti-patterns and Code Smells

*   **Hardcoding selectors:** Avoid hardcoding CSS selectors or XPath expressions directly in your code.  Instead, store them in a configuration file or a separate module and reuse them.
*   **Using fixed timeouts:** Avoid using `setTimeout` or `sleep` for waiting. Use `waitForSelector`, `waitForResponse`, or `waitForFunction` instead.
*   **Ignoring errors:** Never ignore errors.  Always log errors and handle them gracefully.
*   **Overly complex XPath expressions:** Avoid using overly complex XPath expressions that are prone to breaking when the page structure changes.  Use CSS selectors whenever possible.
*   **Direct DOM scraping:** Try to extract data directly from API responses instead of scraping the DOM whenever possible. This is more efficient and stable.
*   **Not closing the browser:** Failing to close the browser instance after the scraping task is complete leads to memory leaks and resource exhaustion.
*   **Parallel scraping without control:** Launching too many parallel scrapers can overload the target server and lead to IP bans or performance issues. Implement rate limiting and concurrency control.

### State Management

*   **Stateless components:**  Design your scrapers to be as stateless as possible.  Pass all necessary data as arguments to functions or components.
*   **Centralized configuration:** Store global configurations (e.g., API keys, URLs, proxy settings) in a centralized configuration file or environment variables.

### Error Handling Patterns

*   **Try-Catch Blocks:** Wrap Puppeteer operations within `try...catch` blocks to catch potential errors. Log the errors and handle them appropriately (e.g., retry the request, skip the item, or stop the scraper).
*   **Global Error Handling:** Implement a global error handler to catch unhandled exceptions and prevent the application from crashing.
*   **Retry Mechanism:** Implement a retry mechanism to automatically retry failed requests. Use exponential backoff to avoid overloading the target server.
*   **Circuit Breaker:** Implement a circuit breaker pattern to prevent repeated failures from cascading and bringing down the scraper.

## 3. Performance Considerations

### Optimization Techniques

*   **Resource Blocking:** Block unnecessary resources (images, CSS, fonts) using `page.route()`.
*   **Viewport Optimization:** Set the viewport to a smaller size to reduce rendering time and memory usage.
*   **JavaScript Optimization:** Minimize the amount of JavaScript that needs to be executed on the page. Use `evaluateHandle` for extracting complex data structures.
*   **Caching:** Cache frequently accessed data to reduce the number of requests to the target server.
*   **Concurrency Control:** Limit the number of concurrent browser instances or pages to avoid overloading the system or the target server. Use libraries like `p-queue` or `async` for managing concurrency.
*   **Disable JavaScript if possible**: `await page.setJavaScriptEnabled(false)` can drastically improve performance if the target site doesn't require JavaScript for the data you need.
*   **EvaluateHandle instead of evaluate**:  For returning complex objects use `evaluateHandle` rather than `evaluate`.  `evaluateHandle` returns a reference to the object in the browser, avoiding serialization overhead.

### Memory Management

*   **Close Browser Instances:** Always close browser instances and pages when they are no longer needed.  Use `browser.close()` and `page.close()`.
*   **Avoid Memory Leaks:** Be mindful of memory leaks in your code, especially when using closures or event listeners.  Remove event listeners when they are no longer needed.
*   **Garbage Collection:**  Force garbage collection periodically to release unused memory. This can be done using `await page.evaluate(() => gc());`. However, use this sparingly as it can impact performance.

### Rendering Optimization

*   **Disable GPU:** Disable GPU acceleration for headless browsers to reduce memory usage. Use the `--disable-gpu` flag when launching the browser.
*   **Set Viewport:** `await page.setViewport({ width: 800, height: 600 });` Avoid very high resolutions when not needed.

### Bundle Size Optimization

*   **Tree Shaking:** Use a bundler like Webpack or Rollup with tree shaking enabled to remove unused code.
*   **Code Minification:** Minify your JavaScript code to reduce its size.
*   **Dependency Optimization:** Analyze your dependencies and remove any that are not needed.  Consider using smaller alternatives for large libraries.

### Lazy Loading

*   **Lazy Loading of Images:** If you are scraping images, consider lazy loading them to reduce initial page load time. Implement this on the scraped site side if applicable.
*   **On-Demand Scraping:** Only scrape data that is currently needed. Defer scraping of less important data until it is requested.

## 4. Security Best Practices

### Common Vulnerabilities and Prevention

*   **Code Injection:** Prevent code injection by carefully validating any user-supplied input that is used in `evaluate()` or other code execution contexts. Avoid using `eval()` altogether.
*   **Cross-Site Scripting (XSS):** Be aware of XSS vulnerabilities when scraping data from untrusted sources. Sanitize data before displaying it to users.
*   **Denial of Service (DoS):** Implement rate limiting and concurrency control to prevent your scraper from overloading the target server and causing a denial of service.
*   **IP Bans:** Use proxy servers and rotate them periodically to avoid IP bans.  Also, respect the target website's `robots.txt` file.

### Input Validation

*   **Sanitize User Input:** Sanitize any user-supplied input before using it in selectors or other code that interacts with the browser. Use libraries like `DOMPurify` to sanitize HTML.
*   **Validate URLs:** Validate URLs before navigating to them using `page.goto()`.  Ensure that the URLs are safe and trusted.

### Authentication and Authorization

*   **Secure Credentials:** Store API keys, passwords, and other sensitive credentials securely using environment variables or a dedicated secrets management tool.  Never hardcode credentials in your code.
*   **Session Management:** If you need to maintain a session with the target website, use cookies and store them securely.  Consider using the `page.cookies()` method to manage cookies.

### Data Protection

*   **Encryption:** Encrypt sensitive data before storing it.  Use libraries like `crypto` to encrypt data.
*   **Data Masking:** Mask sensitive data (e.g., credit card numbers, social security numbers) before displaying it to users or storing it in logs.

### Secure API Communication

*   **HTTPS:** Always use HTTPS for API communication.  Verify that the server's SSL certificate is valid.
*   **API Rate Limiting:** Implement rate limiting on your API endpoints to prevent abuse.
*   **Authentication:** Use authentication to protect your API endpoints.  Consider using API keys or JWT tokens.

## 5. Testing Approaches

### Unit Testing

*   **Mock Puppeteer Objects:** Mock Puppeteer objects (e.g., `browser`, `page`) to isolate units of code during testing. Use libraries like `jest-puppeteer` or `mock-puppeteer` for mocking.
*   **Test Utility Functions:** Unit test utility functions (e.g., CSV export, error handling) to ensure they are working correctly.
*   **Verify Logic:** Write unit tests to verify the logic of your scraping algorithms.

### Integration Testing

*   **Test Interactions:** Write integration tests to verify the interactions between different components of your scraper (e.g., browser launching, page navigation, data extraction).
*   **Use Test Servers:** Use a test server or mock API to simulate the target website during integration testing. Libraries like `nock` can be helpful.
*   **Verify Data Flow:** Write integration tests to verify that data is flowing correctly through your scraper.

### End-to-End Testing

*   **Simulate User Flows:** Write end-to-end tests to simulate real user flows and verify that the scraper is working as expected from start to finish.
*   **Test Real Websites:** Test your scraper against real websites to ensure that it is robust and reliable.
*   **Use Headless Mode:** Run end-to-end tests in headless mode for faster execution.
*   **Screenshot Comparison**: Integrate screenshot comparison tools to detect visual regressions in your scraped data. This is particularly useful when target website layouts change frequently.

### Test Organization

*   **Separate Test Files:** Create separate test files for unit tests, integration tests, and end-to-end tests.
*   **Use Descriptive Names:** Use descriptive names for your test files and test cases.
*   **Organize Tests by Feature:** Organize your tests by feature or functionality.

### Mocking and Stubbing

*   **Mock Network Requests:** Mock network requests to avoid making real requests to the target server during testing. Use libraries like `nock` for mocking.
*   **Stub Puppeteer Methods:** Stub Puppeteer methods (e.g., `page.goto()`, `page.$()`) to control the behavior of your scraper during testing.

## 6. Common Pitfalls and Gotchas

### Frequent Mistakes

*   **Not Handling Dynamic Content:** Failing to handle dynamic content and relying on fixed delays.
*   **Using Brittle Selectors:** Using overly complex or fragile CSS selectors or XPath expressions.
*   **Ignoring Errors:** Not handling errors gracefully and letting the scraper crash.
*   **Not Closing Browser Instances:** Forgetting to close browser instances and leaking memory.
*   **Not Respecting `robots.txt`:** Ignoring the `robots.txt` file and scraping restricted areas of the website.
*   **Missing Rate Limiting:** Not implementing rate limiting and overloading the target server.
*   **Incorrectly Handling Async Operations**: Forgetting to `await` asynchronous calls, leading to race conditions or unexpected behavior.

### Edge Cases

*   **Website Layout Changes:** Websites frequently change their layout, which can break your scraper. Implement robust selector strategies and monitor your scraper for errors.
*   **Anti-Bot Measures:** Websites implement anti-bot measures to detect and block scrapers. Use techniques like user-agent spoofing, proxy rotation, and CAPTCHA solving to avoid detection.
*   **Infinite Scroll:** Handling infinite scroll pages requires special attention to avoid scraping the same data multiple times. Use techniques like pagination or scroll-to-bottom with debouncing.
*   **Frames and Iframes:** Dealing with content inside frames or iframes requires switching to the frame context before scraping.  Use `page.frame()` to get a frame and then interact with it.
*   **Shadow DOM:** Accessing elements inside shadow DOM requires using `shadowRoot` property or special selectors.

### Version-Specific Issues

*   **API Changes:** Be aware of API changes between different versions of Puppeteer.  Consult the release notes for each version.
*   **Chromium Updates:** Puppeteer is tied to specific versions of Chromium.  Chromium updates can sometimes break your scraper.  Test your scraper regularly after Chromium updates.

### Compatibility Concerns

*   **Node.js Version:** Ensure that you are using a compatible version of Node.js.  Refer to the Puppeteer documentation for compatibility information.
*   **Operating System:** Puppeteer may behave differently on different operating systems.  Test your scraper on all target operating systems.

### Debugging Strategies

*   **Headless Mode:** Switch to headed mode for debugging to visually inspect the browser.  Use `headless: false` when launching the browser.
*   **Console Logging:** Use `console.log()` to log debugging information to the console.  Capture console output from the page using `page.on('console', msg => console.log('PAGE LOG:', msg.text()));`
*   **`slowMo` Option:** Use the `slowMo` option when launching the browser to slow down the execution and make it easier to follow along.  `const browser = await puppeteer.launch({ headless: false, slowMo: 50 });`
*   **Debugger Statement:** Add `debugger;` statements to your code to pause execution and inspect variables in the Chrome DevTools.
*   **Screenshot Debugging:**  Take screenshots at key points in the script execution to visually debug the page state.  `await page.screenshot({path: 'screenshot.png'});`
*   **Performance Tracing:** Use Chrome DevTools to record a performance trace and identify performance bottlenecks.  `await page.tracing.start({path: 'trace.json'});` and `await page.tracing.stop();`
*   **Browser DevTools:** Use `devtools: true` when launching puppeteer to directly show the Browser DevTools. `const browser = await puppeteer.launch({ devtools: true });`

## 7. Tooling and Environment

### Recommended Development Tools

*   **IDE:** Visual Studio Code (VS Code) is a popular choice for Puppeteer development due to its excellent JavaScript support, debugging capabilities, and extensions.
*   **Debugger:** The Chrome DevTools debugger is essential for debugging Puppeteer code.  Use the debugger statement or launch the browser in headed mode to use the DevTools.
*   **Linter:** ESLint helps enforce coding standards and identify potential errors. Configure ESLint with appropriate rules for Puppeteer code.
*   **Formatter:** Prettier automatically formats your code to ensure consistency. Configure Prettier with appropriate settings for JavaScript code.

### Build Configuration

*   **Bundler:** Use a bundler like Webpack or Rollup to bundle your code and dependencies into a single file.  This simplifies deployment and improves performance.
*   **Transpiler:** Use a transpiler like Babel to convert your code to a compatible version of JavaScript.
*   **Environment Variables:** Use environment variables to configure your application for different environments (e.g., development, staging, production).

### Linting and Formatting

*   **ESLint:** Configure ESLint with recommended rules for Puppeteer code.  Use the `eslint:recommended` preset and customize it to fit your needs.
*   **Prettier:** Configure Prettier with consistent settings for JavaScript code.  Use the `prettier` command to automatically format your code.
*   **.editorconfig**: Use .editorconfig to enforce consistent coding styles across different editors.

### Deployment Best Practices

*   **Docker:** Use Docker to containerize your application and ensure consistency across different environments.  Create a Dockerfile that specifies the dependencies and configuration for your application.
*   **Serverless Functions:** Deploy your scraper as a serverless function (e.g., AWS Lambda, Google Cloud Functions) to avoid managing servers.
*   **Environment Variables:** Use environment variables to configure your application in the deployment environment.
*   **CI/CD Pipeline:** Automate the deployment process using a CI/CD pipeline.  Use tools like Jenkins, GitLab CI, or GitHub Actions to build, test, and deploy your application.

### CI/CD Integration

*   **Automated Testing:** Integrate automated testing into your CI/CD pipeline to ensure that your scraper is working correctly before deploying it to production.  Run unit tests, integration tests, and end-to-end tests as part of the pipeline.
*   **Code Quality Checks:** Integrate code quality checks into your CI/CD pipeline to ensure that your code meets coding standards and best practices.  Use ESLint and Prettier to check code quality.
*   **Deployment Automation:** Automate the deployment process to ensure that your application is deployed consistently and reliably.  Use tools like Ansible or Terraform to automate deployment.
*   **Monitor Deploys:** Integrate monitoring to immediately notify you of any broken deploys.

By following these best practices, you can develop robust, scalable, and maintainable Puppeteer applications.