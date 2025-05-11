---
description: This rule provides best practices for using Cheerio for web scraping and HTML parsing in JavaScript, covering code organization, performance, security, testing, and common pitfalls.
globs: *.js
---
# Cheerio Best Practices

This document outlines the best practices for using Cheerio, a fast, flexible, and lean implementation of core jQuery designed specifically for the server.  It focuses on efficient and ethical web scraping, robust code organization, and secure data handling.

## Library Information:
- Name: cheerio
- Tags: web-scraping, javascript, html-parsing, nodejs

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

Adopt a structured directory to enhance maintainability and scalability:


project-root/
├── src/
│   ├── scrapers/
│   │   ├── amazon_scraper.js  # Specific website scrapers
│   │   ├── generic_scraper.js # Reusable scraping logic
│   ├── utils/
│   │   ├── helpers.js         # Utility functions
│   │   ├── request.js         # HTTP request handling (Axios wrapper)
│   ├── models/
│   │   ├── product.js        # Data models
│   ├── config/
│   │   ├── config.js         # Configuration settings
│   ├── index.js             # Main application entry point
├── tests/
│   ├── scrapers/
│   │   ├── amazon_scraper.test.js
│   ├── utils/
│   │   ├── helpers.test.js
├── .env                   # Environment variables
├── package.json
├── package-lock.json
├── README.md


### 1.2. File Naming Conventions

Use descriptive names:

-   **Scrapers:** `[website]_scraper.js` (e.g., `amazon_scraper.js`, `ebay_scraper.js`)
-   **Utilities:** `[module_name].js` (e.g., `request.js`, `data_formatter.js`)
-   **Tests:** `[module_name].test.js` (e.g., `amazon_scraper.test.js`)

### 1.3. Module Organization

-   **Encapsulation:**  Group related functionalities into modules (e.g., scraper functions, utility functions).
-   **Separation of Concerns:** Isolate scraping logic, data processing, and storage functionalities.
-   **Reusability:**  Design modules for reuse across different scrapers.

### 1.4. Component Architecture Recommendations

For complex scraping applications, consider a component-based architecture:

-   **Scraper Components:** Components that handle scraping data from specific websites. Should handle request logic and initial parsing of HTML
-   **Parser Components:** Responsible for extracting relevant data from the scraped HTML using cheerio.
-   **Data Model Components:** Define the structure of the data to be extracted.
-   **Storage Components:** Save scraped data to files, databases, etc.

### 1.5. Code Splitting Strategies

-   **Scraper-Specific Logic:** Separate logic for each website into individual modules.
-   **Reusable Utilities:** Move common functions into utility modules.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Strategy Pattern:** Implement different scraping strategies based on website structure or anti-scraping measures.
-   **Factory Pattern:** Dynamically create scraper instances based on website configuration.
-   **Observer Pattern:**  Notify subscribers when new data is scraped.

### 2.2. Recommended Approaches for Common Tasks

-   **Making HTTP Requests:** Use Axios for making HTTP requests.  Handle retries and timeouts gracefully.
-   **Loading HTML:** Load HTML into Cheerio using `cheerio.load()`. Handle loading large HTML documents efficiently.
-   **Selecting Elements:** Use CSS selectors effectively to target desired elements.  Consider specificity and performance.
-   **Extracting Data:** Extract data from selected elements using `.text()`, `.attr()`, `.val()`, etc.

### 2.3. Anti-patterns and Code Smells

-   **Overly Complex Selectors:**  Avoid overly long or complex CSS selectors that can be difficult to maintain and may impact performance.  Refactor into smaller, more manageable selectors.
-   **Ignoring Errors:** Always handle errors and exceptions gracefully.  Log errors and provide meaningful error messages.
-   **Hardcoding Values:**  Avoid hardcoding URLs, selectors, or configuration values.  Use configuration files or environment variables.
-   **Lack of Rate Limiting:**  Always implement rate limiting to avoid overloading target websites and getting IP-blocked.
-   **Lack of Respect for `robots.txt`:** Always check and respect the `robots.txt` file to avoid scraping restricted content.
-   **Not handling dynamic content**: If the site uses JavaScript to render content, Cheerio alone won't work. Integrate with Puppeteer or Playwright to render the page before parsing it.

### 2.4. State Management Best Practices

-   For simple scripts, use local variables to store scraped data.
-   For complex applications, consider using state management libraries like Redux or Zustand (although these are usually overkill for typical cheerio use cases).  These are needed more for front-end state management. 
-   For managing concurrency and scheduling, use task queues (see below).

### 2.5. Error Handling Patterns

-   **Try-Catch Blocks:** Wrap scraping logic in `try-catch` blocks to handle exceptions.
-   **Custom Error Classes:** Define custom error classes to represent specific scraping errors.
-   **Logging:** Log errors and exceptions to a file or logging service.
-   **Retry Logic:** Implement retry logic for failed HTTP requests.

## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Efficient Selectors:** Use the most efficient CSS selectors possible.  Avoid overly complex selectors and prioritize ID selectors or class selectors over attribute selectors.
-   **Caching:** Cache frequently accessed data to reduce the number of HTTP requests.
-   **Rate Limiting:** Implement rate limiting to avoid overloading target websites. Space out requests to be respectful and avoid getting blocked.
-   **Asynchronous Operations:**  Use asynchronous operations (e.g., `async/await`) to avoid blocking the main thread.
-   **Connection Pooling:** Reuse HTTP connections to reduce overhead.  Axios handles connection pooling automatically.
-   **Parallel Scraping:** Use concurrent scraping (e.g. `Promise.all`) to speed up data collection. Implement carefully to avoid overwhelming the target server.
-   **Use streams for large files:** When processing very large scraped files (e.g. JSON), use Node.js streams to avoid loading the entire file into memory at once.

### 3.2. Memory Management

-   **Minimize Data Retention:** Only store the data you need.  Discard unnecessary data as soon as possible.
-   **Garbage Collection:**  Ensure that unused objects are properly garbage collected.
-   **Avoid Memory Leaks:**  Be careful to avoid memory leaks, especially when using callbacks or closures.

### 3.3. Rendering Optimization

-   Cheerio doesn't render content; it parses existing HTML. If dynamic content is required, use Puppeteer or Playwright to render the page first.

### 3.4. Bundle Size Optimization

-   Cheerio itself is very lightweight.  However, using too many utility libraries can increase bundle size.
-   Use tools like Webpack or Parcel to bundle your code and optimize bundle size.

### 3.5. Lazy Loading

-   Lazy load images or other resources that are not immediately visible.
-   However, this is often not relevant for server-side scraping with Cheerio.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

-   **Cross-Site Scripting (XSS):** Cheerio itself does not directly introduce XSS vulnerabilities. However, if you are displaying scraped data in a web browser, be sure to sanitize the data properly to prevent XSS attacks.
-   **Denial of Service (DoS):** Implement rate limiting and request timeouts to prevent DoS attacks on target websites.
-   **Data Injection:** Sanitize and validate scraped data before storing it in a database to prevent data injection attacks.

### 4.2. Input Validation

-   Validate all input data to prevent malicious input from corrupting your application or data.
-   Validate URLs to prevent scraping unintended websites.

### 4.3. Authentication and Authorization

-   For accessing password-protected websites, implement proper authentication and authorization mechanisms.
-   Store credentials securely using environment variables or configuration files.
-   Avoid storing credentials directly in your code.

### 4.4. Data Protection

-   Encrypt sensitive data at rest and in transit.
-   Use secure protocols (e.g., HTTPS) for all network communication.

### 4.5. Secure API Communication

-   If your scraping application interacts with APIs, use secure API keys and tokens.
-   Implement proper authentication and authorization mechanisms for API access.

## 5. Testing Approaches

### 5.1. Unit Testing

-   Unit test individual functions and components.
-   Use mocking and stubbing to isolate units of code.
-   Test edge cases and error conditions.

### 5.2. Integration Testing

-   Integrate test different components of your application to ensure that they work together correctly.
-   Test the interaction between scrapers, parsers, and data storage components.

### 5.3. End-to-End Testing

-   End-to-end test the entire scraping process from start to finish.
-   Simulate real-world scenarios to ensure that your application works as expected.

### 5.4. Test Organization

-   Organize your tests in a clear and consistent manner.
-   Use descriptive test names.
-   Keep your tests small and focused.

### 5.5. Mocking and Stubbing

-   Use mocking and stubbing to isolate units of code during testing.
-   Mock HTTP requests to avoid making real requests to target websites during testing.
-   Tools: Jest, Mocha, Sinon.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Ignoring `robots.txt`:** Always respect the `robots.txt` file.
-   **Not Handling Dynamic Content:** Cheerio cannot render JavaScript. Use Puppeteer or Playwright for dynamic content.
-   **Overloading Target Websites:** Implement rate limiting to avoid overloading target websites.
-   **Not Handling Errors:** Always handle errors and exceptions gracefully.
-   **Using Incorrect Selectors:** Ensure that your CSS selectors are correct and target the desired elements.
-   **Not Validating Data:** Validate scraped data to prevent data corruption.

### 6.2. Edge Cases

-   Websites with infinite scroll: These sites load content dynamically as you scroll. Use Puppeteer or Playwright to simulate scrolling and load all the content before parsing.
-   Websites that require JavaScript to function: As stated before, use a headless browser like Puppeteer or Playwright.
-   Websites with anti-scraping techniques: Websites may use techniques like IP blocking, CAPTCHAs, or dynamic content to prevent scraping. You'll need to implement strategies to avoid these, such as using proxies, solving CAPTCHAs, or rendering JavaScript.
-   Websites that change their structure frequently: Websites may change their HTML structure, which can break your scraper. You'll need to monitor your scraper and update it as needed.

### 6.3. Version-Specific Issues

-   Be aware of version-specific issues and breaking changes in Cheerio.
-   Check the Cheerio documentation and release notes for information on known issues.

### 6.4. Compatibility Concerns

-   Ensure that Cheerio is compatible with your version of Node.js.
-   Be aware of potential compatibility issues with other libraries.

### 6.5. Debugging Strategies

-   Use `console.log()` to print out the values of variables and expressions.
-   Use a debugger to step through your code and inspect the state of your application.
-   Use try-catch blocks to catch exceptions and log error messages.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **IDE:** Visual Studio Code, WebStorm
-   **HTTP Client:** Postman, Insomnia
-   **Debugging Tools:** Node.js Inspector, Chrome DevTools

### 7.2. Build Configuration

-   Use a build tool like Webpack or Parcel to bundle your code and optimize bundle size.
-   Configure your build tool to handle different environments (e.g., development, production).

### 7.3. Linting and Formatting

-   Use a linter like ESLint to enforce code style and prevent errors.
-   Use a formatter like Prettier to automatically format your code.
-   Configure your linter and formatter to work with Cheerio code.

### 7.4. Deployment Best Practices

-   Deploy your scraping application to a reliable hosting provider.
-   Use a process manager like PM2 to ensure that your application is always running.
-   Monitor your application for errors and performance issues.

### 7.5. CI/CD Integration

-   Use a CI/CD tool like Jenkins, Travis CI, or CircleCI to automate the build, test, and deployment process.
-   Configure your CI/CD pipeline to run your unit tests, integration tests, and end-to-end tests.


By following these best practices, you can build robust, efficient, and secure web scraping applications using Cheerio.