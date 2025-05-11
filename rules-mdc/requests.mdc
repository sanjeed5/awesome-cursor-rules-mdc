---
description: This rule file outlines best practices for using the Python requests library, covering performance, security, code organization, and testing.
globs: *.py
---
---

# Requests Library Best Practices

This document provides comprehensive guidelines for using the Python `requests` library effectively. It covers various aspects, including code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and tooling.

## Library Information:

- Name: requests
- Tags: web, python, http-client

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices:

*   **Simple Scripts:** For simple, single-file scripts, the structure is less critical. Keep related resources (e.g., configuration files, data files) in the same directory.
*   **Larger Projects:** For larger projects:
    
    my_project/
    ├── src/
    │   ├── __init__.py
    │   ├── api_client.py  # Contains requests-related functions
    │   ├── models.py       # Data models for API responses
    │   ├── utils.py        # Utility functions
    │   └── config.py       # Configuration settings
    ├── tests/
    │   ├── __init__.py
    │   ├── test_api_client.py
    │   └── conftest.py      # pytest configuration
    ├── README.md
    ├── requirements.txt
    └── .gitignore
    
    *   `src/`: Contains the main application code.
    *   `api_client.py`:  Houses the `requests` calls, session management, and error handling.
    *   `models.py`: Defines data classes/namedtuples to represent the structure of API responses, aiding type hinting and validation.
    *   `tests/`: Contains unit and integration tests.
    *   `requirements.txt`: Lists project dependencies.

### 1.2 File Naming Conventions:

*   Use descriptive names for files related to `requests`, such as `api_client.py`, `http_utils.py`, or `<service_name>_client.py`.
*   Follow PEP 8 guidelines: lowercase with underscores (e.g., `get_data.py`).

### 1.3 Module Organization Best Practices:

*   **Grouping by Functionality:**  Organize code into modules based on functionality. For example, a module for authentication, another for data retrieval, and another for error handling.
*   **Avoiding Circular Dependencies:**  Design modules to minimize dependencies between them and prevent circular imports.

### 1.4 Component Architecture Recommendations:

*   **Layered Architecture:** Use a layered architecture to separate concerns:
    *   **Presentation Layer:** (If applicable) Handles user input and displays data.
    *   **Business Logic Layer:** Orchestrates the application logic and uses the data access layer.
    *   **Data Access Layer:**  Contains the `requests` calls and interacts with external APIs.  This layer should abstract away the details of the HTTP client.
*   **Dependency Injection:** Use dependency injection to provide the `requests` session or client to components that need it, allowing for easier testing and configuration.

### 1.5 Code Splitting Strategies:

*   **By API Endpoint:** If your application interacts with multiple API endpoints, consider creating separate modules or classes for each endpoint to improve maintainability.
*   **Functional Decomposition:**  Split complex tasks into smaller, manageable functions.  This promotes reusability and testability.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to `requests`:

*   **Singleton (for Session):**  Use a singleton pattern to manage a single `requests.Session` instance, ensuring connection pooling is used effectively across the application.  Be mindful of potential thread-safety issues with singletons in multithreaded environments.
*   **Adapter Pattern:** Create an adapter class around the `requests` library to abstract away the underlying HTTP client. This makes it easier to switch to a different library in the future or add custom logic.
*   **Retry Pattern:** Implement retry logic with exponential backoff to handle transient network errors and API rate limits.

### 2.2 Recommended Approaches for Common Tasks:

*   **Fetching Data:** Use `requests.get()` for retrieving data. Handle potential errors using `response.raise_for_status()` and appropriate exception handling.
*   **Sending Data:** Use `requests.post()`, `requests.put()`, or `requests.patch()` for sending data. Set the `Content-Type` header appropriately (e.g., `application/json`).
*   **Authentication:** Utilize the `auth` parameter for authentication. Use appropriate authentication schemes such as HTTP Basic Auth, Bearer tokens, or OAuth.
*   **File Uploads:**  Use the `files` parameter to upload files. Provide a dictionary where the keys are the field names and the values are file-like objects or tuples containing the filename and file-like object.

### 2.3 Anti-patterns and Code Smells to Avoid:

*   **Ignoring Errors:** Not handling exceptions raised by `requests` can lead to unexpected behavior and application crashes.
*   **Hardcoding URLs:** Hardcoding URLs makes the code less flexible and harder to maintain. Store URLs in configuration files or environment variables.
*   **Not Using Sessions:** Failing to use `requests.Session()` for multiple requests to the same host can result in performance degradation due to repeated connection setups.
*   **Disabling SSL Verification Unnecessarily:** Disabling SSL verification (`verify=False`) should only be done when absolutely necessary and with caution, as it can expose the application to man-in-the-middle attacks.  Investigate the root cause of SSL verification failures instead of simply disabling it.
*   **Excessive Retries without Backoff:** Retrying requests excessively without an exponential backoff strategy can overwhelm the server and worsen the situation.
*   **Storing Sensitive Information in Code:** Avoid storing API keys, passwords, and other sensitive information directly in the code. Use environment variables or a secure configuration management system.
*   **Not setting timeouts:** Not setting timeouts on requests can lead to your application hanging indefinitely if a server is unresponsive.

### 2.4 State Management Best Practices:

*   **Stateless API Clients:** Design API clients to be stateless whenever possible. Avoid storing request-specific data within the client object. Pass all necessary data as arguments to the request methods.
*   **Session Management:** Use `requests.Session()` to maintain state (e.g., cookies, authentication) across multiple requests.

### 2.5 Error Handling Patterns:

*   **Catching Exceptions:** Catch `requests.exceptions.RequestException` and its subclasses (e.g., `requests.exceptions.HTTPError`, `requests.exceptions.ConnectionError`, `requests.exceptions.Timeout`) to handle different types of errors.
*   **Using `raise_for_status()`:** Call `response.raise_for_status()` to raise an exception for HTTP error codes (4xx and 5xx).
*   **Logging Errors:** Log detailed error messages, including the URL, status code, and response content, to aid in debugging.
*   **Returning Meaningful Error Responses:** When creating your own APIs that use the `requests` library internally, return informative error responses to the client.
*   **Retry Logic:** Implement retry logic for transient errors (e.g., network timeouts, temporary server errors) with exponential backoff.

## 3. Performance Considerations

### 3.1 Optimization Techniques:

*   **Using Sessions:** Utilize `requests.Session()` to reuse connections and reduce overhead.
*   **Connection Pooling:** The `requests.Session()` object automatically handles connection pooling, which improves performance by reusing existing connections.
*   **Streaming Responses:** For large responses, use `stream=True` to process data incrementally and avoid loading the entire response into memory at once.  Remember to close the response after processing.
*   **Caching Responses:** Consider caching responses to reduce redundant API calls. Use libraries like `requests_cache` to store responses temporarily.
*   **Using HTTP/2:** If the server supports HTTP/2, use the `HTTPX` library, which provides both synchronous and asynchronous support for HTTP/2.
*   **Compression:** Ensure the server is using compression (e.g., gzip) and that the `Accept-Encoding` header is set appropriately in the request.

### 3.2 Memory Management Considerations:

*   **Streaming Large Responses:** Use `stream=True` to avoid loading the entire response into memory.
*   **Closing Responses:** Close the response object after processing the data to release resources.  Use a `try...finally` block or a context manager to ensure the response is always closed.
*   **Iterating over Chunks:** When streaming, iterate over the response content in chunks using `response.iter_content()` or `response.iter_lines()` to process data in smaller pieces.

### 3.3 Bundle Size Optimization:

*   **Minimize Dependencies:** Only include the dependencies that are strictly necessary.

### 3.4 Lazy Loading Strategies:

*   **Lazy Initialization of Clients:**  Initialize the `requests` session or client only when it is first needed, rather than at application startup. This can improve startup time if the API client is not immediately required.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them:

*   **Man-in-the-Middle Attacks:** Always use HTTPS to encrypt communication and prevent eavesdropping. Verify SSL certificates unless absolutely necessary to disable verification.
*   **Cross-Site Scripting (XSS):** If displaying data from API responses in a web application, sanitize the data to prevent XSS attacks.
*   **Server-Side Request Forgery (SSRF):**  Avoid constructing URLs based on user input without proper validation. This can prevent attackers from making requests to internal resources.
*   **Exposure of Sensitive Information:** Store API keys, passwords, and other sensitive information securely using environment variables or a secrets management system.
*   **Denial of Service (DoS):**  Implement timeouts and rate limiting to prevent attackers from overwhelming the server with requests.

### 4.2 Input Validation Best Practices:

*   **Validating Input Data:** Validate all input data before sending it to the API. This can prevent injection attacks and other security vulnerabilities.
*   **Sanitizing Input Data:** Sanitize input data to remove potentially malicious characters or code.
*   **Using Prepared Statements:** When constructing database queries with data from API responses, use prepared statements to prevent SQL injection attacks.

### 4.3 Authentication and Authorization Patterns:

*   **Using Secure Authentication Schemes:** Use strong authentication schemes such as OAuth 2.0 or JWT (JSON Web Tokens) to protect API endpoints.
*   **Storing Credentials Securely:** Store API keys, passwords, and other credentials securely using environment variables, a secrets management system, or a hardware security module (HSM).
*   **Implementing Authorization:** Implement authorization to control which users or applications have access to specific API endpoints.
*   **Using HTTPS:** Always use HTTPS to encrypt communication and protect credentials during transmission.

### 4.4 Data Protection Strategies:

*   **Encrypting Sensitive Data:** Encrypt sensitive data at rest and in transit.
*   **Masking Sensitive Data:** Mask sensitive data in logs and error messages.
*   **Complying with Data Privacy Regulations:** Ensure compliance with relevant data privacy regulations such as GDPR and CCPA.

### 4.5 Secure API Communication:

*   **HTTPS:** Always use HTTPS for all API communication.
*   **TLS Versions:**  Ensure that the server and client support the latest TLS versions (TLS 1.2 or 1.3) and disable older, insecure versions (SSLv3, TLS 1.0, TLS 1.1).
*   **Cipher Suites:**  Configure the server to use strong cipher suites that provide forward secrecy and authenticated encryption.
*   **Certificate Pinning:**  Consider using certificate pinning to prevent man-in-the-middle attacks by verifying the server's SSL certificate against a known good certificate.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies:

*   **Testing Individual Functions:** Write unit tests for individual functions that make `requests` calls.
*   **Mocking `requests`:** Use mocking libraries like `unittest.mock` or `pytest-mock` to mock the `requests` library and isolate the code being tested.
*   **Testing Error Handling:** Write unit tests to verify that the code handles different types of errors correctly.
*   **Parametrizing Tests:** Use parametrization to test the same function with different inputs and expected outputs.
*   **Test Data:**  Create a set of test data (e.g., JSON files) to simulate API responses.

### 5.2 Integration Testing Approaches:

*   **Testing API Client Integrations:** Write integration tests to verify that the API client integrates correctly with other components of the application.
*   **Using a Test API Server:** Set up a test API server (e.g., using Flask or FastAPI) to simulate the real API and control the responses.
*   **Verifying Data Integrity:** Verify that the data returned by the API is processed correctly and stored in the database or other data store.

### 5.3 End-to-End Testing Recommendations:

*   **Testing Complete Workflows:** Write end-to-end tests to verify that complete workflows involving the API client work correctly.
*   **Using Automation Tools:** Use automation tools like Selenium or Playwright to automate end-to-end tests.

### 5.4 Test Organization Best Practices:

*   **Separating Tests:** Separate unit tests, integration tests, and end-to-end tests into different directories or files.
*   **Using a Test Runner:** Use a test runner like pytest to discover and run tests automatically.
*   **Following Test Naming Conventions:** Follow consistent test naming conventions to make it easier to understand the purpose of each test.

### 5.5 Mocking and Stubbing Techniques:

*   **Mocking the `requests` Library:** Use mocking to replace the `requests` library with a mock object that returns predefined responses.
*   **Using Mock Objects:** Create mock objects to simulate the behavior of external APIs.
*   **Stubbing Functions:** Use stubbing to replace functions with simplified versions that return predefined values.
*   **Using Context Managers for Mocking:** Use context managers to temporarily replace objects with mock objects during tests.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make:

*   **Not Handling Exceptions:** Ignoring exceptions raised by `requests` can lead to unexpected behavior and application crashes.
*   **Not Using Sessions:** Failing to use `requests.Session()` for multiple requests to the same host can result in performance degradation.
*   **Not Setting Timeouts:** Not setting timeouts on requests can lead to your application hanging indefinitely if a server is unresponsive.
*   **Disabling SSL Verification Unnecessarily:** Disabling SSL verification (`verify=False`) should only be done when absolutely necessary and with caution.
*   **Not Handling Rate Limits:** Not handling API rate limits can lead to the application being blocked by the API provider.
*   **Incorrectly Handling Character Encoding:**  Failing to properly handle character encoding when dealing with non-ASCII characters in request or response bodies.

### 6.2 Edge Cases to Be Aware Of:

*   **Handling Redirects:**  Be aware of how `requests` handles redirects by default (following them).  If you need to control redirect behavior, use the `allow_redirects` parameter.
*   **Dealing with Large Payloads:** When sending or receiving large payloads, use streaming to avoid memory issues.
*   **Handling Keep-Alive Connections:** Be aware that `requests` uses keep-alive connections by default, which can lead to issues if the server closes the connection unexpectedly.
*   **Proxy Configuration:** Properly configure proxies if your application needs to access the internet through a proxy server.

### 6.3 Version-Specific Issues:

*   **Changes in API:** Be aware of changes in the `requests` API between different versions.  Consult the release notes for each version to identify any breaking changes.
*   **Dependency Conflicts:**  Be aware of potential dependency conflicts between `requests` and other libraries in your project.  Use a virtual environment to isolate dependencies.

### 6.4 Compatibility Concerns:

*   **Python Versions:** Ensure that the version of `requests` you are using is compatible with the version of Python you are using.
*   **Operating Systems:** Be aware of potential compatibility issues between `requests` and different operating systems (e.g., Windows, Linux, macOS).

### 6.5 Debugging Strategies:

*   **Logging Requests and Responses:** Log detailed information about requests and responses, including URLs, headers, and bodies, to aid in debugging.
*   **Using Debugging Tools:** Use debugging tools such as `pdb` or `ipdb` to step through the code and inspect variables.
*   **Capturing Network Traffic:** Use network traffic analysis tools like Wireshark or tcpdump to capture and analyze network traffic.
*   **Using `httpbin.org` for Testing:** Use the `httpbin.org` service to test different types of requests and responses.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools:

*   **Virtual Environments:** Use virtual environments (e.g., `venv`, `virtualenv`) to isolate project dependencies.
*   **pip:** Use `pip` to install and manage dependencies.
*   **IDEs:** Use an IDE such as Visual Studio Code, PyCharm, or Sublime Text to write and debug code.
*   **Debugging Tools:** Use debugging tools such as `pdb` or `ipdb` to step through the code and inspect variables.

### 7.2 Build Configuration Best Practices:

*   **Using `requirements.txt`:** Use a `requirements.txt` file to specify project dependencies.
*   **Using `setup.py`:** Use a `setup.py` file to define the project metadata and package the code for distribution.
*   **Using a Build System:** Consider using a build system such as Make or Poetry to automate the build process.

### 7.3 Linting and Formatting Recommendations:

*   **Using a Linter:** Use a linter such as `flake8` or `pylint` to identify potential code quality issues.
*   **Using a Formatter:** Use a formatter such as `black` or `autopep8` to automatically format the code according to PEP 8 guidelines.
*   **Configuring the IDE:** Configure the IDE to use the linter and formatter automatically.

### 7.4 Deployment Best Practices:

*   **Using a Production Environment:** Deploy the application to a production environment that is separate from the development environment.
*   **Using a Process Manager:** Use a process manager such as Supervisor or systemd to manage the application process.
*   **Using a Load Balancer:** Use a load balancer to distribute traffic across multiple instances of the application.
*   **Monitoring the Application:** Monitor the application for errors and performance issues.

### 7.5 CI/CD Integration Strategies:

*   **Using a CI/CD System:** Use a CI/CD system such as Jenkins, GitLab CI, GitHub Actions, or CircleCI to automate the build, test, and deployment processes.
*   **Running Tests Automatically:** Configure the CI/CD system to run tests automatically on every commit.
*   **Deploying Automatically:** Configure the CI/CD system to deploy the application automatically to the production environment after the tests have passed.