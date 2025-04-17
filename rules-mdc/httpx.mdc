---
description: This rule provides comprehensive best practices for using the httpx library, covering code organization, performance, security, testing, and common pitfalls. Adhering to these guidelines will improve code quality, maintainability, and security when working with httpx.
globs: *.py
---
# httpx Best Practices

This document outlines recommended practices for using the `httpx` library in Python. Following these guidelines will help ensure maintainable, performant, and secure code.

## Library Information:

- Name: httpx
- Tags: web, python, http-client, async

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

- **Dedicated Modules:** Organize httpx-related code into dedicated modules or packages.
- **Configuration:** Separate configuration settings (timeouts, headers, proxies) from the core logic.
- **Data Models:** Define data models (e.g., using `dataclasses` or `pydantic`) to represent request and response data.

Example:


my_project/
├── httpx_client/
│   ├── __init__.py
│   ├── client.py      # httpx client setup, session management
│   ├── utils.py       # Utility functions for request/response handling
│   ├── models.py      # Data models for requests and responses
│   └── exceptions.py  # Custom exceptions
├── ...


### 1.2 File Naming Conventions

- Use descriptive names that reflect the file's purpose (e.g., `api_client.py`, `request_utils.py`).
- Follow PEP 8 conventions (snake_case).

### 1.3 Module Organization

- **Layered Architecture:** Consider a layered architecture with modules for:
    - **Client Initialization:**  Handles httpx client setup (connection pooling, timeouts).
    - **Request Building:**  Constructs httpx Request objects.
    - **Response Handling:** Parses and validates responses, handles errors.
- **Abstraction:** Abstract away direct httpx calls behind service functions or classes.

### 1.4 Component Architecture

- **Reusable Components:** Design reusable components for common tasks like:
    - **Authentication:** Implementing custom authentication flows.
    - **Rate Limiting:** Managing API rate limits.
    - **Retry Logic:** Handling transient errors with retry mechanisms.

### 1.5 Code Splitting Strategies

- **Feature-Based Splitting:** Split code based on features or API endpoints.
- **Abstraction:** Use interfaces or abstract base classes for httpx clients to facilitate mocking and testing.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

- **Factory Pattern:** Use a factory function or class to create httpx clients with different configurations.
- **Strategy Pattern:** Implement different request strategies based on the API requirements.

### 2.2 Recommended Approaches for Common Tasks

- **Client Instance Usage:** Always use `httpx.Client()` or `httpx.AsyncClient()` for making requests. This enables connection pooling.
- **Context Managers:** Use the client as a context manager (`with httpx.Client() as client:`) or explicitly close the connection pool (`client.close()`).
- **Timeouts:** Always set timeouts to prevent hanging requests.
- **Error Handling:** Use `try-except` blocks to handle `HTTPStatusError` and `RequestError`.
- **Async Support:** Utilize `httpx.AsyncClient()` for concurrent requests.
- **Configuration Sharing:** Configure default headers, parameters, and authentication at the client level.

python
import httpx

with httpx.Client(timeout=10.0, headers={'User-Agent': 'MyApp/1.0'}) as client:
    try:
        response = client.get('https://example.com/api/data')
        response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
        data = response.json()
        # Process data
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e}")
    except httpx.RequestError as e:
        print(f"Request Error: {e}")


### 2.3 Anti-patterns and Code Smells

- **Global Client Instances:** Avoid creating global httpx client instances without proper management of their lifecycle.
- **Ignoring Errors:** Never ignore exceptions raised by httpx. Always handle them appropriately.
- **Hardcoding URLs:** Avoid hardcoding URLs directly in the code. Use configuration files or environment variables.

### 2.4 State Management

- **Statelessness:** Design components to be as stateless as possible.
- **Dependency Injection:** Use dependency injection to provide httpx clients to components that need them.

### 2.5 Error Handling Patterns

- **Custom Exceptions:** Define custom exception classes to represent specific httpx-related errors.
- **Logging:** Log errors and warnings with sufficient context.
- **Retry Logic:** Implement retry logic for transient errors.

## 3. Performance Considerations

### 3.1 Optimization Techniques

- **Connection Pooling:** httpx automatically uses connection pooling when using `httpx.Client()` or `httpx.AsyncClient()`.  Ensure you are leveraging this by reusing client instances.
- **HTTP/2:** Enable HTTP/2 support if the server supports it.
- **Keep-Alive:** Ensure keep-alive connections are enabled.
- **Asynchronous Operations:** Use `httpx.AsyncClient()` for concurrent I/O-bound operations.

### 3.2 Memory Management

- **Streaming Responses:** Use response streaming (`response.iter_bytes()`, `response.iter_text()`) for large responses to avoid loading the entire response into memory.
- **Chunked Uploads:** Use chunked uploads for large file uploads.

### 3.3 Bundle Size Optimization

- **Tree Shaking:** Ensure your build tools perform tree shaking to remove unused httpx code.

### 3.4 Lazy Loading

- **On-Demand Initialization:** Initialize httpx clients only when they are needed.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

- **Injection Attacks:** Sanitize user inputs to prevent injection attacks.
- **Man-in-the-Middle Attacks:** Always use HTTPS and verify SSL certificates.
- **Sensitive Data Exposure:** Avoid logging sensitive data (API keys, passwords).

### 4.2 Input Validation

- **Validate Request Data:** Validate all data before sending it to the API.
- **Validate Response Data:** Validate responses from the API.

### 4.3 Authentication and Authorization

- **Secure Authentication:** Use secure authentication mechanisms (OAuth 2.0, JWT).
- **Least Privilege:** Grant only the necessary permissions to users.

### 4.4 Data Protection

- **Encryption:** Encrypt sensitive data at rest and in transit.
- **Data Masking:** Mask sensitive data in logs and error messages.

### 4.5 Secure API Communication

- **HTTPS:** Always use HTTPS.
- **TLS 1.3:** Use TLS 1.3 or higher.
- **Certificate Pinning:** Consider certificate pinning for added security (advanced).

## 5. Testing Approaches

### 5.1 Unit Testing

- **Mocking:** Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`) to mock httpx calls.
- **Isolate Components:** Isolate components that use httpx for unit testing.

### 5.2 Integration Testing

- **Test Against Real APIs:** Perform integration tests against real APIs (with appropriate mocking or stubbing).

### 5.3 End-to-End Testing

- **Simulate User Flows:** Simulate end-to-end user flows that involve httpx calls.

### 5.4 Test Organization

- **Separate Test Files:** Create separate test files for each module or component.

### 5.5 Mocking and Stubbing Techniques

- **Mock Responses:** Mock httpx responses to simulate different API scenarios.
- **Patch httpx Functions:** Patch httpx functions to control their behavior during tests.
- **RESPX:** Use RESPX for advanced mocking of httpx requests and responses.

python
import httpx
import pytest
import respx

@respx.mock
async def test_get_data():
    respx.get("https://example.com/api/data").mock(return_value=httpx.Response(200, json={"key": "value"}))
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com/api/data")
        assert response.status_code == 200
        assert response.json() == {"key": "value"}


## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

- **Forgetting to Close Clients:** Always close httpx clients to release resources.
- **Ignoring Timeouts:** Not setting timeouts can lead to hanging requests.
- **Incorrect Error Handling:** Improper error handling can mask critical errors.
- **Misunderstanding Connection Pooling:** Not reusing client instances negates connection pooling benefits.
- **Not Handling Asynchronous Context Correctly**: Using `await` improperly when using `AsyncClient()`.

### 6.2 Edge Cases

- **Character Encoding Issues:** Handle character encoding issues when processing text responses.
- **SSL Certificate Errors:** Handle SSL certificate errors gracefully.
- **Network Connectivity Issues:** Implement robust error handling for network connectivity issues.

### 6.3 Version-Specific Issues

- **Consult httpx Documentation:** Refer to the httpx documentation for version-specific issues and compatibility notes.

### 6.4 Compatibility Concerns

- **Asynchronous Libraries:** Ensure compatibility with other asynchronous libraries (e.g., asyncio).
- **Third-Party Packages:** Be aware of compatibility issues with third-party packages that use httpx.

### 6.5 Debugging Strategies

- **Logging:** Use logging to trace httpx requests and responses.
- **Debugging Tools:** Use debugging tools (e.g., pdb, ipdb) to inspect httpx code.
- **Wireshark/tcpdump:** Use network analysis tools (Wireshark, tcpdump) to capture and analyze network traffic.

## 7. Tooling and Environment

### 7.1 Recommended Tools

- **Development Environment:** Use a modern IDE (e.g., VS Code, PyCharm).
- **Dependency Management:** Use a dependency management tool (e.g., pipenv, poetry).
- **Testing Framework:** Use a testing framework (e.g., pytest, unittest).
- **Mocking Libraries:** Use mocking libraries (e.g., `unittest.mock`, `pytest-mock`, RESPX).

### 7.2 Build Configuration

- **Reproducible Builds:** Ensure reproducible builds using dependency locking.

### 7.3 Linting and Formatting

- **PEP 8 Compliance:** Use a linter (e.g., flake8, pylint) to enforce PEP 8 compliance.
- **Code Formatting:** Use a code formatter (e.g., black, autopep8) to automatically format code.

### 7.4 Deployment

- **Containerization:** Use containerization (e.g., Docker) to ensure consistent deployments.
- **Environment Variables:** Configure httpx clients using environment variables.

### 7.5 CI/CD Integration

- **Automated Testing:** Integrate automated testing into your CI/CD pipeline.
- **Code Analysis:** Integrate code analysis tools into your CI/CD pipeline.