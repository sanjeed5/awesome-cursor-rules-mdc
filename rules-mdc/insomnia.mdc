---
description: This rule file provides best practices for using the Insomnia API Client, including project organization, environment management, testing, and collaboration to improve API development workflows.
globs: *.json,*.yaml,*.yml,*.graphql,*.proto
---
- **Project Organization and Structure:**
  - **Workspace Structure:** Organize requests into workspaces based on projects or functional areas. Within workspaces, use folders to group related requests (e.g., authentication, user management, product catalog).
  - **Environment Management:** Leverage Insomnia's environment variables for managing configurations (API keys, base URLs, authentication tokens) across different environments (development, staging, production).  Use environment variables to store sensitive data and secrets instead of hardcoding them directly into requests. Create separate environments for each stage of the API lifecycle (development, testing, staging, production). This allows you to easily switch between different configurations without modifying your requests.
  - **File Naming Conventions:**  Use descriptive file names for Insomnia collections (e.g., `users-api.insomnia.json`, `product-management.insomnia.yaml`). Ensure that your naming convention is clear and consistent. Name files according to their function and the API they're targeting. Using date-based versioning can also be valuable.
  - **Collection Splitting:** For large API definitions, consider splitting collections into smaller, more manageable files based on API modules or resources. Use Insomnia's import/export functionality to combine them when needed.
  - **Git Integration:** Store your Insomnia collections and environment files in a Git repository to track changes, collaborate with team members, and implement version control. Using Git also gives you a backup of your configuration.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Environment Overrides:** Use environment overrides to customize settings for specific requests without modifying the base environment.
    - **Chained Requests:**  Utilize chained requests to pass data from one request to another, enabling complex workflows (e.g., creating a user, then retrieving the user's details).
    - **Dynamic Values:** Generate dynamic values (e.g., timestamps, UUIDs) using Insomnia's built-in functions to create realistic test data.
  - **Recommended Approaches:**
    - **API Testing:** Use Insomnia to test HTTP-based RESTful APIs as well as GraphQL APIs. Take advantage of API testing as part of your testing strategy to test your application's core business rules and help deliver better software faster.
    - **GraphQL Support:** Leverage Insomnia's built-in GraphQL support for constructing and executing GraphQL queries.
    - **Authentication Handling:** Configure authentication (API keys, OAuth 2.0, JWT) at the environment or request level for consistent authentication across all requests.
  - **Anti-patterns:**
    - **Hardcoding Sensitive Data:** Avoid storing sensitive information directly in request bodies or headers. Use environment variables instead.
    - **Ignoring Response Validation:** Always validate API responses (status codes, headers, data) to ensure that the API is functioning correctly.
  - **State Management:** Insomnia itself is stateless; however, you can simulate stateful interactions using chained requests and environment variables to store and reuse data across multiple requests.
  - **Error Handling:** Use Insomnia's response inspection tools to analyze error responses. Define specific tests to assert that error responses are handled correctly.

- **Performance Considerations:**
  - **Request Optimization:** Minimize request payload size and optimize request headers to improve API performance.
  - **Connection Pooling:** Insomnia automatically manages connection pooling for efficient API communication.
  - **Response Caching (Client-Side):** Consider using a proxy or interceptor to cache API responses locally for faster testing and development.

- **Security Best Practices:**
  - **Vulnerability Prevention:**
    - **Rate Limiting:**  Simulate rate limiting scenarios to test your API's resilience to abuse.
    - **Input Validation:** Thoroughly validate all inputs (headers, query parameters, request body) to prevent injection attacks.
  - **Input Validation:** Utilize Insomnia's request body editor to ensure that the data being sent is correctly formatted and validated against the API's expected schema.
  - **Authentication and Authorization:**
    - **OAuth 2.0 Flows:** Implement OAuth 2.0 flows within Insomnia to test API authorization.
    - **JWT Validation:** Validate JWT tokens returned by the API to ensure their integrity.
  - **Data Protection:**
    - **HTTPS:** Always use HTTPS for secure API communication. Insomnia automatically handles HTTPS requests.
    - **End-to-End Encryption:** Ensure that sensitive data is encrypted both in transit and at rest. Consider using a VPN for added security.
  - **Security Helpers**: Take advantage of security helpers, code creation, and environment variables.

- **Testing Approaches:**
  - **Unit Testing:** While Insomnia isn't a unit testing framework, you can use it to create individual requests that target specific API endpoints and validate their responses.
  - **Integration Testing:** Use Insomnia to test the interaction between different API services and components.
  - **End-to-End Testing:** Create comprehensive test suites in Insomnia to simulate real-world user scenarios and validate the entire API workflow.
  - **Test Organization:** Organize tests into folders based on API resources or functional areas.
  - **Mocking and Stubbing:**  Insomnia's mocking capabilities are essential during early stages of development or when dependent services are not available. Kong Insomnia’s mocking capabilities allow teams to simulate API responses without needing access to the actual API or back-end systems, facilitating parallel development and ensuring seamless integration testing. Kong Insomnia offers the option to mock on cloud or local for data sensitive customers.
  - **Test Automation:** Integrate Insomnia with CI/CD pipelines to automate API testing as part of your development workflow.  Utilize the Insomnia CLI or Newman (Postman's CLI tool) to run your Insomnia collections as part of your automated test suite.
  - **Assertions:**
    - Validate the structure and content of API responses using Insomnia's response inspection tools and custom JavaScript assertions.
    - Verify HTTP status codes, headers, and response times to ensure that the API is performing as expected.

- **Common Pitfalls and Gotchas:**
  - **Incorrect Environment Variables:**  Double-check that environment variables are correctly configured and referenced in your requests.
  - **Missing Authentication Headers:** Ensure that all authenticated requests include the necessary authentication headers.
  - **Invalid Request Payloads:**  Validate that your request payloads are correctly formatted and match the API's expected schema.
  - **Rate Limit Exceeded Errors:**  Be aware of API rate limits and implement appropriate retry mechanisms.
  - **Data Type Mismatches:**  Ensure that the data types in your request payloads match the API's expected data types.
  - **Handling Large Responses:** Insomnia might struggle with extremely large API responses. Consider using a tool like `jq` to process large JSON responses.
  - **Import/Export Issues:** Be aware of potential compatibility issues when importing or exporting Insomnia collections. Always test imported collections to ensure that they function correctly.
  - **Version Compatibility:** Test Insomnia against the versions of APIs that you intend to target. Old API versions might have compatibility issues.
  - **Debugging strategies:**
    - **Response Inspection:** Use Insomnia's response inspection tools to examine the API's responses. Check the status code, headers, and body for errors or unexpected data.
    - **Request Logging:** Enable request logging to capture detailed information about the API requests being sent. This can help you diagnose issues with the request payload, headers, or authentication.
    - **Console Output:** Use the `console.log()` function in Insomnia's request scripts to output debug messages to the console. This can help you track the flow of data and identify potential problems.
    - **Network Monitoring:** Use a network monitoring tool (e.g., Wireshark) to capture and analyze the network traffic between Insomnia and the API server. This can help you identify issues with the network connection, SSL/TLS configuration, or API server behavior.

- **Tooling and Environment:**
  - **Recommended Tools:**
    - Insomnia Designer: Use Insomnia Designer to visually design and document your APIs.
    - Insomnia CLI: Use the Insomnia CLI to automate API testing and integrate it with your CI/CD pipelines.
    - JSON Formatter: Use a JSON formatter to validate and format your JSON request payloads.
    - YAML Linter: Use a YAML linter to validate your YAML API definitions.
  - **Build Configuration:** Store your Insomnia collections and environment files in a Git repository to track changes, collaborate with team members, and implement version control.
  - **Linting and Formatting:** Enforce consistent code style and API design by using linting and formatting tools.
  - **Deployment:**  Store your API keys and other sensitive information as environment variables on your deployment server. Use a secure deployment pipeline to ensure that your Insomnia collections and environment files are deployed correctly.
  - **CI/CD Integration:** Use Insomnia CLI to integrate API testing into your CI/CD pipelines. Configure automated API tests to run whenever code is committed or deployed.

- **Additional Best Practices:**
  - **Documentation:** Maintain clear and up-to-date documentation for your API collections and environments.
  - **Collaboration:** Use Insomnia's team collaboration features to share API collections and environments with your team members.
  - **Regular Updates:** Keep Insomnia up-to-date to take advantage of the latest features and security patches.
  - **Modular Design:** Break down large API collections into smaller, more manageable modules to improve maintainability.

By following these best practices, you can leverage Insomnia to streamline your API development workflow, improve API quality, and ensure the security of your API services.

- **References**
  - @file:./insomnia_example_rule.mdc (Example rule for chaining)