---
description: This rule provides comprehensive best practices for developing Azure applications, covering code organization, security, performance, testing, and common pitfalls. It aims to improve code quality, security posture, and overall efficiency when working with the Azure ecosystem.
globs: *.az,*.tf,*.bicep,*.py,*.js,*.ts,*.json,*.yml,*.yaml,*.ps1,*.sh,*.cs,*.java
---
# Azure Library Best Practices and Coding Standards

This document outlines the recommended best practices for developing applications and infrastructure using Azure services. It covers various aspects including code organization, security, performance, testing, and tooling to ensure robust, scalable, and secure solutions.

## 1. Code Organization and Structure

A well-organized codebase is crucial for maintainability, scalability, and collaboration. The following guidelines provide a structured approach to organizing your Azure projects.

### Directory Structure Best Practices

Adopt a modular and logical directory structure based on the application's architecture and components.

*   **`src/`**: Contains the source code of your application.
    *   **`modules/`**: Groups related functionalities into independent modules.
    *   **`components/`**: Contains reusable UI or logical components.
    *   **`services/`**: Encapsulates external service integrations (e.g., Azure Storage, Cosmos DB).
    *   **`models/`**: Defines data models and schemas.
    *   **`utils/`**: Contains utility functions and helper classes.
*   **`infra/` or `terraform/` or `bicep/`**: Infrastructure-as-Code definitions (Terraform, Bicep, Azure Resource Manager templates).
*   **`config/`**: Configuration files for different environments (development, staging, production).
*   **`scripts/`**: Automation scripts for deployment, build processes, etc.
*   **`tests/`**: Unit, integration, and end-to-end tests.
*   **`docs/`**: Documentation for the project and its components.

Example:


my-azure-project/
├── src/
│   ├── modules/
│   │   ├── user-management/
│   │   │   ├── ...
│   │   ├── data-processing/
│   │   │   ├── ...
│   ├── components/
│   │   ├── button/
│   │   │   ├── ...
│   ├── services/
│   │   ├── storage-service.js
│   │   ├── cosmosdb-service.js
│   ├── models/
│   │   ├── user.js
│   ├── utils/
│   │   ├── logger.js
├── infra/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
├── config/
│   ├── development.json
│   ├── production.json
├── scripts/
│   ├── deploy.sh
│   ├── build.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
├── docs/
│   ├── README.md
├── .gitignore
├── package.json


### File Naming Conventions

Maintain consistency in file naming to improve readability and searchability.

*   Use descriptive names that reflect the file's purpose.
*   Use a consistent case (e.g., camelCase or kebab-case).
*   Use appropriate file extensions (e.g., `.js`, `.ts`, `.py`, `.tf`, `.bicep`).
*   For React components, use `ComponentName.jsx` or `ComponentName.tsx`.

Example:

*   `user-service.js` (for a user service module)
*   `UserProfile.jsx` (for a user profile component)
*   `storage-account.tf` (for a Terraform file defining a storage account)

### Module Organization

Divide the application into independent and reusable modules based on functionality.

*   Each module should have a clear responsibility and a well-defined interface.
*   Minimize dependencies between modules to promote loose coupling.
*   Consider using a module bundler (e.g., Webpack, Parcel) to manage dependencies and optimize the build process.

Example:

A `user-management` module could contain components and services related to user authentication, authorization, and profile management.

### Component Architecture

For UI-based applications, adopt a component-based architecture (e.g., React, Angular, Vue.js).

*   Divide the UI into reusable components with clear inputs and outputs.
*   Follow the principles of single responsibility and separation of concerns.
*   Use a component library (e.g., Material UI, Ant Design) to accelerate development and maintain consistency.

### Code Splitting Strategies

Improve initial load time by splitting the application into smaller chunks that can be loaded on demand.

*   Use dynamic imports to load modules only when they are needed.
*   Configure your module bundler to create separate chunks for different parts of the application.
*   Consider using route-based code splitting for single-page applications.

## 2. Common Patterns and Anti-patterns

Understanding common patterns and anti-patterns is essential for building efficient, maintainable, and reliable Azure applications.

### Design Patterns Specific to Azure

*   **Retry Pattern:** Implement retry logic to handle transient failures when interacting with Azure services.
*   **Circuit Breaker Pattern:** Prevent cascading failures by temporarily blocking access to a service that is experiencing problems.
*   **Queue-Based Load Leveling:** Use queues (e.g., Azure Queue Storage, Azure Service Bus) to decouple components and handle bursts of traffic.
*   **Cache-Aside Pattern:** Use a cache (e.g., Azure Cache for Redis) to improve performance by storing frequently accessed data.
*   **Gateway Aggregation Pattern:** Expose multiple backend services through a single API gateway (e.g., Azure API Management).

### Recommended Approaches for Common Tasks

*   **Configuration Management:** Use Azure App Configuration to manage application settings and feature flags.
*   **Secret Management:** Use Azure Key Vault to securely store and manage secrets, keys, and certificates.
*   **Logging and Monitoring:** Use Azure Monitor to collect and analyze logs, metrics, and traces.
*   **Identity and Access Management:** Use Azure Active Directory (Azure AD) for authentication and authorization.
*   **Data Storage:** Choose the appropriate Azure storage service based on your data requirements (e.g., Azure Blob Storage for unstructured data, Azure Cosmos DB for NoSQL data, Azure SQL Database for relational data).

### Anti-Patterns and Code Smells to Avoid

*   **Hardcoding Secrets:** Never hardcode secrets or API keys in your code. Use Azure Key Vault to manage them securely.
*   **Ignoring Errors:** Always handle errors gracefully and log relevant information for debugging.
*   **Over-Engineering:** Avoid unnecessary complexity and focus on delivering business value.
*   **Long-Running Transactions:** Keep transactions short and avoid holding resources for extended periods.
*   **Tight Coupling:** Design components and modules to be loosely coupled to improve maintainability and testability.
*   **Monolithic Functions:** Avoid creating large, complex functions. Break them down into smaller, more manageable units.

### State Management Best Practices

*   For serverless functions, design for statelessness whenever possible. Store state in external storage services like Azure Cosmos DB or Azure Storage.
*   For web applications, use appropriate state management techniques (e.g., cookies, sessions, local storage, or dedicated state management libraries like Redux or Zustand).
*   Consider using distributed caching (e.g., Azure Cache for Redis) to improve performance and scalability.

### Error Handling Patterns

*   Use try-catch blocks to handle exceptions gracefully.
*   Log errors with sufficient detail to aid in debugging.
*   Implement a centralized error handling mechanism to catch and handle unhandled exceptions.
*   Provide meaningful error messages to the user.
*   Use Polly library for implementing resilience policies like retry, circuit breaker, and timeout.

## 3. Performance Considerations

Optimizing performance is crucial for delivering a responsive and scalable Azure application.

### Optimization Techniques

*   **Caching:** Implement caching at various levels (e.g., browser, server, database) to reduce latency and improve performance.
*   **Compression:** Enable compression (e.g., Gzip) to reduce the size of HTTP responses.
*   **Connection Pooling:** Use connection pooling to reuse database connections and reduce connection overhead.
*   **Content Delivery Network (CDN):** Use Azure CDN to cache static assets closer to users.
*   **Database Optimization:** Optimize database queries and indexes to improve query performance.
*   **Asynchronous Operations:** Use asynchronous operations to avoid blocking the main thread.

### Memory Management

*   **Dispose Resources:** Dispose of disposable resources (e.g., database connections, file streams) properly to avoid memory leaks.
*   **Garbage Collection:** Be aware of garbage collection behavior and avoid creating unnecessary objects.
*   **Large Object Heap (LOH):** Be mindful of the LOH and avoid allocating large objects frequently.

### Rendering Optimization

*   **Virtualization:** Use virtualization techniques (e.g., React Virtualized, Angular CDK Virtual Scroll) to efficiently render large lists of data.
*   **Debouncing and Throttling:** Use debouncing and throttling to limit the frequency of event handlers.
*   **Lazy Loading:** Load images and other resources only when they are visible on the screen.

### Bundle Size Optimization

*   **Tree Shaking:** Use tree shaking to remove unused code from your bundles.
*   **Code Splitting:** Split your code into smaller chunks that can be loaded on demand.
*   **Minification:** Minify your code to reduce its size.
*   **Image Optimization:** Optimize images by compressing them and using appropriate formats.

### Lazy Loading Strategies

*   **Component-Based Lazy Loading:** Load components only when they are needed.
*   **Route-Based Lazy Loading:** Load modules associated with specific routes only when those routes are accessed.
*   **Image Lazy Loading:** Load images only when they are scrolled into view.

## 4. Security Best Practices

Security should be a top priority when developing Azure applications.

### Common Vulnerabilities and How to Prevent Them

*   **SQL Injection:** Use parameterized queries or an ORM to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** Sanitize user input and output to prevent XSS attacks.
*   **Cross-Site Request Forgery (CSRF):** Use anti-CSRF tokens to prevent CSRF attacks.
*   **Authentication and Authorization Flaws:** Implement robust authentication and authorization mechanisms.
*   **Insecure Direct Object References (IDOR):** Validate that users have access to the resources they are requesting.
*   **Security Misconfiguration:** Follow security best practices for configuring Azure services.
*   **Sensitive Data Exposure:** Protect sensitive data by encrypting it at rest and in transit.

### Input Validation

*   Validate all user input on both the client and server sides.
*   Use a schema validation library (e.g., Joi, Yup) to define and enforce data validation rules.
*   Sanitize user input to remove potentially malicious characters.

### Authentication and Authorization Patterns

*   Use Azure Active Directory (Azure AD) for authentication and authorization.
*   Implement role-based access control (RBAC) to restrict access to resources based on user roles.
*   Use multi-factor authentication (MFA) to enhance security.
*   Follow the principle of least privilege.
*   Leverage Managed Identities for Azure resources to securely access other Azure services.

### Data Protection Strategies

*   Encrypt sensitive data at rest using Azure Storage Service Encryption (SSE) or Azure Disk Encryption.
*   Encrypt data in transit using TLS/SSL.
*   Use Azure Key Vault to store and manage encryption keys.
*   Implement data masking and tokenization to protect sensitive data.

### Secure API Communication

*   Use HTTPS for all API communication.
*   Implement API authentication using API keys, tokens, or OAuth 2.0.
*   Use rate limiting to prevent abuse.
*   Use Azure API Management to manage and secure your APIs.

## 5. Testing Approaches

Thorough testing is essential for ensuring the quality and reliability of your Azure applications.

### Unit Testing Strategies

*   Write unit tests for individual components and functions.
*   Use a unit testing framework (e.g., Jest, Mocha, JUnit, pytest).
*   Aim for high code coverage.
*   Follow the Arrange-Act-Assert pattern.
*   Use mocks and stubs to isolate the code under test.

### Integration Testing

*   Write integration tests to verify the interaction between different components and services.
*   Test the integration with Azure services (e.g., Azure Storage, Cosmos DB).
*   Use a testing environment that closely resembles the production environment.

### End-to-End Testing

*   Write end-to-end tests to verify the entire application workflow.
*   Use a testing framework (e.g., Cypress, Selenium, Playwright).
*   Automate end-to-end tests as part of your CI/CD pipeline.

### Test Organization

*   Organize tests into a clear directory structure.
*   Use descriptive test names.
*   Group related tests together.

### Mocking and Stubbing

*   Use mocks to simulate the behavior of external dependencies.
*   Use stubs to provide predefined responses for API calls.
*   Use a mocking framework (e.g., Mockito, Jest Mocks).

## 6. Common Pitfalls and Gotchas

Be aware of common pitfalls and gotchas to avoid making mistakes when developing Azure applications.

### Frequent Mistakes Developers Make

*   **Insufficient Resource Sizing:** Underestimating the required resources can lead to performance issues.
*   **Ignoring Azure Service Limits:** Being unaware of Azure service limits can cause unexpected errors.
*   **Improper Error Handling:** Inadequate error handling can make it difficult to diagnose and resolve issues.
*   **Lack of Monitoring:** Failing to monitor the application's performance and health can lead to undetected problems.
*   **Neglecting Security:** Overlooking security best practices can expose the application to vulnerabilities.

### Edge Cases to Be Aware Of

*   **Network Latency:** Be aware of network latency when interacting with Azure services.
*   **Transient Faults:** Implement retry logic to handle transient faults.
*   **Resource Contention:** Be aware of resource contention and design the application to handle it gracefully.
*   **Data Consistency:** Understand the data consistency models of Azure storage services.

### Version-Specific Issues

*   Be aware of breaking changes in Azure SDKs and APIs.
*   Keep your dependencies up to date.
*   Test your application with different versions of Azure services.

### Compatibility Concerns

*   Ensure that your application is compatible with the target Azure environment.
*   Test your application on different browsers and devices.

### Debugging Strategies

*   Use Azure Monitor to collect and analyze logs, metrics, and traces.
*   Use remote debugging to debug your application running in Azure.
*   Use logging to track the flow of execution and identify errors.
*   Use breakpoints to pause execution and inspect variables.

## 7. Tooling and Environment

Choosing the right tools and environment can significantly improve your development experience.

### Recommended Development Tools

*   **Visual Studio Code:** A popular code editor with excellent support for Azure development.
*   **Azure CLI:** A command-line tool for managing Azure resources.
*   **Azure PowerShell:** A PowerShell module for managing Azure resources.
*   **Terraform:** An infrastructure-as-code tool for provisioning and managing Azure resources.
*   **Bicep:** A declarative language for deploying Azure resources.
*   **Azure SDKs:** Libraries for interacting with Azure services from various programming languages.

### Build Configuration

*   Use a build automation tool (e.g., Azure DevOps, Jenkins, GitHub Actions) to automate the build process.
*   Configure your build process to run tests, linters, and code formatters.
*   Use environment variables to configure your application for different environments.

### Linting and Formatting

*   Use a linter (e.g., ESLint, Pylint) to enforce code style guidelines.
*   Use a code formatter (e.g., Prettier, Black) to automatically format your code.
*   Configure your editor to automatically run the linter and formatter on save.

### Deployment Best Practices

*   Use Infrastructure-as-Code (IaC) to automate the deployment of Azure resources.
*   Use a deployment pipeline (e.g., Azure Pipelines, GitHub Actions) to automate the deployment process.
*   Use blue-green deployments or canary deployments to minimize downtime.
*   Implement rollback strategies to quickly revert to a previous version if necessary.

### CI/CD Integration

*   Integrate your CI/CD pipeline with Azure DevOps, GitHub Actions, or other CI/CD tools.
*   Automate the build, test, and deployment processes.
*   Use environment variables to configure your application for different environments.
*   Implement automated testing to catch errors early in the development cycle.

By following these best practices, you can build robust, scalable, secure, and maintainable Azure applications.