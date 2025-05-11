---
description: This rule provides best practices for developing and managing infrastructure and applications on Google Cloud Platform (GCP), encompassing code organization, security, performance, and deployment strategies.
globs: *.tf,*.py,*.js,*.go,*.java,*.proto,*.sh,*.yaml,*.yml
---
---
## GCP Library Best Practices

This document outlines best practices for developing and managing applications and infrastructure on Google Cloud Platform (GCP). It covers various aspects, including code organization, common patterns, performance, security, testing, common pitfalls, and tooling.

### 1. Code Organization and Structure

-   **Follow a Standard Module Structure (especially for Terraform)**:
    -   Start every module with a `main.tf` file. This file contains the primary resources defined in the module.
    -   Include a `README.md` file for module documentation. This should explain the module's purpose, usage, inputs, and outputs.
    -   Group related resources logically in separate files (e.g., `network.tf`, `compute.tf`, `storage.tf`).
    -   Use folders to logically group files.
-   **Directory Structure Best Practices:**
    -   `modules/`: Contains reusable Terraform modules. Each module should encapsulate a specific set of resources or functionality.
    -   `environments/`: Contains environment-specific configurations (e.g., `dev/`, `staging/`, `prod/`). Each environment directory should contain its own `terraform.tfvars` and `backend.tf` (or equivalent).
    -   `scripts/`: Contains utility scripts (e.g., Python, Bash) for automation.
    -   `docs/`: Contains documentation for the project.
    -   `examples/`: Contains example configurations or usage scenarios.
-   **File Naming Conventions:**
    -   Use descriptive names for Terraform files (e.g., `vpc.tf`, `firewall.tf`, `iam.tf`).
    -   Use underscores for resource names and avoid repeating resource types in names (e.g., `instance_web_server` instead of `web_server_instance`).
    -   Use consistent naming conventions for variables and outputs.
-   **Module Organization:**
    -   Keep modules small and focused. Each module should have a single responsibility.
    -   Use input variables to parameterize modules. This makes them reusable and configurable.
    -   Define outputs for modules to expose important information. This allows other modules or configurations to consume the module's results.
-   **Component Architecture (for application code):**
    -   Adopt a layered architecture (e.g., presentation, business logic, data access).
    -   Use dependency injection to decouple components and improve testability.
    -   Design components for reusability and maintainability.
    -   Use well-defined interfaces between components.
-   **Code Splitting Strategies (especially for client-side code):**
    -   Split code into logical chunks based on functionality or features.
    -   Use lazy loading to load code only when it's needed. This can improve initial page load time.
    -   Consider using code splitting tools like Webpack or Parcel.

### 2. Common Patterns and Anti-patterns

-   **Design Patterns Specific to GCP:**
    -   **Service Facade:** Abstract complex GCP service interactions behind a simple interface.
    -   **Pub/Sub Fanout:** Use Cloud Pub/Sub to distribute messages to multiple subscribers for parallel processing.
    -   **Cloud Functions Chaining:** Chain Cloud Functions together to create complex workflows.
    -   **Idempotent Operations:** Ensure operations can be safely retried without unintended side effects, especially crucial in distributed systems like GCP.
-   **Recommended Approaches for Common Tasks:**
    -   **Configuration Management:** Use environment variables or Cloud KMS for storing sensitive configuration data.
    -   **Secret Management:** Use Cloud Secret Manager to securely store and manage secrets.
    -   **Data Storage:** Choose the appropriate storage solution based on the data type and access patterns (e.g., Cloud Storage for objects, Cloud SQL for relational data, Cloud Datastore for NoSQL data).
    -   **Logging:** Use Cloud Logging for centralized logging and monitoring.
    -   **Monitoring:** Utilize Cloud Monitoring and Cloud Trace for performance monitoring and troubleshooting.
    -   **Identity and Access Management (IAM):**  Adopt the principle of least privilege. Grant users and service accounts only the necessary permissions.
-   **Anti-patterns and Code Smells to Avoid:**
    -   **Hardcoding Credentials:** Never hardcode credentials in code or configuration files. Use environment variables, secret management, or IAM roles.
    -   **Overly Complex IAM Roles:** Avoid creating overly complex IAM roles with too many permissions. This can lead to security vulnerabilities.
    -   **Ignoring Error Handling:** Always handle errors gracefully and provide informative error messages.
    -   **Long-Running Processes in Cloud Functions:**  Cloud Functions have time limits. Offload long-running processes to other services like Cloud Run or Compute Engine.
    -   **Lack of Monitoring:** Failing to monitor your application can lead to undetected performance issues and errors.
    -   **Ignoring Security Updates:**  Regularly update your dependencies and software to address security vulnerabilities.
-   **State Management Best Practices:**
    -   **Use Terraform State:** Store Terraform state remotely in Cloud Storage for collaboration and version control.
    -   **Stateless Applications:** Design applications to be stateless whenever possible. If state is required, store it in a persistent storage solution like Cloud SQL or Cloud Datastore.
    -   **Avoid Local Storage:** Do not rely on local storage for persistent data. Use Cloud Storage or other persistent storage solutions.
-   **Error Handling Patterns:**
    -   **Centralized Error Handling:** Implement a centralized error handling mechanism to handle exceptions consistently.
    -   **Logging and Monitoring:** Log all errors to Cloud Logging and set up alerts for critical errors in Cloud Monitoring.
    -   **Retry Mechanisms:** Implement retry mechanisms for transient errors.
    -   **Circuit Breaker Pattern:** Use the circuit breaker pattern to prevent cascading failures in distributed systems.

### 3. Performance Considerations

-   **Optimization Techniques:**
    -   **Caching:** Implement caching at various levels (e.g., browser, CDN, server-side) to reduce latency and improve performance. Consider using Cloud CDN or Memorystore.
    -   **Load Balancing:** Use Cloud Load Balancing to distribute traffic across multiple instances for high availability and scalability.
    -   **Database Optimization:** Optimize database queries and indexing to improve performance. Consider using Cloud SQL Insights for query analysis.
    -   **Asynchronous Operations:** Use asynchronous operations for long-running tasks to prevent blocking the main thread.
-   **Memory Management (for application code):**
    -   **Efficient Data Structures:** Choose appropriate data structures to minimize memory usage.
    -   **Garbage Collection:** Understand how garbage collection works in your chosen language and optimize code to minimize memory leaks.
    -   **Resource Pooling:** Use resource pooling to reuse expensive resources like database connections.
-   **Rendering Optimization (for web applications):**
    -   **Minimize DOM Manipulation:** Minimize DOM manipulation to improve rendering performance.
    -   **Use Virtual DOM:** Use a virtual DOM library like React or Vue.js to optimize rendering updates.
    -   **Optimize Images:** Optimize images for web use by compressing them and using appropriate formats (e.g., WebP).
-   **Bundle Size Optimization (for web applications):**
    -   **Code Splitting:** Split code into smaller chunks to reduce initial bundle size.
    -   **Tree Shaking:** Use tree shaking to remove unused code from the bundle.
    -   **Minification:** Minify code to reduce bundle size.
-   **Lazy Loading Strategies (for web applications):**
    -   **Lazy Load Images:** Lazy load images that are not initially visible on the page.
    -   **Lazy Load Modules:** Lazy load modules that are not immediately required.
    -   **Intersection Observer:** Use the Intersection Observer API to detect when elements are visible in the viewport.

### 4. Security Best Practices

-   **Common Vulnerabilities and How to Prevent Them:**
    -   **SQL Injection:** Prevent SQL injection by using parameterized queries or object-relational mappers (ORMs).
    -   **Cross-Site Scripting (XSS):** Prevent XSS by sanitizing user inputs and encoding outputs.
    -   **Cross-Site Request Forgery (CSRF):** Prevent CSRF by using anti-CSRF tokens.
    -   **Authentication and Authorization:** Implement strong authentication and authorization mechanisms to protect sensitive data.
-   **Input Validation:**
    -   **Validate All Inputs:** Validate all user inputs on both the client-side and server-side.
    -   **Use Regular Expressions:** Use regular expressions to validate input formats.
    -   **Sanitize Inputs:** Sanitize inputs to remove potentially malicious characters.
-   **Authentication and Authorization Patterns:**
    -   **Identity-Aware Proxy (IAP):** Use IAP to control access to applications running on Compute Engine, GKE, and App Engine.
    -   **Firebase Authentication:** Use Firebase Authentication to easily implement authentication in web and mobile applications.
    -   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization.
    -   **Service Accounts:** Use service accounts to authenticate applications running on GCP.
-   **Data Protection Strategies:**
    -   **Encryption at Rest:** Encrypt data at rest using Cloud KMS or Cloud Storage encryption.
    -   **Encryption in Transit:** Encrypt data in transit using HTTPS (TLS).
    -   **Data Loss Prevention (DLP):** Use Cloud DLP to prevent sensitive data from being leaked.
-   **Secure API Communication:**
    -   **HTTPS:** Always use HTTPS for API communication.
    -   **API Keys:** Use API keys to authenticate API clients.
    -   **JWTs:** Use JWTs for secure API authorization.
    -   **Rate Limiting:** Implement rate limiting to prevent abuse.

### 5. Testing Approaches

-   **Unit Testing Strategies:**
    -   **Test Individual Units:** Test individual units of code (e.g., functions, classes) in isolation.
    -   **Use Mocking:** Use mocking to isolate units of code from dependencies.
    -   **Test Edge Cases:** Test edge cases and boundary conditions.
-   **Integration Testing:**
    -   **Test Interactions:** Test the interactions between different units of code or components.
    -   **Test Database Interactions:** Test interactions with databases and other external systems.
    -   **Use Test Databases:** Use test databases to avoid affecting production data.
-   **End-to-End Testing:**
    -   **Test Complete Workflows:** Test complete workflows from start to finish.
    -   **Simulate User Interactions:** Simulate user interactions to ensure the application behaves as expected.
    -   **Use Automated Testing Tools:** Use automated testing tools to automate end-to-end tests.
-   **Test Organization:**
    -   **Organize Tests by Component:** Organize tests by component or module.
    -   **Use a Consistent Naming Convention:** Use a consistent naming convention for test files and test cases.
    -   **Keep Tests Separate from Code:** Keep tests separate from production code.
-   **Mocking and Stubbing:**
    -   **Use Mocking Libraries:** Use mocking libraries to create mock objects and stubs.
    -   **Mock External Dependencies:** Mock external dependencies like APIs and databases.
    -   **Stub Responses:** Stub responses from external services to control test behavior.

### 6. Common Pitfalls and Gotchas

-   **Frequent Mistakes Developers Make:**
    -   **Incorrect IAM Permissions:** Assigning overly broad IAM permissions.
    -   **Misconfiguring Network Settings:** Misconfiguring firewall rules or VPC settings.
    -   **Ignoring Cost Optimization:** Failing to optimize resource usage and costs.
    -   **Not Using Managed Services:**  Trying to manage infrastructure manually instead of using GCP managed services.
    -   **Lack of Automation:** Not automating deployments and infrastructure management.
-   **Edge Cases to be Aware of:**
    -   **Network Latency:** Account for network latency in distributed systems.
    -   **Concurrency Issues:** Handle concurrency issues correctly in multi-threaded applications.
    -   **Data Consistency:** Ensure data consistency across distributed systems.
-   **Version-Specific Issues:**
    -   **API Compatibility:** Be aware of API compatibility issues when upgrading GCP SDKs or services.
    -   **Deprecated Features:** Avoid using deprecated features.
-   **Compatibility Concerns:**
    -   **Cross-Browser Compatibility:** Ensure web applications are compatible with different browsers.
    -   **Mobile Compatibility:** Ensure web applications are responsive and compatible with mobile devices.
-   **Debugging Strategies:**
    -   **Use Cloud Debugger:** Use Cloud Debugger to debug applications running on GCP.
    -   **Use Logging:** Use Cloud Logging to log debug information.
    -   **Use Monitoring:** Use Cloud Monitoring to monitor application performance and identify issues.
    -   **Local Debugging:** Debug applications locally before deploying them to GCP.

### 7. Tooling and Environment

-   **Recommended Development Tools:**
    -   **Google Cloud SDK (gcloud):** Command-line tool for interacting with GCP services.
    -   **Terraform:** Infrastructure as code tool for managing GCP resources.
    -   **IDE with GCP Support:** Use an IDE with GCP support (e.g., VS Code with the Google Cloud Code extension).
    -   **Docker:** Containerization platform for building and deploying applications.
-   **Build Configuration:**
    -   **Use Build Automation Tools:** Use build automation tools like Maven, Gradle, or npm.
    -   **Define Build Steps:** Define clear build steps for compiling, testing, and packaging applications.
    -   **Manage Dependencies:** Manage dependencies using dependency management tools like Maven, Gradle, or npm.
-   **Linting and Formatting:**
    -   **Use Linters:** Use linters to enforce code style and identify potential issues.
    -   **Use Formatters:** Use formatters to automatically format code.
    -   **Configure Linters and Formatters:** Configure linters and formatters to match project coding standards.
-   **Deployment Best Practices:**
    -   **Automate Deployments:** Automate deployments using CI/CD pipelines.
    -   **Use Infrastructure as Code:** Use infrastructure as code (e.g., Terraform) to manage infrastructure.
    -   **Blue/Green Deployments:** Use blue/green deployments to minimize downtime.
    -   **Canary Deployments:** Use canary deployments to gradually roll out new features.
    -   **Rollback Strategy:** Have a rollback strategy in place to revert to a previous version if necessary.
-   **CI/CD Integration:**
    -   **Use Cloud Build:** Use Cloud Build for CI/CD pipelines on GCP.
    -   **Integrate with Version Control:** Integrate CI/CD pipelines with version control systems (e.g., GitHub, GitLab, Cloud Source Repositories).
    -   **Automate Testing:** Automate testing as part of the CI/CD pipeline.
    -   **Automate Deployments:** Automate deployments as part of the CI/CD pipeline.

By following these best practices, developers can build robust, secure, and scalable applications and infrastructure on Google Cloud Platform.