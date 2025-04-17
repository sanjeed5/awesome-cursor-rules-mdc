---
description: This rule provides comprehensive guidelines for DigitalOcean infrastructure and application development, covering code organization, security, performance, and deployment best practices.  It aims to ensure consistent, scalable, and secure cloud solutions on the DigitalOcean platform.
globs: *.tf,*.yml,*.yaml,*.sh,*.py
---
# DigitalOcean Development Best Practices

This document provides comprehensive guidelines for developing infrastructure and applications on DigitalOcean. It covers various aspects, including code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and recommended tooling.

## 1. Code Organization and Structure

### Directory Structure

A well-defined directory structure is crucial for maintainability and scalability. Consider these patterns:


project-root/
├── modules/                 # Reusable Terraform modules
│   ├── droplet/           # Module for creating Droplets
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── ...
│   ├── database/          # Module for creating Managed Databases
│   │   └── ...
│   └── ...
├── environments/            # Environment-specific configurations
│   ├── dev/                # Development environment
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── terraform.tfvars  # Environment-specific variables
│   │   └── ...
│   ├── staging/
│   │   └── ...
│   ├── prod/
│   │   └── ...
│   └── ...
├── scripts/                # Utility scripts (e.g., deployment, backup)
│   ├── deploy.sh
│   ├── backup.sh
│   └── ...
├── tests/                   # Infrastructure tests
│   ├── integration/
│   │   └── ...
│   ├── unit/
│   │   └── ...
│   └── ...
├── README.md
├── LICENSE
└── ...


*   **`modules/`**: Contains reusable Terraform modules.  Each module encapsulates a specific DigitalOcean resource or set of resources (e.g., Droplet, Load Balancer, Database). This promotes code reuse and reduces duplication.
*   **`environments/`**: Defines configurations specific to each environment (development, staging, production).  This allows for environment-specific settings without modifying the core module code.  Use `terraform.tfvars` files to store environment-specific variable values.
*   **`scripts/`**: Includes utility scripts for tasks such as deployment, backups, and monitoring.  Use descriptive names for scripts and include comments explaining their purpose.
*   **`tests/`**: Contains automated tests to verify the infrastructure.  Separate integration tests (testing the interaction between resources) from unit tests (testing individual modules).

### File Naming Conventions

*   **Terraform files:**  Use `.tf` extension.  Name files descriptively (e.g., `main.tf`, `variables.tf`, `outputs.tf`).  For environment-specific variables, use `terraform.tfvars`.
*   **YAML/YML files:** Use `.yml` or `.yaml` extensions for configuration files (e.g., Docker Compose, Kubernetes manifests). Choose one consistently within the project.
*   **Shell scripts:** Use `.sh` extension.  Name scripts according to their function (e.g., `deploy.sh`, `backup.sh`).
*   **Python scripts:** Use `.py` extension. Follow PEP 8 guidelines for naming variables, functions, and classes.

### Module Organization

*   **Single Responsibility Principle:**  Each module should have a single, well-defined purpose (e.g., creating a Droplet, configuring a firewall).
*   **Abstraction:**  Modules should abstract away the complexity of underlying DigitalOcean resources, providing a simple and consistent interface for users.
*   **Versioning:**  Use Terraform module registry to version and share reusable modules. This allows you to track changes and ensure consistent deployments across environments.
*   **Input Validation:**  Validate input variables to ensure they meet expected criteria (e.g., data type, allowed values).  This helps prevent errors and improves the reliability of your infrastructure.

### Component Architecture

*   **Microservices:**  For complex applications, consider a microservices architecture.  Each microservice can be deployed as a separate Droplet or containerized application within a DigitalOcean Kubernetes cluster.
*   **Stateless Applications:** Design applications to be stateless whenever possible. This simplifies scaling and improves resilience. Store persistent data in Managed Databases or Object Storage.
*   **API Gateway:** Use an API gateway to manage external access to your microservices.  This provides a single point of entry and allows you to implement security policies, rate limiting, and other features.

### Code Splitting

*   **Terraform workspaces:** Use Terraform workspaces to manage different environments within the same Terraform configuration. This avoids code duplication and simplifies environment management.
*   **Modular design:** Follow a modular design approach for your application code. Break down complex functionalities into smaller, manageable modules.
*   **Lazy loading:** Implement lazy loading for non-critical components to improve initial load time.  This can be particularly useful for web applications.

## 2. Common Patterns and Anti-patterns

### Design Patterns

*   **Infrastructure as Code (IaC):**  Use Terraform, Pulumi, or other IaC tools to manage your DigitalOcean infrastructure programmatically. This ensures consistency, repeatability, and version control.
*   **Twelve-Factor App:**  Follow the principles of the Twelve-Factor App methodology for building cloud-native applications. This promotes portability, scalability, and maintainability.
*   **Containerization:**  Use Docker to containerize your applications. This provides a consistent runtime environment and simplifies deployment to DigitalOcean Kubernetes or Droplets.
*   **Service Discovery:**  Implement service discovery to allow your microservices to locate each other dynamically.  Tools like Consul or etcd can be used for this purpose.

### Recommended Approaches

*   **Automated Backups:** Regularly back up your Managed Databases and Droplet volumes.  Use DigitalOcean's backup features or implement a custom backup solution using scripts and Object Storage.
*   **Monitoring:**  Set up monitoring for your DigitalOcean resources using DigitalOcean Monitoring or third-party tools like Prometheus and Grafana.  This allows you to track performance, identify issues, and receive alerts.
*   **Load Balancing:** Use DigitalOcean Load Balancers to distribute traffic across multiple Droplets. This improves performance and availability.
*   **CDN:** Use DigitalOcean CDN to cache static assets and improve website performance for users around the world.

### Anti-patterns

*   **Manual Infrastructure Management:**  Avoid manually creating or modifying DigitalOcean resources through the web interface.  This can lead to inconsistencies and makes it difficult to track changes.
*   **Hardcoding Credentials:** Never hardcode API keys, passwords, or other sensitive information in your code. Store them in environment variables or use a secrets management tool like HashiCorp Vault.
*   **Ignoring Security Updates:**  Keep your Droplet operating systems and application dependencies up-to-date with the latest security patches.  Use automated tools like `unattended-upgrades` to apply security updates automatically.
*   **Lack of Monitoring:**  Failing to monitor your DigitalOcean resources can lead to undetected performance issues and outages.

### State Management

*   **Terraform State:** Store Terraform state remotely in DigitalOcean Spaces or other supported backends.  This ensures that your Terraform state is stored securely and can be accessed by multiple team members.
*   **Application State:**  For stateful applications, use Managed Databases or Object Storage to store persistent data. Avoid storing state directly on Droplets.

### Error Handling

*   **Comprehensive Logging:** Implement comprehensive logging throughout your application.  Include enough information to diagnose issues but avoid logging sensitive data.
*   **Centralized Logging:** Use a centralized logging system to collect and analyze logs from all your DigitalOcean resources.  Tools like ELK stack (Elasticsearch, Logstash, Kibana) can be used for this purpose.
*   **Alerting:** Set up alerting to notify you of errors, performance issues, and other critical events.
*   **Retry Logic:** Implement retry logic for transient errors, such as network timeouts. This can improve the resilience of your application.

## 3. Performance Considerations

### Optimization Techniques

*   **Resource Selection:** Choose the appropriate Droplet size and Managed Database tier for your workload.  Start with a smaller instance and scale up as needed.
*   **Caching:** Implement caching at various levels, including CDN, HTTP caching, and application-level caching. Use Redis or Memcached for in-memory caching.
*   **Database Optimization:** Optimize your database queries and schema. Use indexes to improve query performance.  Consider using a connection pool to reduce database connection overhead.
*   **Code Profiling:** Profile your application code to identify performance bottlenecks.  Use profiling tools to identify slow-running functions and optimize them.

### Memory Management

*   **Memory Leaks:**  Be aware of memory leaks in your application code.  Use memory profiling tools to identify and fix memory leaks.
*   **Garbage Collection:** Understand how garbage collection works in your programming language. Tune garbage collection settings to optimize performance.

### Rendering Optimization

*   **Frontend Optimization:**  Optimize your frontend code by minimizing HTTP requests, compressing images, and using a CDN.
*   **Lazy Loading:** Implement lazy loading for images and other non-critical resources.

### Bundle Size Optimization

*   **Code Minification:**  Minify your JavaScript and CSS code to reduce bundle size.
*   **Tree Shaking:** Use tree shaking to remove unused code from your JavaScript bundles.
*   **Code Splitting:**  Split your code into smaller bundles that can be loaded on demand.

### Lazy Loading

*   **Dynamic Imports:** Use dynamic imports to load modules on demand. This can significantly reduce initial load time.
*   **Intersection Observer:**  Use the Intersection Observer API to lazy load images and other resources when they become visible in the viewport.

## 4. Security Best Practices

### Common Vulnerabilities

*   **SQL Injection:**  Prevent SQL injection by using parameterized queries or an ORM.
*   **Cross-Site Scripting (XSS):**  Prevent XSS by escaping user input and using a Content Security Policy (CSP).
*   **Cross-Site Request Forgery (CSRF):**  Prevent CSRF by using anti-CSRF tokens.
*   **Authentication and Authorization:**  Implement strong authentication and authorization mechanisms to protect your application.

### Input Validation

*   **Whitelisting:** Use whitelisting to validate user input. Only allow specific characters, data types, and formats.
*   **Escaping:** Escape user input before displaying it to prevent XSS attacks.
*   **Sanitization:** Sanitize user input to remove potentially harmful characters or code.

### Authentication and Authorization

*   **Strong Passwords:** Enforce strong password policies and use a secure password hashing algorithm like bcrypt or Argon2.
*   **Multi-Factor Authentication (MFA):** Implement MFA for all users.
*   **Role-Based Access Control (RBAC):** Use RBAC to control access to resources based on user roles.
*   **Least Privilege:**  Grant users only the minimum necessary privileges.

### Data Protection

*   **Encryption at Rest:**  Encrypt sensitive data at rest using DigitalOcean's encryption features or a third-party encryption solution.
*   **Encryption in Transit:**  Use HTTPS to encrypt data in transit between your application and users.
*   **Data Masking:**  Mask sensitive data when displaying it to users.
*   **Data Retention Policies:**  Implement data retention policies to ensure that data is only stored for as long as necessary.

### Secure API Communication

*   **API Keys:**  Use API keys to authenticate requests to your APIs.
*   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your APIs.
*   **API Versioning:** Use API versioning to allow you to make changes to your APIs without breaking existing clients.

## 5. Testing Approaches

### Unit Testing

*   **Test-Driven Development (TDD):**  Consider using TDD to write unit tests before writing your application code.
*   **Mocking:** Use mocking to isolate components and test them in isolation.
*   **Test Coverage:** Aim for high test coverage to ensure that your code is thoroughly tested.

### Integration Testing

*   **Real DigitalOcean Resources:**  Use real DigitalOcean resources for integration tests. This ensures that your code works correctly with the DigitalOcean platform.
*   **Automated Setup and Teardown:**  Automate the setup and teardown of your test environment.

### End-to-End Testing

*   **Browser Automation:** Use browser automation tools like Selenium or Cypress to test your application from end to end.
*   **Realistic Scenarios:** Test realistic user scenarios to ensure that your application works as expected.

### Test Organization

*   **Separate Test Directory:**  Create a separate directory for your tests.
*   **Descriptive Test Names:**  Use descriptive names for your test files and test functions.

### Mocking and Stubbing

*   **Mock DigitalOcean API:** Mock the DigitalOcean API to test your code without making actual API calls. This is particularly useful for unit tests.
*   **Stub External Dependencies:** Stub external dependencies to isolate your code and make tests more reliable.

## 6. Common Pitfalls and Gotchas

### Frequent Mistakes

*   **Not Using IaC:**  Manually managing infrastructure is error-prone and time-consuming. Use Terraform or other IaC tools.
*   **Hardcoding Secrets:**  Storing secrets in code is a major security risk. Use environment variables or a secrets management tool.
*   **Ignoring Security Updates:**  Failing to apply security updates can leave your systems vulnerable to attack.
*   **Lack of Monitoring:**  Not monitoring your systems can lead to undetected problems.

### Edge Cases

*   **API Rate Limits:** Be aware of DigitalOcean API rate limits and implement retry logic to handle rate limit errors.
*   **Resource Limits:** Be aware of DigitalOcean resource limits (e.g., number of Droplets, volumes, databases) and plan accordingly.

### Version-Specific Issues

*   **Terraform Provider Versions:**  Use specific versions of the DigitalOcean Terraform provider to avoid compatibility issues.
*   **API Versioning:** Be aware of changes in the DigitalOcean API and update your code accordingly.

### Compatibility Concerns

*   **Operating System Compatibility:**  Ensure that your application is compatible with the operating system running on your Droplets.
*   **Database Compatibility:** Ensure that your application is compatible with the version of Managed Databases you are using.

### Debugging Strategies

*   **Logging:**  Use comprehensive logging to diagnose issues.
*   **Remote Debugging:** Use remote debugging to debug your application code running on a Droplet.
*   **Terraform Debugging:** Use Terraform's debugging features to troubleshoot Terraform configurations.

## 7. Tooling and Environment

### Recommended Tools

*   **Terraform:** For Infrastructure as Code.
*   **Pulumi:** Alternative Infrastructure as Code using familiar programming languages.
*   **Docker:** For containerization.
*   **DigitalOcean CLI (doctl):** For managing DigitalOcean resources from the command line.
*   **Visual Studio Code (VS Code):** Code editor with extensions for Terraform, Docker, and other technologies.
*   **Git:** Version control system.
*   **GitHub/GitLab/Bitbucket:** Code hosting and collaboration platforms.

### Build Configuration

*   **Terraform Modules:**  Structure your Terraform code into reusable modules.
*   **Terraform Workspaces:** Use Terraform workspaces to manage different environments.

### Linting and Formatting

*   **Terraform fmt:** Use `terraform fmt` to format your Terraform code.
*   **Shellcheck:** Use Shellcheck to lint your shell scripts.
*   **Pylint:** Use Pylint to lint your Python code.
*   **Prettier:** Use Prettier to format your JavaScript and CSS code.

### Deployment Best Practices

*   **CI/CD Pipelines:** Use CI/CD pipelines to automate the deployment process.
*   **Blue/Green Deployments:** Use blue/green deployments to minimize downtime during deployments.
*   **Canary Deployments:** Use canary deployments to test new versions of your application with a small subset of users before rolling them out to everyone.

### CI/CD Integration

*   **GitHub Actions:** Use GitHub Actions to automate your CI/CD pipelines.
*   **GitLab CI/CD:** Use GitLab CI/CD to automate your CI/CD pipelines.
*   **Jenkins:** Use Jenkins to automate your CI/CD pipelines.

By following these best practices, you can build scalable, secure, and reliable applications on the DigitalOcean platform.