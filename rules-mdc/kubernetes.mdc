---
description: This rule provides comprehensive best practices for developing and maintaining Kubernetes applications and infrastructure, covering coding standards, security, performance, testing, and deployment.
globs: *.go,*.yaml,*.yml,*.sh,*.tf,*.tfvars,*.json
---
# Kubernetes Development and Operations Best Practices

This document outlines a collection of guidelines, style suggestions, and tips for writing code and managing infrastructure within the Kubernetes ecosystem. It emphasizes clarity, maintainability, security, and performance.

## 1. Code Organization and Structure

### 1.1 Directory Structure

- **Root Level:**
  - `cmd/`:  Main application entry points. Each subdirectory represents a separate command-line tool or service.
  - `pkg/`: Reusable libraries and components that can be imported by other projects.
  - `internal/`: Private code that should not be imported by external projects.  Enforces encapsulation.
  - `api/`:  API definitions, including protobuf files and OpenAPI specifications.
  - `config/`: Configuration files, such as YAML manifests, Kustomize configurations, and Helm charts.
  - `scripts/`: Utility scripts for building, testing, and deploying the application.
  - `docs/`: Documentation for the project.
  - `examples/`: Example usage of the library or application.
  - `vendor/`: (If using `go modules` without external dependency management) Contains vendored dependencies.  Generally discouraged in modern Go with `go modules`.
- **Component-Specific Directories:** Inside `pkg/` or `internal/`, organize code by component or module. Each component should have its own directory with clear separation of concerns.

Example:


my-kubernetes-project/
├── cmd/
│   └── controller/
│       └── main.go
├── pkg/
│   └── api/
│       ├── types.go
│   └── controller/
│       ├── controller.go
│       ├── reconciler.go
│   └── util/
│       └── util.go
├── internal/
│   └── admission/
│       └── webhook.go
├── config/
│   ├── deploy/
│   │   └── deployment.yaml
│   └── kustomize/
│       ├── base/
│       │   ├── kustomization.yaml
│       │   └── ...
│       └── overlays/
│           ├── dev/
│           │   ├── kustomization.yaml
│           │   └── ...
│           └── prod/
│               ├── kustomization.yaml
│               └── ...
├── scripts/
│   └── build.sh
├── docs/
│   └── architecture.md
└── go.mod


### 1.2 File Naming Conventions

- **Go Files:** Use lowercase with underscores (e.g., `my_controller.go`).
- **YAML Files:** Use lowercase with dashes (e.g., `deployment.yaml`).
- **Configuration Files:** Be descriptive and consistent (e.g., `config.yaml`, `kustomization.yaml`).
- **Test Files:** Follow the standard Go convention: `*_test.go` (e.g., `my_controller_test.go`).

### 1.3 Module Organization (Go)

- **Packages:**  Organize code into meaningful packages that represent logical units of functionality.
- **Internal Packages:** Use `internal/` directories to create packages that are only visible within the project.
- **Interfaces:** Define interfaces to abstract dependencies and promote testability.

### 1.4 Component Architecture

- **Microservices:** Design applications as a collection of loosely coupled microservices.
- **Separation of Concerns:** Each component should have a single responsibility and well-defined interfaces.
- **API Gateway:** Use an API gateway to handle routing, authentication, and rate limiting for external requests.
- **Service Mesh:** Consider using a service mesh (e.g., Istio, Linkerd) to manage inter-service communication, observability, and security.

### 1.5 Code Splitting Strategies

- **Feature-Based Splitting:** Group code by feature or functionality.
- **Layer-Based Splitting:** Separate code into layers, such as data access, business logic, and presentation.
- **Component-Based Splitting:** Divide code into reusable components that can be shared across multiple projects.

## 2. Common Patterns and Anti-Patterns

### 2.1 Design Patterns

- **Controller Pattern:** Implement controllers to reconcile the desired state of Kubernetes resources with the actual state.
- **Operator Pattern:** Extend the Kubernetes API with custom resources and controllers to automate complex application management tasks.
- **Sidecar Pattern:** Deploy a sidecar container alongside the main application container to provide supporting functionality, such as logging, monitoring, or security.
- **Ambassador Pattern:**  Use an ambassador container to proxy network traffic to the main application container, providing features such as load balancing, routing, and authentication.
- **Adapter Pattern:** Translate requests from one interface to another, allowing different components to work together.
- **Singleton Pattern:** Implement a singleton pattern for managing global resources, such as database connections or configuration settings. Be extremely cautious, as this can hurt testability and introduce implicit dependencies.

### 2.2 Recommended Approaches for Common Tasks

- **Resource Management:** Use Kubernetes resource requests and limits to ensure that applications have sufficient resources and prevent resource contention.
- **Configuration Management:** Use ConfigMaps and Secrets to manage configuration data and sensitive information separately from the application code.
- **Service Discovery:** Use Kubernetes services to provide a stable endpoint for accessing applications, even when pods are scaled up or down.
- **Health Checks:** Implement liveness and readiness probes to monitor the health of applications and automatically restart unhealthy pods.
- **Logging and Monitoring:** Use a centralized logging and monitoring system to collect and analyze application logs and metrics.
- **Rolling Updates:** Use Kubernetes deployments to perform rolling updates of applications with zero downtime.

### 2.3 Anti-Patterns and Code Smells

- **Naked Pods:** Avoid creating pods directly without a deployment or replica set, as they will not be automatically rescheduled if a node fails.
- **Hardcoded Configuration:** Avoid hardcoding configuration data in the application code. Use ConfigMaps and Secrets instead.
- **Ignoring Resource Limits:** Failing to set resource requests and limits can lead to resource contention and performance issues.
- **Oversized Containers:**  Keep container images small and focused to improve startup time and reduce security risks.
- **Privileged Containers:** Avoid running containers in privileged mode, as it can create security vulnerabilities.
- **Long-Lived Branches:** Avoid creating long-lived branches, and prefer small, frequent merges to the main branch.
- **God Classes:** Avoid creating classes that are too large and complex.  Break them down into smaller, more manageable classes.
- **Shotgun Surgery:**  Avoid making changes to multiple classes when a single feature is modified. This suggests poor class design and coupling.
- **Feature Envy:**  Avoid methods that access the data of another object more than their own.  This suggests that the method might be in the wrong class.

### 2.4 State Management Best Practices

- **Stateless Applications:** Prefer stateless applications whenever possible, as they are easier to scale and manage.
- **Persistent Volumes:** Use Persistent Volumes to store persistent data for stateful applications.
- **External Databases:** Consider using external databases for managing application state, such as databases hosted on cloud providers.
- **Kubernetes Operators:** Implement Kubernetes operators to automate the management of stateful applications.
- **Etcd:** Understand the importance of etcd as Kubernetes' data store and protect it accordingly.

### 2.5 Error Handling Patterns

- **Centralized Error Handling:** Implement a centralized error handling mechanism to handle exceptions and log errors consistently.
- **Retry Mechanism:** Implement a retry mechanism to automatically retry failed operations.
- **Circuit Breaker Pattern:** Use a circuit breaker pattern to prevent cascading failures in distributed systems.
- **Logging Error Details:** Log detailed error messages, including stack traces and relevant context, to help with debugging.
- **Graceful Degradation:** Design applications to gracefully degrade functionality when errors occur.
- **Alerting on Critical Errors:** Set up alerts to notify administrators when critical errors occur.

## 3. Performance Considerations

### 3.1 Optimization Techniques

- **Caching:** Implement caching to reduce latency and improve performance.
- **Load Balancing:** Use load balancing to distribute traffic across multiple instances of an application.
- **Connection Pooling:** Use connection pooling to reuse database connections and reduce overhead.
- **Compression:** Use compression to reduce the size of data transmitted over the network.
- **Gzip:** Enable Gzip compression in web servers to reduce the size of HTTP responses.

### 3.2 Memory Management

- **Memory Profiling:** Use memory profiling tools to identify memory leaks and optimize memory usage.
- **Garbage Collection:** Understand how garbage collection works in the programming language used for the application.
- **Resource Limits:** Set memory resource limits for containers to prevent them from consuming excessive memory.
- **Monitor Memory Usage:** Monitor memory usage regularly to identify potential issues.

### 3.3 Rendering Optimization

- **Minimize DOM Manipulation:** Reduce the number of DOM manipulations to improve rendering performance in web applications.
- **Virtual DOM:** Use a virtual DOM to optimize rendering updates in web applications.
- **Lazy Loading:** Use lazy loading to load images and other resources only when they are needed.

### 3.4 Bundle Size Optimization

- **Code Minification:** Use code minification to reduce the size of JavaScript and CSS files.
- **Tree Shaking:** Use tree shaking to remove unused code from JavaScript bundles.
- **Image Optimization:** Optimize images to reduce their file size without sacrificing quality.
- **Code Splitting:** Split the application code into smaller bundles that can be loaded on demand.

### 3.5 Lazy Loading Strategies

- **On-Demand Loading:** Load resources only when they are needed by the application.
- **Intersection Observer:** Use the Intersection Observer API to detect when elements are visible in the viewport and load them accordingly.
- **Placeholder Images:** Use placeholder images while loading the actual images to improve the user experience.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

- **Injection Attacks:** Prevent injection attacks by validating and sanitizing all user input.
- **Cross-Site Scripting (XSS):** Prevent XSS attacks by escaping all user-generated content before rendering it in the browser.
- **Cross-Site Request Forgery (CSRF):** Prevent CSRF attacks by using anti-CSRF tokens.
- **Authentication and Authorization Flaws:** Implement robust authentication and authorization mechanisms to protect sensitive data and resources.
- **Security Misconfiguration:** Avoid using default configurations and ensure that all components are properly configured with security in mind.
- **Using Components with Known Vulnerabilities:** Keep all dependencies up to date to patch known vulnerabilities.
- **Insufficient Logging and Monitoring:** Implement comprehensive logging and monitoring to detect and respond to security incidents.
- **Container Security:** Follow best practices for securing containers, such as using minimal images, running as non-root, and limiting capabilities.
- **Network Policies:** Use network policies to restrict network traffic between pods.
- **RBAC (Role-Based Access Control):** Implement RBAC to control access to Kubernetes resources.
- **Secrets Management:** Use Kubernetes Secrets to store sensitive information, and encrypt secrets at rest.
- **Pod Security Policies/Pod Security Standards:** Enforce Pod Security Standards to restrict the capabilities of pods.

### 4.2 Input Validation

- **Validate All Input:** Validate all input, including user input, API requests, and configuration data.
- **Use Strong Data Types:** Use strong data types to enforce data integrity.
- **Sanitize Input:** Sanitize input to remove potentially harmful characters and prevent injection attacks.
- **Whitelist Input:** Use a whitelist approach to only allow known good input.
- **Blacklist Input:** Avoid using a blacklist approach, as it can be easily bypassed.

### 4.3 Authentication and Authorization Patterns

- **Multi-Factor Authentication (MFA):** Use MFA to enhance authentication security.
- **OAuth 2.0:** Use OAuth 2.0 for authorization and delegation of access.
- **JSON Web Tokens (JWT):** Use JWTs for securely transmitting claims between parties.
- **Role-Based Access Control (RBAC):** Implement RBAC to control access to resources based on roles.
- **Least Privilege Principle:** Grant users and applications only the minimum necessary permissions.

### 4.4 Data Protection Strategies

- **Encryption at Rest:** Encrypt sensitive data at rest to protect it from unauthorized access.
- **Encryption in Transit:** Encrypt sensitive data in transit using HTTPS or other secure protocols.
- **Data Masking:** Mask sensitive data to prevent it from being exposed to unauthorized users.
- **Data Anonymization:** Anonymize data to remove personally identifiable information (PII).
- **Data Loss Prevention (DLP):** Implement DLP measures to prevent sensitive data from leaving the organization.

### 4.5 Secure API Communication

- **HTTPS:** Use HTTPS for all API communication to encrypt data in transit.
- **API Authentication:** Implement API authentication to verify the identity of clients.
- **API Authorization:** Implement API authorization to control access to API endpoints.
- **Rate Limiting:** Implement rate limiting to prevent abuse and denial-of-service attacks.
- **Input Validation:** Validate all API requests to prevent injection attacks and other vulnerabilities.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

- **Test-Driven Development (TDD):** Write unit tests before writing the application code.
- **Mock Dependencies:** Use mocks to isolate the unit being tested from its dependencies.
- **Test Boundary Conditions:** Test boundary conditions and edge cases to ensure that the code handles them correctly.
- **Test Error Conditions:** Test error conditions to ensure that the code handles errors gracefully.
- **Code Coverage:** Aim for high code coverage to ensure that all parts of the code are tested.
- **Table-Driven Tests:**  Use table-driven tests to easily test multiple inputs and outputs.

### 5.2 Integration Testing

- **Test Interactions Between Components:** Test the interactions between different components of the application.
- **Test with Real Dependencies:** Use real dependencies or integration mocks for integration tests.
- **Test Data Flows:** Test the data flows through the application to ensure that data is processed correctly.
- **Contract Tests:**  Use contract tests to ensure that services adhere to a defined contract.

### 5.3 End-to-End Testing

- **Test the Entire System:** Test the entire system from end to end to ensure that all components work together correctly.
- **Automate End-to-End Tests:** Automate end-to-end tests to ensure that they are run regularly.
- **Use Realistic Test Data:** Use realistic test data to simulate real-world scenarios.
- **CI/CD Integration:** Integrate end-to-end tests into the CI/CD pipeline.

### 5.4 Test Organization

- **Keep Tests Separate from Code:** Keep tests separate from the application code in a dedicated `test/` directory.
- **Organize Tests by Component:** Organize tests by component or module to make them easier to find and maintain.
- **Use Clear Naming Conventions:** Use clear naming conventions for test files and test functions.

### 5.5 Mocking and Stubbing

- **Use Mocking Frameworks:** Use mocking frameworks to simplify the creation of mocks and stubs.
- **Mock External Dependencies:** Mock external dependencies, such as databases and APIs, to isolate the unit being tested.
- **Stub Responses:** Use stubs to provide predefined responses for external dependencies.
- **Verify Interactions:** Verify that the code under test interacts with dependencies as expected.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

- **Ignoring Error Handling:** Failing to handle errors properly can lead to unexpected behavior and crashes.
- **Not Using Version Control:** Not using version control can lead to lost code and conflicts.
- **Hardcoding Configuration:** Hardcoding configuration data can make it difficult to deploy the application in different environments.
- **Not Securing Sensitive Data:** Not securing sensitive data can lead to security breaches.
- **Not Testing Thoroughly:** Not testing thoroughly can lead to bugs and performance issues.
- **Over-Engineering:** Adding unnecessary complexity to the code can make it difficult to understand and maintain.
- **Premature Optimization:** Optimizing code before it is necessary can waste time and make the code harder to read.
- **Not Documenting Code:** Not documenting code can make it difficult for others to understand and maintain it.

### 6.2 Edge Cases to Be Aware Of

- **Network Connectivity Issues:** Handle network connectivity issues gracefully.
- **Resource Exhaustion:** Handle resource exhaustion gracefully.
- **Concurrency Issues:** Avoid concurrency issues by using proper synchronization mechanisms.
- **Data Corruption:** Protect against data corruption by using checksums and other data integrity techniques.
- **Time Zone Issues:** Be aware of time zone issues when working with dates and times.

### 6.3 Version-Specific Issues

- **API Version Compatibility:** Be aware of API version compatibility issues when upgrading Kubernetes or other dependencies.
- **Feature Deprecation:** Be aware of feature deprecation when upgrading Kubernetes or other dependencies.
- **Configuration Changes:** Be aware of configuration changes when upgrading Kubernetes or other dependencies.

### 6.4 Compatibility Concerns

- **Operating System Compatibility:** Ensure that the application is compatible with the target operating systems.
- **Architecture Compatibility:** Ensure that the application is compatible with the target architectures (e.g., x86, ARM).
- **Browser Compatibility:** Ensure that web applications are compatible with the target browsers.

### 6.5 Debugging Strategies

- **Logging:** Use detailed logging to help identify the root cause of issues.
- **Debugging Tools:** Use debugging tools, such as debuggers and profilers, to analyze the code and identify performance bottlenecks.
- **Remote Debugging:** Use remote debugging to debug applications running in Kubernetes.
- **Log Aggregation:**  Use log aggregation tools (e.g., Elasticsearch, Loki) to centralize and analyze logs.
- **Metrics Monitoring:** Use metrics monitoring tools (e.g., Prometheus, Grafana) to track application performance.
- **Tracing:** Implement distributed tracing (e.g., Jaeger, Zipkin) to track requests across multiple services.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

- **IDE:** Use a modern IDE with support for the programming language used for the application (e.g., VS Code, IntelliJ IDEA, GoLand).
- **Kubectl:** Use `kubectl` for interacting with Kubernetes clusters.
- **Minikube/Kind:** Use Minikube or Kind for local Kubernetes development.
- **Helm:** Use Helm for managing Kubernetes packages.
- **Kustomize:** Use Kustomize for customizing Kubernetes configurations.
- **Docker:** Use Docker for building and managing container images.
- **Tilt:** Use Tilt for fast, local Kubernetes development.
- **Skaffold:** Use Skaffold for automated build, push, and deploy workflows.
- **Telepresence:** Use Telepresence to debug applications running in Kubernetes from your local machine.

### 7.2 Build Configuration

- **Makefile:** Use a Makefile to automate common build tasks.
- **CI/CD Pipeline:** Integrate the build process into a CI/CD pipeline.
- **Dependency Management:** Use a dependency management tool, such as `go modules`, to manage dependencies.
- **Version Control:** Use version control to track changes to the build configuration.

### 7.3 Linting and Formatting

- **Linters:** Use linters to enforce code style and best practices (e.g., `golangci-lint`, `eslint`, `stylelint`).
- **Formatters:** Use formatters to automatically format code according to a predefined style (e.g., `go fmt`, `prettier`).
- **Pre-Commit Hooks:** Use pre-commit hooks to run linters and formatters before committing code.

### 7.4 Deployment Best Practices

- **Infrastructure as Code (IaC):** Use IaC tools, such as Terraform or CloudFormation, to manage infrastructure.
- **Immutable Infrastructure:** Deploy immutable infrastructure to ensure consistency and repeatability.
- **Blue-Green Deployments:** Use blue-green deployments to minimize downtime during deployments.
- **Canary Deployments:** Use canary deployments to test new versions of the application with a small subset of users.
- **Rolling Updates:** Use rolling updates to gradually update the application with zero downtime.
- **Automated Rollbacks:** Implement automated rollbacks to quickly revert to a previous version of the application if something goes wrong.

### 7.5 CI/CD Integration

- **Automated Testing:** Automate all tests in the CI/CD pipeline.
- **Automated Deployment:** Automate the deployment process in the CI/CD pipeline.
- **Continuous Integration:** Use continuous integration to automatically build and test the application whenever code is committed.
- **Continuous Delivery:** Use continuous delivery to automatically deploy the application to production whenever a new version is released.
- **Pipeline Security:** Secure the CI/CD pipeline to prevent unauthorized access and code injection.

## Bibliography

- Kubernetes documentation: [https://kubernetes.io/docs/](https://kubernetes.io/docs/)
- Kubernetes Best Practices: [https://kubernetes.io/docs/concepts/configuration/overview/](https://kubernetes.io/docs/concepts/configuration/overview/)
- Application Security Checklist: [https://kubernetes.io/docs/concepts/security/application-security-checklist/](https://kubernetes.io/docs/concepts/security/application-security-checklist/)
- Kubernetes coding conventions: [https://www.kubernetes.dev/docs/guide/coding-convention/](https://www.kubernetes.dev/docs/guide/coding-convention/)