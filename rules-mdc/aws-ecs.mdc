---
description: This rule covers best practices for developing, deploying, and maintaining applications using AWS Elastic Container Service (ECS). It includes guidance on code organization, performance, security, testing, and common pitfalls.
globs: *.tf,*.yml,*.yaml,*.json,*.sh,*.dockerfile
---
- **General Principles:**
  - **Infrastructure as Code (IaC):** Treat your ECS infrastructure as code. Use tools like Terraform, AWS CloudFormation, or AWS CDK to define and manage your ECS clusters, task definitions, and services.
  - **Configuration Management:** Externalize configuration from your application code. Use environment variables, AWS Secrets Manager, or AWS Systems Manager Parameter Store to manage configuration data.
  - **Idempotency:** Ensure that your deployment scripts and automation are idempotent. This means that running the same script multiple times should have the same result as running it once.
  - **Automation:** Automate all aspects of your ECS deployment, from building container images to deploying services and scaling resources.

- **1. Code Organization and Structure:**
  - **1.1 Directory Structure:**
    - **Project Root:** The base directory containing all project-related files.
    - `infrastructure/`: Contains IaC code (Terraform, CloudFormation) for ECS clusters, task definitions, services, VPC, etc.
    - `application/`: Contains source code for your containerized application.
      - `src/`: Application source code.
      - `tests/`: Unit, integration, and end-to-end tests.
      - `Dockerfile`: Dockerfile for building the application container image.
    - `scripts/`: Contains deployment, build, and other utility scripts.
    - `docs/`: Documentation for the project.
    - `README.md`: Project README file.

  - **1.2 File Naming Conventions:**
    - **Terraform:** `main.tf`, `variables.tf`, `outputs.tf`, `<module_name>.tf`
    - **CloudFormation:** `<stack_name>.yml` or `<stack_name>.yaml`
    - **Docker:** `Dockerfile` (no extension), `.dockerignore`
    - **Scripts:** `<script_name>.sh` (Bash), `<script_name>.py` (Python)
    - **Configuration:** `config.yml`, `config.json`, `.env`

  - **1.3 Module Organization:**
    - **Terraform Modules:** Create reusable Terraform modules for common ECS infrastructure components (e.g., ECS cluster, task definition, load balancer).
    - **Application Modules:** Organize your application code into logical modules based on functionality (e.g., authentication, API, database access).
    -  Use separate git repositories for independent services/applications, and ECS services accordingly. This avoids monolithic deployments and allows independent scaling and versioning.

  - **1.4 Component Architecture:**
    - **Microservices:** Design your application as a collection of microservices, each deployed as an independent ECS service.
    - **API Gateway:** Use an API gateway (e.g., Amazon API Gateway) to route requests to different ECS services.
    - **Message Queue:** Use a message queue (e.g., Amazon SQS, Amazon MQ) for asynchronous communication between services.
    - **Database:** Use a managed database service (e.g., Amazon RDS, Amazon DynamoDB) for data persistence. Ensure proper security and access control configurations.

  - **1.5 Code Splitting:**
    - Break down large applications or services into smaller, more manageable components that can be deployed independently.
    - Optimize container images to only include code required for each specific component.
    - Utilize ECS services to deploy each component separately, enabling independent scaling and upgrades.

- **2. Common Patterns and Anti-patterns:**
  - **2.1 Design Patterns:**
    - **Sidecar:** Use sidecar containers (e.g., for logging, monitoring, or service mesh) to add functionality to your main application container without modifying the application code. Use a separate container in the same task definition.
    - **Backend for Frontend (BFF):** Create a BFF layer that provides a specific API for each client application, improving performance and security.
    - **Strangler Fig:** Gradually migrate a legacy application to ECS by introducing new microservices and slowly replacing the old functionality.
    - **Aggregator Pattern:** Use an aggregator service to combine data from multiple backend services into a single response.

  - **2.2 Recommended Approaches:**
    - **Health Checks:** Implement robust health checks for your containers and configure ECS to use them.  Use both `HEALTHCHECK` in the `Dockerfile` for initial container health and ECS health checks (ELB health checks) for service availability.
    - **Service Discovery:** Use service discovery (e.g., AWS Cloud Map) to enable services to find each other dynamically.
    - **Load Balancing:** Use a load balancer (e.g., Application Load Balancer, Network Load Balancer) to distribute traffic across multiple ECS tasks.
    - **Auto Scaling:** Configure auto scaling to automatically adjust the number of ECS tasks based on demand.  Use CloudWatch metrics (CPU, memory, request count) as scaling triggers.
    - **Immutable Infrastructure:** Avoid making changes to running containers. Instead, redeploy a new container image with the changes.
    - **Use ECS Exec for debugging** avoid SSH into running EC2 instances

  - **2.3 Anti-patterns:**
    - **Hardcoding Configuration:** Avoid hardcoding configuration values in your application code.
    - **Large Container Images:** Keep your container images small by using multi-stage builds and removing unnecessary dependencies.
    - **Ignoring Security Best Practices:** Neglecting security best practices can lead to vulnerabilities. Always follow security best practices for container images, task definitions, and IAM roles.
    - **Manual Scaling:** Manually scaling ECS tasks is inefficient and error-prone. Use auto scaling instead.
    - **Monolithic Containers:** Avoid creating single containers that run multiple applications. Break them down into smaller, single-purpose containers.

  - **2.4 State Management:**
    - **Stateless Applications:** Design your applications to be stateless whenever possible. Store state in a database or other external storage service.
    - **Persistent Storage:** Use persistent storage volumes (e.g., Amazon EFS, Amazon EBS) for stateful applications.
    - **Container Lifecycle:** Understand the container lifecycle and how ECS manages container restarts and replacements.

  - **2.5 Error Handling:**
    - **Logging:** Implement comprehensive logging and monitoring for your applications. Log to stdout/stderr, and use a logging driver (e.g., FireLens) to ship logs to a central logging service (e.g., CloudWatch Logs, Splunk).
    - **Exception Handling:** Implement proper exception handling in your application code.
    - **Retry Logic:** Implement retry logic for transient errors.
    - **Dead Letter Queues:** Use dead letter queues (DLQs) to handle messages that cannot be processed after multiple retries.

- **3. Performance Considerations:**
  - **3.1 Optimization Techniques:**
    - **Resource Allocation:** Properly size your ECS tasks based on the application's resource requirements (CPU, memory).  Monitor resource utilization and adjust task sizes as needed.
    - **Container Image Optimization:** Optimize container images by minimizing size, using efficient base images, and leveraging layer caching.  Use tools like `docker image prune` to remove unused images.
    - **Load Balancing:** Configure load balancing algorithms and connection draining to optimize traffic distribution and minimize downtime.
    - **Caching:** Implement caching at various levels (e.g., application, CDN) to reduce latency and improve performance.
    - **Connection Pooling:** Reuse database connections to minimize overhead.

  - **3.2 Memory Management:**
    - **Memory Limits:** Set appropriate memory limits for your containers.  Monitor memory usage and adjust limits to prevent out-of-memory errors.
    - **Garbage Collection:** Optimize garbage collection settings for your application's runtime environment.
    - **Memory Leaks:** Identify and fix memory leaks in your application code.

  - **3.3 Bundle Size Optimization:**
    - **Code Splitting:** Split your application code into smaller bundles to reduce the initial load time.  Utilize dynamic imports to only load code when it's needed.
    - **Tree Shaking:** Remove unused code from your application bundle.
    - **Minification:** Minify your application code to reduce its size.
    - **Compression:** Compress your application code to reduce its size.

  - **3.4 Lazy Loading:**
    - Load resources (e.g., images, data) only when they are needed.
    - Use lazy-loading techniques to improve the initial load time of your application.

- **4. Security Best Practices:**
  - **4.1 Common Vulnerabilities:**
    - **Container Image Vulnerabilities:** Unpatched vulnerabilities in container images can be exploited by attackers. Regularly scan container images for vulnerabilities and apply patches.
    - **IAM Role Misconfiguration:** Overly permissive IAM roles can allow attackers to access sensitive resources. Follow the principle of least privilege and grant only the necessary permissions.
    - **Network Security Misconfiguration:** Misconfigured network security can expose your ECS services to unauthorized access. Use security groups and network ACLs to restrict network access.
    - **Secrets Management Vulnerabilities:** Storing secrets in plain text can expose them to attackers. Use a secrets management service (e.g., AWS Secrets Manager) to securely store and manage secrets.

  - **4.2 Input Validation:**
    - Validate all input data to prevent injection attacks (e.g., SQL injection, command injection).
    - Use a framework or library to perform input validation.

  - **4.3 Authentication and Authorization:**
    - Implement authentication and authorization to control access to your ECS services.
    - Use a standard authentication protocol (e.g., OAuth 2.0, OpenID Connect).
    - Use fine-grained authorization policies to restrict access to specific resources.

  - **4.4 Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS to encrypt data in transit.
    - Use AWS KMS to encrypt data at rest.
    - Implement data masking and tokenization to protect sensitive data.

  - **4.5 Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement authentication and authorization for all API endpoints.
    - Validate all input data to prevent injection attacks.
    - Protect against Cross-Site Request Forgery (CSRF) attacks.

- **5. Testing Approaches:**
  - **5.1 Unit Testing:**
    - Write unit tests for individual components of your application.
    - Use a unit testing framework (e.g., JUnit, pytest).
    - Mock external dependencies to isolate the component being tested.

  - **5.2 Integration Testing:**
    - Write integration tests to verify the interaction between different components of your application.
    - Use a test environment that is similar to the production environment.
    - Test the integration with external services (e.g., databases, message queues).

  - **5.3 End-to-End Testing:**
    - Write end-to-end tests to verify the functionality of the entire application.
    - Use an end-to-end testing framework (e.g., Selenium, Cypress).
    - Test the application from the user's perspective.

  - **5.4 Test Organization:**
    - Organize your tests into a logical directory structure.
    - Use meaningful names for your test files and test methods.
    - Use a test runner to execute your tests.

  - **5.5 Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components during testing.
    - Use a mocking framework (e.g., Mockito, EasyMock).
    - Create mock objects that simulate the behavior of external dependencies.

- **6. Common Pitfalls and Gotchas:**
  - **6.1 Frequent Mistakes:**
    - **Incorrect IAM Permissions:** Granting insufficient or excessive IAM permissions can lead to security vulnerabilities or application failures.
    - **Misconfigured Network Settings:** Misconfigured network settings can prevent containers from communicating with each other or with external services.
    - **Insufficient Resource Limits:** Setting insufficient resource limits (CPU, memory) can cause containers to crash or become unresponsive.
    - **Ignoring Health Checks:** Ignoring health checks can lead to ECS tasks being considered healthy even when they are not.
    - **Failing to handle SIGTERM signals:**  Applications in containers must gracefully handle SIGTERM signals to allow ECS to shutdown tasks cleanly.

  - **6.2 Edge Cases:**
    - **Task Placement Constraints:** Be aware of task placement constraints (e.g., availability zone, instance type) and how they can affect task scheduling.
    - **Service Discovery Limitations:** Understand the limitations of service discovery and how it can affect service resolution.
    - **Load Balancer Connection Draining:** Be aware of load balancer connection draining and how it can affect application availability during deployments.

  - **6.3 Version-Specific Issues:**
    - Stay up-to-date with the latest ECS agent and Docker versions to avoid known issues and security vulnerabilities.
    - Check the AWS documentation for any version-specific issues or known bugs.

  - **6.4 Compatibility Concerns:**
    - Be aware of compatibility issues between ECS and other AWS services (e.g., VPC, IAM, CloudWatch).
    - Test your application with different versions of these services to ensure compatibility.

  - **6.5 Debugging Strategies:**
    - **Logging:** Use comprehensive logging to track application behavior and identify errors.
    - **Monitoring:** Use monitoring tools (e.g., CloudWatch) to track resource utilization and application performance.
    - **Debugging Tools:** Use debugging tools (e.g., Docker exec, AWS Systems Manager Session Manager) to troubleshoot running containers.
    - **ECS Exec:** Utilize the ECS Exec feature to directly connect to containers for debugging purposes.

- **7. Tooling and Environment:**
  - **7.1 Recommended Tools:**
    - **Terraform:** For infrastructure as code.
    - **AWS CLI:** For interacting with AWS services from the command line.
    - **Docker:** For building and managing container images.
    - **AWS SAM CLI:** For local development of serverless applications.
    - **Visual Studio Code (VS Code):** With AWS and Docker extensions for development and debugging.
    - **cdk8s:** Define Kubernetes applications and ECS using general purpose programming languages.

  - **7.2 Build Configuration:**
    - **Makefile:** Use a Makefile to automate build, test, and deployment tasks.
    - **Build Scripts:** Use build scripts to customize the build process.
    - **CI/CD Pipeline:** Integrate your build process with a CI/CD pipeline (e.g., AWS CodePipeline, Jenkins).

  - **7.3 Linting and Formatting:**
    - **Linters:** Use linters (e.g., ESLint, Pylint) to enforce code style and identify potential errors.
    - **Formatters:** Use formatters (e.g., Prettier, Black) to automatically format your code.
    - **Editor Integration:** Integrate linters and formatters with your code editor.

  - **7.4 Deployment:**
    - **Blue/Green Deployments:** Use blue/green deployments to minimize downtime during deployments.
    - **Canary Deployments:** Use canary deployments to gradually roll out new versions of your application.
    - **Rolling Updates:** Use rolling updates to gradually update ECS tasks with the new version.

  - **7.5 CI/CD Integration:**
    - **Automated Builds:** Automate the build process using a CI/CD pipeline.
    - **Automated Tests:** Run automated tests as part of the CI/CD pipeline.
    - **Automated Deployments:** Automate the deployment process using a CI/CD pipeline.