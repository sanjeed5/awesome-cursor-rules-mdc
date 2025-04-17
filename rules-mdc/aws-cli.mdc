---
description: This rule provides comprehensive guidance on best practices for developing and managing AWS resources using the AWS CLI, covering code organization, security, performance, and testing.
globs: *.sh,*.py,*.yml,*.yaml,*.tf,*.json,*.md
---
# AWS CLI Best Practices and Coding Standards

This document provides guidelines and recommendations for developing and managing AWS resources using the AWS CLI. Following these practices enhances code quality, security, and operational efficiency.

## 1. Code Organization and Structure

### 1.1. Directory Structure

A well-organized directory structure promotes maintainability and readability.


project-root/
├── scripts/                  # Shell scripts for common AWS CLI operations
│   ├── create-s3-bucket.sh
│   ├── deploy-lambda.sh
│   └── ...
├── modules/                  # Reusable modules (e.g., Python scripts)
│   ├── s3_utils.py           # S3 utility functions
│   ├── iam_utils.py          # IAM utility functions
│   └── ...
├── config/                   # Configuration files
│   ├── aws_config.yml        # AWS CLI configuration
│   └── ...
├── terraform/                # Infrastructure as Code (IaC) using Terraform (optional)
│   ├── main.tf
│   ├── variables.tf
│   └── ...
├── tests/                    # Test scripts
│   ├── test_s3_utils.py
│   └── ...
├── README.md                 # Project documentation
└── .gitignore                # Git ignore file


### 1.2. File Naming Conventions

*   **Scripts:** Use descriptive names (e.g., `create-ec2-instance.sh`).  Suffix with `.sh` for shell scripts.
*   **Modules:** Use descriptive names (e.g., `s3_utils.py`).  Suffix with `.py` for Python modules.
*   **Configuration:**  Use `.yml` or `.yaml` for YAML configuration files (e.g., `aws_config.yml`).
*   **Terraform:** Follow Terraform conventions (`main.tf`, `variables.tf`).
*   **Tests:** Prefix test files with `test_` (e.g., `test_s3_utils.py`).

### 1.3. Module Organization

*   **Purpose-Driven Modules:** Group related functionalities into modules (e.g., `s3_utils.py` for S3 operations).
*   **Avoid Circular Dependencies:** Design modules to minimize inter-dependencies.
*   **Clear Interfaces:** Define clear function signatures and API contracts for each module.
*   **Documentation:** Include docstrings to explain the purpose and usage of each function within a module.

### 1.4. Component Architecture

For larger projects, consider a component-based architecture. Each component encapsulates specific AWS-related functionalities. This approach promotes modularity, testability, and reusability.

Example:


components/
├── s3/
│   ├── s3_component.py
│   ├── ...
├── iam/
│   ├── iam_component.py
│   ├── ...
└── ...


### 1.5. Code Splitting

*   **Break Down Complex Scripts:** Split large, complex scripts into smaller, more manageable functions or modules.
*   **Configuration-Driven Scripts:** Use configuration files to drive script behavior, making them more flexible and reusable.
*   **Parameterization:** Parameterize scripts to accept input variables, reducing hardcoding and improving adaptability.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Facade:** Create a facade to simplify complex AWS CLI operations.
*   **Adapter:** Use adapters to translate between different data formats (e.g., JSON and YAML).
*   **Strategy:** Implement different strategies for error handling or retry mechanisms.
*   **Command Pattern:** Encapsulate AWS CLI commands as objects, facilitating command queuing, logging, and undo operations.

### 2.2. Recommended Approaches

*   **Configuration Management:** Use environment variables, configuration files, or AWS Systems Manager Parameter Store to manage AWS credentials and configurations.
*   **Idempotent Scripts:** Design scripts to be idempotent, meaning that running them multiple times has the same effect as running them once.
*   **Logging:** Implement comprehensive logging to track script execution and identify issues.
*   **Error Handling:** Use appropriate error handling mechanisms to gracefully handle exceptions.
*   **Progress Indicators:** Add progress indicators to provide feedback during long-running operations.

### 2.3. Anti-patterns

*   **Hardcoding Credentials:** Avoid hardcoding AWS credentials directly in scripts.
*   **Lack of Error Handling:** Failing to handle exceptions can lead to unexpected script termination.
*   **Overly Complex Scripts:** Overly complex scripts are difficult to maintain and debug.
*   **Ignoring Security Best Practices:** Neglecting security best practices can lead to vulnerabilities.
*   **Not using IaC:** Managing infrastructure manually instead of using Infrastructure as Code (IaC) tools such as CloudFormation or Terraform.

### 2.4. State Management

*   **AWS Systems Manager Parameter Store:** Use Parameter Store to store and manage configuration data and secrets.
*   **DynamoDB:** Use DynamoDB to store application state data.
*   **S3:** Store state data in S3 buckets, ensuring proper access control.
*   **CloudFormation Stack Outputs:** If using CloudFormation, use stack outputs to manage state information.
*   **Terraform State:** If using Terraform, properly manage the Terraform state file (e.g., using S3 backend with DynamoDB lock). Never commit the state file to version control.

### 2.5. Error Handling

*   **Try-Except Blocks:** Use `try-except` blocks in Python to handle exceptions.
*   **Error Codes:** Check for specific error codes returned by AWS CLI commands.
*   **Retry Mechanisms:** Implement retry mechanisms with exponential backoff for transient errors.
*   **Alerting:** Set up alerting to notify administrators of critical errors.
*   **Consistent Error Messages:** Provide clear, informative error messages that help in debugging.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Parallel Execution:** Use parallel processing to execute multiple AWS CLI commands concurrently.
*   **Caching:** Cache frequently accessed data to reduce API calls.
*   **Pagination:** Use pagination to retrieve large datasets in smaller chunks.
*   **Filtering:** Filter data on the server-side to reduce the amount of data transferred.
*   **Asynchronous Operations:**  Utilize asynchronous operations when possible to avoid blocking the main thread of execution.

### 3.2. Memory Management

*   **Large Datasets:** Avoid loading large datasets into memory at once.  Use iterators or generators to process data in chunks.
*   **Resource Cleanup:** Ensure that resources are properly released after use to prevent memory leaks.
*   **Efficient Data Structures:** Use efficient data structures to minimize memory consumption.

### 3.3. Rendering Optimization (If applicable)

N/A - The AWS CLI is not a rendering library.

### 3.4. Bundle Size Optimization (If applicable)

N/A - Bundle size optimization is not directly relevant to the AWS CLI itself, but it might be relevant for tools built on top of it that are distributed as standalone applications.

### 3.5. Lazy Loading

*   **Import Modules on Demand:** Import modules only when they are needed to reduce startup time.
*   **Load Configuration As Needed:** Load configuration data only when it is required.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Credential Exposure:** Exposing AWS credentials in code or configuration files.
*   **Insufficient Access Control:** Granting excessive permissions to IAM roles or users.
*   **SQL Injection (if applicable):**  If constructing SQL queries dynamically, protect against SQL injection.
*   **Cross-Site Scripting (XSS) (if applicable):** If the CLI interacts with web interfaces, sanitize inputs to prevent XSS.
*   **Command Injection:**  If constructing shell commands dynamically from user inputs, sanitize inputs to prevent command injection.

### 4.2. Input Validation

*   **Sanitize Inputs:** Sanitize all user inputs to prevent malicious code injection.
*   **Validate Data Types:** Validate that input data conforms to expected data types.
*   **Limit Input Lengths:** Limit the length of input strings to prevent buffer overflows.
*   **Regular Expressions:** Use regular expressions to validate input patterns.
*   **Whitelist Allowed Values:**  If applicable, use a whitelist of allowed values rather than a blacklist.

### 4.3. Authentication and Authorization

*   **IAM Roles:** Use IAM roles to grant permissions to AWS resources.
*   **Least Privilege:** Follow the principle of least privilege when granting permissions.
*   **Multi-Factor Authentication (MFA):** Enable MFA for all IAM users.
*   **Credential Rotation:** Rotate AWS credentials regularly.
*   **IAM Policies:** Implement granular IAM policies to control access to resources.

### 4.4. Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Access Control Lists (ACLs):** Use ACLs to control access to S3 buckets and objects.
*   **Data Masking:** Mask sensitive data to prevent unauthorized access.
*   **Regular Backups:** Create regular backups of critical data.
*   **Secure Logging:**  Ensure that logs do not contain sensitive information and are stored securely.

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **TLS:** Enforce TLS 1.2 or higher for secure connections.
*   **API Gateway:** Use API Gateway to secure and manage API endpoints.
*   **Rate Limiting:** Implement rate limiting to prevent denial-of-service attacks.
*   **WAF (Web Application Firewall):** Use WAF to protect against common web exploits.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Mock AWS CLI Calls:** Use mocking libraries to simulate AWS CLI calls and verify expected behavior.
*   **Test Individual Functions:** Write unit tests for each function or module.
*   **Boundary Conditions:** Test boundary conditions and edge cases.
*   **Assertion Libraries:** Use assertion libraries to verify expected results.

### 5.2. Integration Testing

*   **Test End-to-End Workflows:** Test complete workflows that involve multiple AWS services.
*   **Use Real AWS Resources:** Use real AWS resources in a test environment.
*   **Automated Test Execution:** Automate integration tests as part of the CI/CD pipeline.
*   **Verify Resource Creation and Deletion:** Ensure that resources are created and deleted correctly during testing.

### 5.3. End-to-End Testing

*   **Simulate Real-World Scenarios:** Simulate real-world scenarios to test the application's overall behavior.
*   **Automated Test Scripts:** Create automated test scripts to verify functionality.
*   **Monitor System Behavior:** Monitor system behavior during end-to-end tests.

### 5.4. Test Organization

*   **Dedicated Test Directory:** Store test files in a dedicated directory (e.g., `tests/`).
*   **Test Naming Conventions:** Follow consistent test naming conventions.
*   **Test Suites:** Group related tests into test suites.
*   **Test Reports:** Generate test reports to track test results.

### 5.5. Mocking and Stubbing

*   **Mock AWS CLI:** Use mocking libraries to simulate AWS CLI behavior.
*   **Stub External Dependencies:** Stub external dependencies to isolate components during testing.
*   **Control Test Data:** Use controlled test data to ensure predictable test results.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Incorrect IAM Permissions:** Granting insufficient or excessive IAM permissions.
*   **Missing Configuration:** Failing to configure the AWS CLI properly.
*   **Hardcoding Values:** Hardcoding values instead of using configuration files or environment variables.
*   **Ignoring Error Messages:** Ignoring error messages can lead to unexpected behavior.
*   **Not Using Pagination:** Retrieving incomplete datasets due to lack of pagination.

### 6.2. Edge Cases

*   **Rate Limiting:** Handling API rate limits gracefully.
*   **Service Outages:** Handling service outages or transient errors.
*   **Data Consistency:** Ensuring data consistency across multiple AWS services.
*   **Concurrency Issues:** Managing concurrency issues when multiple scripts access the same resources.

### 6.3. Version-Specific Issues

*   **API Changes:** Handling API changes across different AWS CLI versions.
*   **Deprecated Features:** Avoiding deprecated features.
*   **Compatibility Issues:** Addressing compatibility issues between different AWS CLI versions and AWS services.

### 6.4. Compatibility Concerns

*   **Operating Systems:** Ensuring compatibility across different operating systems (e.g., Linux, Windows, macOS).
*   **Programming Languages:** Addressing compatibility issues with different programming languages.
*   **Third-Party Libraries:** Managing dependencies on third-party libraries.

### 6.5. Debugging Strategies

*   **Logging:** Use comprehensive logging to track script execution.
*   **Debugging Tools:** Use debugging tools to step through code and inspect variables.
*   **Error Messages:** Analyze error messages carefully to identify the root cause of issues.
*   **Verbose Mode:** Use verbose mode to print detailed information about AWS CLI commands.
*   **AWS CloudTrail:** Use AWS CloudTrail to track API calls and identify potential issues.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **AWS CLI:** The core command-line interface for interacting with AWS services.
*   **Text Editor/IDE:** A suitable text editor or IDE for writing scripts (e.g., VS Code, Sublime Text, PyCharm).
*   **Version Control:** Git for version control.
*   **Shell:** Bash, Zsh, or PowerShell for running scripts.
*   **Python (if applicable):** Python for scripting and automation.
*   **jq:** A command-line JSON processor.
*   **yq:** A command-line YAML processor.

### 7.2. Build Configuration

*   **Makefile:** Use Makefiles to automate common build tasks.
*   **Build Scripts:** Create build scripts to package and deploy applications.
*   **Dependency Management:** Use appropriate dependency management tools (e.g., pip for Python).

### 7.3. Linting and Formatting

*   **Shellcheck:** Use Shellcheck to lint shell scripts.
*   **Pylint:** Use Pylint to lint Python code.
*   **Black:** Use Black to format Python code.
*   **YAML Lint:** Use a YAML linter to validate YAML files.

### 7.4. Deployment

*   **AWS CloudFormation:** Deploy infrastructure as code using AWS CloudFormation.
*   **AWS CodeDeploy:** Automate application deployments using AWS CodeDeploy.
*   **AWS Lambda:** Deploy serverless functions using AWS Lambda.
*   **Docker:** Containerize applications using Docker.

### 7.5. CI/CD Integration

*   **AWS CodePipeline:** Use AWS CodePipeline to automate the CI/CD process.
*   **Jenkins:** Integrate with Jenkins for continuous integration.
*   **GitHub Actions:** Use GitHub Actions for CI/CD.
*   **Automated Testing:** Integrate automated testing into the CI/CD pipeline.

## Infrastructure as Code (IaC)

*   Treat your infrastructure as code using tools like AWS CloudFormation or Terraform. This allows for version control, consistent deployments, and easier management of infrastructure changes.

## Continuous Integration/Continuous Deployment (CI/CD)

*   Implement CI/CD pipelines using AWS CodePipeline, ensuring that all code changes are automatically tested and deployed. This includes defining clear stages for building, testing, and deploying applications, and incorporating security checks throughout the pipeline.

## Security

*   Employ least-privilege access controls using IAM, enforce multi-factor authentication (MFA), and regularly review permissions. Security should be integrated at every stage of the development and deployment process.

## Automate tasks

*   Automate repetitive tasks to increase efficiency and reduce errors.

## Version control

*   Ensure robust version control for all code and configuration files.

## Prioritize testing

*   Integrate comprehensive testing at every stage to ensure code quality.

## Maintain and adapt

*   Regularly update and adapt the pipeline to evolving requirements.

## Monitor and analyze

*   Continuously monitor and analyze pipeline performance and outcomes.

## IAM Best Practices:

*   Require human users to use federation with an identity provider to access AWS using temporary credentials
*   Require workloads to use temporary credentials with IAM roles to access AWS
*   Require multi-factor authentication (MFA)
*   Update access keys when needed for use cases that require long-term credentials
*   Follow best practices to protect your root user credentials
*   Apply least-privilege permissions
*   Get started with AWS managed policies and move toward least-privilege permissions
*   Use IAM Access Analyzer to generate least-privilege policies based on access activity
*   Regularly review and remove unused users, roles, permissions, policies, and credentials
*   Use conditions in IAM policies to further restrict access
*   Verify public and cross-account access to resources with IAM Access Analyzer
*   Use IAM Access Analyzer to validate your IAM policies to ensure secure and functional permissions
*   Establish permissions guardrails across multiple accounts
*   Use permissions boundaries to delegate permissions management within an account

By adhering to these best practices, you can develop robust, secure, and efficient AWS CLI applications.