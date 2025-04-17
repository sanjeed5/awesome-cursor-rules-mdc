---
description: Comprehensive rules and best practices for AWS development using Terraform, covering code organization, security, performance, and testing.  Adherence to these guidelines ensures maintainable, secure, and efficient infrastructure code.
globs: *.tf
---
# AWS Development Best Practices using Terraform

This document outlines best practices for developing and maintaining AWS infrastructure using Terraform. Following these guidelines will help ensure your code is maintainable, secure, and efficient.

## 1. Code Organization and Structure

### 1.1 Directory Structure

-   **Root Module**: The primary entry point for your infrastructure definition. Contains `main.tf`, `variables.tf`, `outputs.tf`, `providers.tf`, `versions.tf` and `locals.tf`.
-   **Modules Directory**:  Reusable components of your infrastructure. Each module should have its own directory under `modules/`.
-   **Examples Directory**: Demonstrates how to use the modules. Each example should be a self-contained Terraform configuration.
-   **Scripts Directory**: Custom scripts used by Terraform.
-   **Templates Directory**:  Templates for files that are read using the `templatefile` function, using the extension `.tftpl`.
-   **Files Directory**: Contains static files referenced by Terraform (e.g., startup scripts).
-   **Helpers Directory**:  Contains utility scripts that are *not* directly called by Terraform.
-   **.tfvars files**: Files containing variable definitions.  Use `terraform.tfvars` for common values and create environment specific files in the `envs/` directory (e.g., `envs/dev/terraform.tfvars`, `envs/prod/terraform.tfvars`).
-   **.gitignore**:  Specifies intentionally untracked files that Git should ignore.
-   **docs/**: Place any supplemental documentation in a `docs/` subdirectory.
-   **CODEOWNERS**:  File to document who is responsible for the module. Before any merge request is merged, an owner should approve it.

Example:


├── .terraform/              # Terraform working directory
├── modules/
│   ├── ec2/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   ├── s3/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
├── examples/
│   ├── ec2/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   ├── s3/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
├── scripts/
│   ├── configure_instance.sh
├── templates/
│   ├── user_data.tftpl
├── files/
│   ├── application.conf
├── helpers/
│   ├── backup_database.sh
├── terraform.tfvars
├── envs/
│   ├── dev/
│   │   ├── terraform.tfvars
│   ├── prod/
│   │   ├── terraform.tfvars
├── main.tf
├── variables.tf
├── outputs.tf
├── providers.tf
├── versions.tf
├── locals.tf
└── .gitignore


### 1.2 File Naming Conventions

-   `main.tf`: Primary entry point for resource definitions.
-   `variables.tf`:  Variable declarations.
-   `outputs.tf`: Output declarations.
-   `providers.tf`:  Provider configuration.
-   `versions.tf`:  Terraform and provider version constraints.
-   `locals.tf`: Local variable definitions.
-   `*.tfvars`: Variable value files. Use `terraform.tfvars` for default values and `env/<environment>.tfvars` for environment-specific values.

### 1.3 Module Organization

-   **Reusable Modules**: Create modules for common infrastructure components (e.g., EC2 instances, S3 buckets, VPCs).
-   **Module Composition**:  Compose complex infrastructure by combining smaller, reusable modules.
-   **Module Versioning**: Use version control (e.g., Git tags) to manage module versions.
-   **Module Naming**: Module repositories must use the format `terraform-<PROVIDER>-<NAME>`.  For example: `terraform-aws-ec2`
-   **Module sources**: When referencing another module on a public registry, pin the Git commit hash by pointing directly to the underlying Git repo URL. Example:

    terraform
    module "lambda" {
        source = "github.com/terraform-aws-modules/terraform-aws-lambda.git?ref=e78cdf1f82944897ca6e30d6489f43cf24539374" # --> v4.18.0
    }
    

### 1.4 Component Architecture

-   **Layered Architecture**: Separate infrastructure into logical layers (e.g., network, compute, data).
-   **Loose Coupling**: Design components to be loosely coupled, minimizing dependencies between them.
-   **Clear Abstractions**: Use modules to create clear abstractions for complex infrastructure components.

### 1.5 Code Splitting Strategies

-   **Functional Decomposition**: Split code into smaller, manageable functions.
-   **Resource Grouping**: Group related resources into separate files or modules.
-   **Environment Separation**: Use separate files or modules for environment-specific configurations.
-   **Service-Based Separation**: Separate resources by service (e.g., `iam.tf`, `ec2.tf`, `s3.tf`) only if they exceed 150 lines in `main.tf`.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to AWS

-   **Immutable Infrastructure**:  Deploy new infrastructure components instead of modifying existing ones.
-   **Infrastructure as Code (IaC)**: Manage infrastructure through code, using tools like Terraform.
-   **Blue/Green Deployments**:  Deploy new versions of applications alongside existing ones, then switch traffic.
-   **Canary Deployments**:  Deploy new versions of applications to a small subset of users before rolling them out to everyone.

### 2.2 Recommended Approaches for Common Tasks

-   **Creating VPCs**: Use the `terraform-aws-modules/vpc/aws` module for creating VPCs.
-   **Creating EC2 Instances**: Use the `terraform-aws-modules/ec2-instance/aws` module for creating EC2 instances.
-   **Creating S3 Buckets**: Use the `terraform-aws-modules/s3-bucket/aws` module for creating S3 buckets.
-   **Managing IAM Roles**: Use the `aws_iam_role` and `aws_iam_policy` resources to manage IAM roles and policies.
-   **Using Data Sources**: Utilize data sources (e.g., `aws_ami`, `aws_vpc`) to fetch information about existing AWS resources.
-   **Resource Tagging**:  Tag all AWS resources for cost allocation, automation, and identification.
-   **State File Storage**: Use a remote backend (e.g., S3 with DynamoDB locking) for storing Terraform state.

### 2.3 Anti-patterns and Code Smells to Avoid

-   **Hardcoding Values**:  Avoid hardcoding values in your code. Use variables instead.
-   **Storing Secrets in Code**:  Never store secrets (e.g., passwords, API keys) in your code. Use AWS Secrets Manager or other secret management solutions.
-   **Overly Complex Modules**:  Avoid creating modules that are too complex and difficult to understand.
-   **Ignoring Errors**: Always handle errors gracefully.
-   **Manual Infrastructure Changes**: Avoid making manual changes to infrastructure outside of Terraform.
-   **Using embedded resource attributes**: avoid using these embedded resource attributes and instead you should use the unique resource to attach that pseudo-resource. These resource relationships can cause chicken/egg issues that are unique per resource.

### 2.4 State Management Best Practices

-   **Remote Backend**:  Use a remote backend (e.g., S3 with DynamoDB locking) for storing Terraform state. This allows for collaboration and prevents data loss.
-   **State Locking**:  Enable state locking to prevent concurrent modifications to the state file.
-   **State Encryption**:  Encrypt the state file at rest and in transit.
-   **Workspace Management**:  Use Terraform workspaces to manage multiple environments (e.g., development, staging, production).
-   **Secure State Access**: Restrict access to the state file to authorized users and systems only. Consider implementing least privilege principles.
-   **Versioning**: Implement a mechanism for versioning your state files, which could include frequent backups and using version control systems.
-   **Avoid local state**: Never store Terraform state locally.

### 2.5 Error Handling Patterns

-   **Validation Rules**: Use validation rules within variable definitions to check input values.
-   **Error Propagation**:  Propagate errors up the call stack to be handled at a higher level.
-   **Retry Logic**: Implement retry logic for transient errors (e.g., network errors, API throttling).
-   **Logging**:  Log errors and warnings to aid in debugging.
-   **Custom Error Messages**: Provide clear and informative error messages to users.
-   **Panic Handling**: Avoid using `panic()` function that stops the execution, implement error handling.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Efficient Resource Usage**:  Use the appropriate instance types and storage classes for your workloads.
-   **Caching**: Implement caching to reduce latency and improve performance.
-   **Load Balancing**: Use load balancing to distribute traffic across multiple instances.
-   **Content Delivery Networks (CDNs)**: Use CDNs to deliver static content closer to users.
-   **Database Optimization**: Optimize database queries and indexing for faster performance.
-   **Parallelization**: Use `for_each` and `count` to parallelize resource creation and modification.

### 3.2 Memory Management

-   **Resource Limits**:  Set resource limits on EC2 instances and other AWS resources.
-   **Memory Profiling**:  Use memory profiling tools to identify memory leaks and optimize memory usage.
-   **Garbage Collection**:  Configure garbage collection settings for your applications.

### 3.3 Rendering Optimization

-   **Minimize Rendering**: Minimize the amount of data that needs to be rendered.
-   **Efficient Templates**: Use efficient templates to generate configuration files.

### 3.4 Bundle Size Optimization

-   **Code Splitting**: Split code into smaller bundles that can be loaded on demand.
-   **Tree Shaking**: Remove unused code from your bundles.
-   **Minification**: Minify your code to reduce bundle sizes.
-   **Compression**: Compress your bundles to reduce download times.

### 3.5 Lazy Loading Strategies

-   **Lazy Loading of Resources**: Load resources only when they are needed.
-   **Virtualization**: Use virtualization to reduce the number of resources that need to be loaded.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

-   **IAM Misconfiguration**: Prevent IAM misconfiguration by following the principle of least privilege.
-   **Unsecured S3 Buckets**: Secure S3 buckets by enabling encryption, access logging, and versioning.
-   **Security Group Misconfiguration**:  Configure security groups to allow only necessary traffic.
-   **SQL Injection**:  Prevent SQL injection by using parameterized queries and input validation.
-   **Cross-Site Scripting (XSS)**:  Prevent XSS by sanitizing user input and output.
-   **Cross-Site Request Forgery (CSRF)**: Prevent CSRF by using anti-CSRF tokens.
-   **Exposed Secrets**: Avoid exposing secrets by using AWS Secrets Manager and IAM roles.

### 4.2 Input Validation

-   **Validate User Input**: Validate user input to prevent malicious code from being injected into your application.
-   **Use Data Types**: Use appropriate data types for your variables to prevent type-related errors.
-   **Regular Expressions**: Use regular expressions to validate complex input patterns.

### 4.3 Authentication and Authorization Patterns

-   **IAM Roles**:  Use IAM roles to grant permissions to AWS resources.
-   **Multi-Factor Authentication (MFA)**:  Enable MFA for all user accounts.
-   **Principle of Least Privilege**:  Grant users only the permissions they need to perform their tasks.
-   **AWS Cognito**:  Use AWS Cognito to manage user authentication and authorization.
-   **API Gateway Authorization**: Implement authorization at the API Gateway level to control access to your APIs.

### 4.4 Data Protection Strategies

-   **Encryption at Rest**:  Encrypt data at rest using AWS KMS, S3 encryption, or other encryption solutions.
-   **Encryption in Transit**:  Encrypt data in transit using HTTPS, TLS, or other encryption protocols.
-   **Data Masking**: Mask sensitive data to prevent unauthorized access.
-   **Data Redaction**: Redact sensitive data from logs and other output.

### 4.5 Secure API Communication

-   **HTTPS**: Use HTTPS for all API communication.
-   **API Keys**:  Use API keys to authenticate API requests.
-   **OAuth 2.0**: Use OAuth 2.0 for authorization.
-   **Rate Limiting**:  Implement rate limiting to prevent abuse.
-   **Web Application Firewall (WAF)**:  Use a WAF to protect your APIs from common web attacks.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

-   **Test Individual Modules**: Unit test individual Terraform modules to ensure they are working correctly.
-   **Mock AWS Resources**: Mock AWS resources to isolate your modules from external dependencies.
-   **Test Variable Validation**: Verify variable validation rules work as expected.
-   **Test Output Values**: Check that module outputs are correct.

### 5.2 Integration Testing

-   **Test Module Integration**:  Test how modules interact with each other.
-   **Deploy to Test Environment**: Deploy your infrastructure to a test environment and verify that it is working correctly.
-   **Automated Tests**: Write automated tests to verify that your infrastructure is working correctly.

### 5.3 End-to-End Testing

-   **Test Full Infrastructure**:  Test the entire infrastructure from end to end.
-   **Simulate User Behavior**:  Simulate user behavior to verify that your infrastructure is working correctly.
-   **Monitor Performance**: Monitor performance to identify any performance bottlenecks.

### 5.4 Test Organization

-   **Separate Test Directory**: Create a separate directory for your tests.
-   **Test Naming Conventions**: Use clear naming conventions for your tests.
-   **Test Documentation**: Document your tests to explain what they are testing.

### 5.5 Mocking and Stubbing

-   **Use Mock Objects**: Use mock objects to isolate your code from external dependencies.
-   **Use Stub Objects**: Use stub objects to provide canned responses to external dependencies.
-   **Terratest**:  Utilize Terratest for creating test environments and performing infrastructure tests.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

-   **Forgetting to Enable State Locking**:  Enabling state locking is crucial for preventing corruption and ensuring integrity when multiple users work on the same Terraform project.
-   **Overlooking IAM Permissions**:  Ensuring the Terraform service account has the necessary IAM permissions to create, modify, and delete resources is vital.
-   **Incorrect Resource Dependencies**:  Managing resource dependencies correctly to prevent errors during creation or deletion.
-   **Handling Sensitive Data Insecurely**:  Implementing robust measures for managing sensitive data, such as API keys and passwords, by using AWS Secrets Manager or other appropriate services.
-   **Neglecting to Monitor Infrastructure**:  Setting up comprehensive monitoring and alerting to detect and respond to infrastructure issues promptly.
-   **Using hardcoded values**: Always use variables

### 6.2 Edge Cases to be Aware Of

-   **Resource Limits**:  Be aware of AWS resource limits and plan accordingly.
-   **Availability Zones**: Understand the implications of availability zones and design your infrastructure to be resilient to AZ failures.
-   **Data Consistency**: Understand the consistency models of different AWS services and design your applications accordingly.
-   **API Throttling**: Be aware of API throttling limits and implement retry logic.

### 6.3 Version-Specific Issues

-   **Terraform Version Compatibility**:  Ensure that your Terraform code is compatible with the version of Terraform you are using.
-   **Provider Version Compatibility**: Ensure that your Terraform providers are compatible with the version of Terraform you are using.
-   **AWS API Version Compatibility**: Be aware of changes to the AWS API and update your code accordingly.

### 6.4 Compatibility Concerns

-   **Backwards Compatibility**: Ensure that your changes are backwards compatible.
-   **Cross-Region Compatibility**:  Ensure that your infrastructure is compatible across different AWS regions.
-   **Cross-Account Compatibility**: Ensure that your infrastructure is compatible across different AWS accounts.

### 6.5 Debugging Strategies

-   **Terraform Plan**: Use `terraform plan` to preview changes before applying them.
-   **Terraform Show**: Use `terraform show` to inspect the current state of your infrastructure.
-   **Terraform Graph**: Use `terraform graph` to visualize the dependencies between resources.
-   **Terraform Console**: Use `terraform console` to evaluate Terraform expressions.
-   **AWS CloudTrail**:  Use AWS CloudTrail to audit API calls and identify errors.
-   **CloudWatch Logs**:  Use CloudWatch Logs to collect and analyze application logs.
-   **Enable Debug Logging**: Enable debug logging to get more detailed information about what Terraform is doing.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

-   **Terraform CLI**: The Terraform command-line interface.
-   **IDE with Terraform Support**: Use an IDE with Terraform support (e.g., VS Code with the Terraform extension).
-   **tfenv**:  Use tfenv to manage multiple versions of Terraform.
-   **AWS CLI**:  The AWS command-line interface.
-   **jq**:  A command-line JSON processor.

### 7.2 Build Configuration

-   **Makefile**:  Use a Makefile to automate common build tasks (e.g., `terraform init`, `terraform plan`, `terraform apply`).
-   **.terraformignore**: Create a `.terraformignore` to prevent files from being included in the Terraform package.

### 7.3 Linting and Formatting

-   **terraform fmt**:  Use `terraform fmt` to automatically format your Terraform code.
-   **terraform validate**:  Use `terraform validate` to validate your Terraform code.
-   **tflint**: Use `tflint` to lint your Terraform code.
-   **pre-commit hooks**: Enforce code quality standards using pre-commit hooks.

### 7.4 Deployment Best Practices

-   **Automated Deployments**: Use automated deployments to reduce errors and improve consistency.
-   **Infrastructure as Code (IaC)**: Manage infrastructure as code using tools like Terraform.
-   **Immutable Infrastructure**: Deploy new infrastructure components instead of modifying existing ones.
-   **Blue/Green Deployments**: Deploy new versions of applications alongside existing ones, then switch traffic.
-   **Canary Deployments**:  Deploy new versions of applications to a small subset of users before rolling them out to everyone.

### 7.5 CI/CD Integration

-   **Integrate with CI/CD Pipeline**: Integrate Terraform with your CI/CD pipeline.
-   **Automated Testing**: Run automated tests as part of your CI/CD pipeline.
-   **Automated Deployments**: Automatically deploy your infrastructure as part of your CI/CD pipeline.
-   **Version Control**: Store your Terraform code in version control.
-   **Terraform Cloud/Enterprise**: Use Terraform Cloud/Enterprise for remote state management and collaboration.

This document provides a comprehensive set of best practices for AWS development using Terraform. By following these guidelines, you can ensure your code is maintainable, secure, and efficient.