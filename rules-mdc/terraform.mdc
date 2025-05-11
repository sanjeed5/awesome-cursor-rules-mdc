---
description: This rule provides guidelines for Terraform best practices, coding standards, and security considerations to ensure maintainable, efficient, and secure infrastructure-as-code.
globs: *.tf
---
- **Use Remote State**: Store Terraform state files remotely to enable collaboration and prevent conflicts. Consider using Terraform Cloud, HashiCorp Consul, AWS S3 with DynamoDB locking, or Azure Storage Account with Blob Storage Locking.
	- Properly configure access controls for your remote state backend to prevent unauthorized access.
	- Implement versioning for state files to track changes and facilitate rollbacks.
- **Consistent Naming Conventions**: Adopt a consistent naming convention for resources, modules, and variables to improve readability and maintainability.
	- Use nouns for resource names (e.g., `aws_instance.web_server` instead of `web_server_instance`).
	- Include environment or project context in names (e.g., `dev_web_server` or `prod_db`).
	- Follow a consistent casing style (e.g., snake_case or camelCase).
- **Modularity and Code Structure**: Organize Terraform code into modules to promote reusability and maintainability.
	- Create modules for common infrastructure components (e.g., VPC, EC2 instance, database).
	- Follow a standard module structure (e.g., `variables.tf`, `outputs.tf`, `main.tf`, `versions.tf`).
	- Limit the use of custom scripts within modules; prefer Terraform resources and data sources.
- **Validation and Formatting**: Always validate and format Terraform code using `terraform fmt` and `terraform validate` to ensure quality and consistency.
	- Integrate `terraform fmt` and `terraform validate` into your CI/CD pipeline.
	- Use a linter such as TFLint to enforce organization-specific coding best practices.
- **Use Existing Shared and Community Modules**: Leverage pre-built modules from the Terraform Registry or other trusted sources to avoid reinventing the wheel.
	- Thoroughly review modules before use to understand their functionality and security implications.
	- Pin module versions to prevent unexpected changes.
- **Import Existing Infrastructure**: Use the `terraform import` command to bring existing infrastructure under Terraform management.
	- Understand the limitations of `terraform import` and manually verify the imported configuration.
- **Avoid Hard-coding Variables**: Use variables to parameterize Terraform configurations and avoid hard-coding values.
	- Define variables in `variables.tf` with appropriate descriptions, types, and default values.
	- Use environment variables or Terraform Cloud variables to pass sensitive values.
- **Tag Resources**: Tag all Terraform resources with relevant metadata (e.g., `Name`, `Environment`, `Project`, `Owner`).
	- Use consistent tagging conventions across your infrastructure.
	- Leverage tags for cost allocation and resource management.
- **Introduce Policy as Code**: Implement policy as code using tools like HashiCorp Sentinel or Open Policy Agent (OPA) to enforce compliance and security policies.
	- Define policies for resource configurations, naming conventions, and security settings.
	- Integrate policy checks into your CI/CD pipeline.
- **Code Organization and Structure:**
	- **Directory Structure Best Practices:** Organize your Terraform project with a clear directory structure. A common pattern:
		
		├── modules/
		│   ├── vpc/
		│   │   ├── main.tf
		│   │   ├── variables.tf
		│   │   └── outputs.tf
		│   ├── ec2/
		│   │   ├── main.tf
		│   │   ├── variables.tf
		│   │   └── outputs.tf
		├── environments/
		│   ├── dev/
		│   │   ├── main.tf
		│   │   ├── variables.tf
		│   │   └── terraform.tfvars
		│   ├── prod/
		│   │   ├── main.tf
		│   │   ├── variables.tf
		│   │   └── terraform.tfvars
		├── main.tf  # Top-level resources (if any)
		├── variables.tf # Global variables
		└── outputs.tf   # Global outputs
		
	- **File Naming Conventions:** Adhere to consistent file naming. Use `main.tf` for the primary resource definitions, `variables.tf` for variables, `outputs.tf` for outputs, and `terraform.tfvars` for environment-specific variable values.
	- **Module Organization:** Keep modules self-contained and reusable. Each module should have a specific purpose (e.g., creating a VPC, an EC2 instance, or a database).
	- **Component Architecture:** Design your infrastructure as a collection of loosely coupled components (modules) that can be composed together.
	- **Code Splitting Strategies:** Break down large configurations into smaller, more manageable modules and files.
- **Common Patterns and Anti-patterns:**
	- **Design Patterns:**
		- **Singleton Pattern:** Ensure only one instance of a critical resource exists (e.g., a VPC). Use `count = var.create_vpc ? 1 : 0` to conditionally create a single VPC.
		- **Factory Pattern:** Use modules to create multiple instances of a resource with different configurations (e.g., multiple EC2 instances with different sizes and roles).
		- **Facade Pattern:** Create a module that simplifies the creation of complex infrastructure by abstracting away the underlying details.
	- **Recommended Approaches:**
		- Use data sources to retrieve information about existing resources instead of hardcoding their IDs or names.
		- Use dynamic blocks to create multiple resources or configure resource properties based on variable values.
		- Use lifecycle rules to manage resource creation, modification, and deletion.
	- **Anti-patterns:**
		- **Hardcoding values:** Avoid hardcoding values in your Terraform configurations. Use variables instead.
		- **Creating monolithic configurations:** Break down large configurations into smaller, more manageable modules.
		- **Ignoring errors:** Always handle errors and provide meaningful error messages.
	- **State Management Best Practices:**
		- **Remote State:** Always use remote state to store Terraform state files.
		- **State Locking:** Enable state locking to prevent concurrent modifications.
		- **State Encryption:** Encrypt state files to protect sensitive data.
		- **State Versioning:** Implement versioning for state files.
	- **Error Handling Patterns:**
		- Use the `try` and `can` functions to handle errors when retrieving data or evaluating expressions.
		- Use `validation` blocks to validate variable values and prevent invalid configurations.
		- Provide meaningful error messages to help users diagnose and fix issues.
- **Performance Considerations:**
	- **Optimization Techniques:**
		- Use the `count` and `for_each` meta-arguments to create multiple resources efficiently.
		- Use data sources to retrieve information about existing resources instead of creating new ones.
		- Use the `depends_on` meta-argument sparingly to avoid unnecessary dependencies.
	- **Memory Management:**
		- Be mindful of the memory usage of your Terraform configurations, especially when working with large datasets.
		- Avoid creating large variables or outputs that can consume excessive memory.
	- **Rendering Optimization:**
		- Use efficient string interpolation techniques to avoid unnecessary string concatenation.
		- Use the `templatefile` function to render complex templates efficiently.
- **Security Best Practices:**
	- **Common Vulnerabilities:**
		- **Hardcoded secrets:** Avoid hardcoding secrets in your Terraform configurations.
		- **Publicly accessible resources:** Ensure that resources are not publicly accessible unless explicitly required.
		- **Insufficient access controls:** Implement strict access controls to prevent unauthorized access to resources.
	- **Input Validation:**
		- Validate variable values to prevent invalid or malicious input.
		- Use regular expressions to enforce specific input formats.
	- **Authentication and Authorization Patterns:**
		- Use IAM roles and policies to grant resources the necessary permissions.
		- Use Terraform Cloud or other secrets management tools to manage sensitive credentials.
	- **Data Protection Strategies:**
		- Encrypt sensitive data at rest and in transit.
		- Use encryption keys managed by a key management service (KMS).
	- **Secure API Communication:**
		- Use HTTPS for all API communication.
		- Validate API responses to prevent data injection attacks.
- **Testing Approaches:**
	- **Unit Testing Strategies:**
		- Use `terraform show` and `terraform plan` to verify that your Terraform configurations create the expected resources.
		- Use `terratest` or other testing frameworks to write automated unit tests.
	- **Integration Testing:**
		- Deploy your Terraform configurations to a test environment and verify that the resources are functioning correctly.
		- Use automated testing tools to perform integration tests.
	- **End-to-end Testing:**
		- Simulate real-world scenarios and verify that your infrastructure can handle them.
		- Use automated testing tools to perform end-to-end tests.
	- **Test Organization:**
		- Organize your tests into a clear directory structure.
		- Use meaningful test names to describe the purpose of each test.
	- **Mocking and Stubbing:**
		- Use mocking and stubbing to isolate your tests and prevent dependencies on external resources.
		- Use testing frameworks that support mocking and stubbing.
- **Common Pitfalls and Gotchas:**
	- **Frequent Mistakes:**
		- **Incorrect resource dependencies:** Ensure that resource dependencies are correctly defined.
		- **Ignoring resource lifecycle:** Understand the lifecycle of Terraform resources and how they are created, modified, and deleted.
		- **Using outdated Terraform versions:** Keep your Terraform version up to date to take advantage of new features and bug fixes.
	- **Edge Cases:**
		- **Handling resource conflicts:** Be prepared to handle resource conflicts that can occur when multiple Terraform configurations are applied simultaneously.
		- **Managing resources with external dependencies:** Be aware of resources that have external dependencies (e.g., DNS records) and handle them appropriately.
	- **Version-specific Issues:**
		- Be aware of version-specific issues and compatibility concerns when upgrading Terraform or provider versions.
		- Consult the Terraform and provider documentation for any breaking changes or migration guides.
	- **Compatibility Concerns:**
		- Ensure that your Terraform configurations are compatible with the target infrastructure environment.
		- Use provider versions that are compatible with the Terraform version.
	- **Debugging Strategies:**
		- Use the `terraform plan` command to preview the changes that will be made to your infrastructure.
		- Use the `terraform apply` command with the `-auto-approve` flag to apply changes automatically.
		- Use the `terraform show` command to inspect the current state of your infrastructure.
- **Tooling and Environment:**
	- **Recommended Development Tools:**
		- **Terraform CLI:** The official Terraform command-line interface.
		- **Terraform Cloud/Enterprise:** A collaboration and automation platform for Terraform.
		- **IDE/Text Editor:** Visual Studio Code with the Terraform extension, Atom, or Sublime Text.
		- **TFLint:** A linter for Terraform code.
		- **Terratest:** A testing framework for Terraform code.
	- **Build Configuration:**
		- Use a consistent build configuration across all environments.
		- Use environment variables or Terraform Cloud variables to configure the build environment.
	- **Linting and Formatting:**
		- Integrate linting and formatting into your CI/CD pipeline.
		- Use `terraform fmt` and TFLint to ensure code quality and consistency.
	- **Deployment Best Practices:**
		- Use a CI/CD pipeline to automate Terraform deployments.
		- Use version control to track changes to your Terraform configurations.
		- Use infrastructure-as-code principles to manage your infrastructure.
	- **CI/CD Integration:**
		- Integrate Terraform into your CI/CD pipeline using tools like Jenkins, GitLab CI, or GitHub Actions.
		- Automate the execution of `terraform plan` and `terraform apply` commands.
		- Implement automated testing and validation as part of the CI/CD process.