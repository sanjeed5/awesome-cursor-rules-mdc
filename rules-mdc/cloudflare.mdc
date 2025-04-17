---
description: This rule provides a comprehensive set of best practices and coding standards for developing with the Cloudflare library, specifically focusing on Terraform configurations. It aims to guide developers in creating efficient, secure, and maintainable infrastructure code.
globs: *.tf
---
# Cloudflare Terraform Best Practices

This document outlines the recommended best practices and coding standards for working with Cloudflare resources using Terraform. Following these guidelines will help you create robust, scalable, and maintainable infrastructure code.

## 1. Code Organization and Structure

### 1.1 Directory Structure

*   **Account-Based Segregation:** Organize your Terraform projects by Cloudflare accounts. This allows for clear separation of resources and access control.
*   **Zone-Based Grouping:** Within each account, further group resources by zones. This makes it easier to manage resources associated with specific domains.
*   **Product/Feature-Based Organization:** Within each zone, create subdirectories for individual Cloudflare products or features (e.g., `dns`, `page_rules`, `rulesets`, `waf`). This promotes modularity and maintainability.
*   **Environment Separation:** While not strictly enforced within a single repository, using separate Cloudflare accounts for different environments (development, staging, production) is *strongly* recommended for complete isolation.

Example:


example-tf/
├── demo_account_a  # Per account segregation of resources
│   ├── users        # Top level directory for account members as they are "zoneless"
│   │   ├── provider.tf # `provider.tf` is for configuring the providers
│   │   ├── users.tf    # `<subject>.tf` (users.tf) is for managing the individual resources
│   │   └── vars.tf     # Manage all variables for this component
│   ├── zone_a       # Group all zone based features together
│   │   ├── dns        # Individual (or grouped, your choice) of products or features to manage together
│   │   │   ├── dns.tf     # `<subject>.tf` (dns.tf) is for managing the individual resources
│   │   │   ├── provider.tf # `provider.tf` is for configuring the providers
│   │   │   └── vars.tf    # Manage all variables for this component
│   │   └── page_rules # ... same as above but for Page Rules (legacy)
│   │       ├── page_rules.tf
│   │       ├── provider.tf
│   │       └── vars.tf
│   ├── zone_b
│   │   ├── dns
│   │   │   ├── dns.tf
│   │   │   ├── provider.tf
│   │   │   └── vars.tf
│   │   └── page_rules
│   │       ├── page_rules.tf
│   │       ├── provider.tf
│   │       └── vars.tf
│   └── zone_c
│       ├── dns
│       │   ├── dns.tf
│       │   ├── provider.tf
│       │   └── vars.tf
│       └── page_rules
│           ├── page_rules.tf
│           ├── provider.tf
│           └── vars.tf
└── demo_account_b
    ├── users
    │   ├── provider.tf
    │   ├── users.tf
    │   └── vars.tf
    ├── zone_a
    │   ├── dns
    │   │   ├── dns.tf
    │   │   ├── provider.tf
    │   │   └── vars.tf
    │   └── page_rules
    │       ├── page_rules.tf
    │       ├── provider.tf
    │       └── vars.tf
    ├── zone_b
    │   ├── dns
    │   │   ├── dns.tf
    │   │   ├── provider.tf
    │   │   └── vars.tf
    │   └── page_rules
    │       ├── page_rules.tf
    │       ├── provider.tf
    │       └── vars.tf
    └── zone_c
        ├── dns
        │   ├── dns.tf
        │   ├── provider.tf
        │   └── vars.tf
        └── page_rules
            ├── page_rules.tf
            ├── provider.tf
            └── vars.tf


### 1.2 File Naming Conventions

*   `provider.tf`:  Contains provider configuration (e.g., Cloudflare provider settings).
*   `variables.tf`: Defines input variables for the module/component.
*   `outputs.tf`:  Declares output values that can be used by other modules/components.
*   `<resource_type>.tf`:  Contains resource definitions for a specific Cloudflare resource type (e.g., `dns_records.tf`, `page_rules.tf`, `waf_rules.tf`).
*   `data.tf`: Contains Terraform data sources.
*   `main.tf`: If `provider.tf` and `outputs.tf` are not required, `main.tf` should contain your main resource and data source definitions.

### 1.3 Module Organization

*   **Avoid Modules (or Use Sparingly):**  While modules can provide abstraction, they can also introduce complexity and make debugging more difficult.  Carefully consider whether a module is truly necessary before creating one.  If you do use modules, keep them small and well-defined.
*   **Module Structure:**  If using modules, follow a consistent internal structure, including `variables.tf`, `outputs.tf`, and `main.tf`.
*   **Versioning:** If you share Terraform modules (internal or public), ensure you practice module versioning best practices. A well-defined versioning strategy, such as semantic versioning is recommended.

### 1.4 Component Architecture

*   **Small, Focused Components:** Design your Terraform code as a collection of small, focused components, each responsible for a specific aspect of your Cloudflare configuration.
*   **Loose Coupling:** Minimize dependencies between components to improve reusability and reduce the impact of changes.
*   **Well-Defined Interfaces:** Clearly define the input variables and output values for each component to create well-defined interfaces.

### 1.5 Code Splitting Strategies

*   **Resource Type:** Split code based on resource type (e.g., DNS records, Page Rules).
*   **Functionality:** Split code based on logical functionality (e.g., WAF rules for specific types of attacks).
*   **Environment:**  While *strongly* recommending separate Cloudflare accounts for environments, you can also use Terraform workspaces to manage different environments within the same codebase, though this is generally discouraged due to potential blast radius of changes.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to Cloudflare

*   **Data Source-Driven Configuration:** Use data sources to dynamically fetch information about existing Cloudflare resources (e.g., zone ID, account ID) rather than hardcoding them.
*   **Looping with `for_each` and `count`:** Use `for_each` or `count` to create multiple similar resources (e.g., multiple DNS records) based on variables or data sources.  Favor `for_each` over `count` when possible, as it provides more explicit resource naming and updating behaviors.
*   **Dynamic Blocks:** Leverage dynamic blocks to conditionally create resource attributes based on variables or data sources. This allows for more flexible and reusable code.

### 2.2 Recommended Approaches for Common Tasks

*   **Creating DNS Records:** Use `cloudflare_record` resource to manage DNS records.  Utilize `for_each` to create multiple records based on a variable containing a list of record definitions.
*   **Managing Page Rules:** Use the `cloudflare_page_rule` resource to create and manage page rules. Consider using a module to encapsulate common page rule configurations.
*   **Configuring WAF Rules:** Use the `cloudflare_ruleset` and related resources to create and manage WAF rulesets.  Organize rulesets into logical groups based on functionality.
*   **Setting up Load Balancers:** Use `cloudflare_load_balancer`, `cloudflare_load_balancer_pool`, and `cloudflare_load_balancer_monitor` resources to create and manage load balancers.
*   **Worker Deployment**: Use `cloudflare_worker` to manage Cloudflare Workers.

### 2.3 Anti-patterns and Code Smells to Avoid

*   **Hardcoding Values:** Avoid hardcoding values such as API keys, account IDs, and zone IDs in your code. Use variables and data sources instead.
*   **Managing Resources Outside of Terraform:**  Terraform works best when it manages all changes to and the lifecycle of a resource. Avoid making manual changes to Cloudflare resources outside of Terraform, as this can lead to inconsistencies and drift.  If external changes are unavoidable, plan to re-import the resources or reconcile the differences.
*   **Overly Complex Modules:**  Avoid creating overly complex modules that perform too many tasks.  Break down complex configurations into smaller, more manageable components.
*   **Ignoring `terraform fmt` and `terraform validate`:** Always run `terraform fmt` to format your code and `terraform validate` to check for syntax errors and other issues before committing changes.
*   **Storing Secrets in Code:** Never store API tokens or other secrets directly in your Terraform code or version control. Use a secure secret management solution like Vault or environment variables.

### 2.4 State Management Best Practices

*   **Remote State:**  Always use remote state management (e.g., Terraform Cloud, AWS S3, Azure Blob Storage) to store your Terraform state file. This ensures that your state is stored securely and is accessible to all members of your team.
*   **State Locking:**  Enable state locking to prevent concurrent modifications to your state file.
*   **Encryption:** Encrypt your state file at rest to protect sensitive information.
*   **Backup:** Regularly back up your state file to prevent data loss.
*   **Consider Terraform Cloud:** For teams, consider using Terraform Cloud for state management, collaboration, and automation. It offers features like remote state storage, state locking, plan previews, and automated runs.

### 2.5 Error Handling Patterns

*   **Use `try` and `catch` in Expressions:** Use `try` and `catch` to handle errors in Terraform expressions gracefully.
*   **Validate Input Variables:** Validate input variables to ensure that they are in the correct format and within the expected range.
*   **Use `depends_on` Sparingly:** Use `depends_on` only when necessary to explicitly define dependencies between resources. Overuse of `depends_on` can lead to performance issues and deadlocks.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Minimize Resource Dependencies:** Reduce the number of dependencies between resources to improve plan and apply times.
*   **Use Data Sources Effectively:** Use data sources to retrieve information about existing resources rather than creating new resources when possible.
*   **Targeted Applies:** Use targeted applies (`terraform apply -target=resource`) to apply changes to specific resources rather than applying the entire configuration.

### 3.2 Memory Management

*   **Large State Files:** Be mindful of large state files. Break down large configurations into smaller, more manageable components to reduce state file size. Review state files periodically and consider refactoring if they become unwieldy.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

*   **Exposed API Keys:** Never expose your Cloudflare API keys in your code or version control. Use environment variables or a secure secret management solution instead.
*   **Insufficient Input Validation:** Validate all input variables to prevent injection attacks and other security vulnerabilities.
*   **Overly Permissive Permissions:** Grant only the necessary permissions to your Cloudflare API keys and Terraform service accounts.

### 4.2 Input Validation

*   **Variable Validation:** Use the `validation` block in variable definitions to enforce constraints on input values.  This is crucial for preventing unexpected behavior and security vulnerabilities.
*   **Regular Expressions:** Use regular expressions to validate input strings.
*   **Type Checking:** Use Terraform's type system to ensure that variables are of the correct type.

### 4.3 Authentication and Authorization Patterns

*   **API Tokens vs. Global API Key:** Prefer using API tokens with specific permissions over the global API key, as API tokens can be scoped to specific resources and actions.
*   **Least Privilege Principle:** Grant only the necessary permissions to your Terraform service accounts.
*   **Secure Storage:** Store API tokens and other secrets securely using a secret management solution.

### 4.4 Data Protection Strategies

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Masking:** Mask sensitive data in logs and error messages.
*   **Access Control:** Implement strict access control policies to protect sensitive data.

### 4.5 Secure API Communication

*   **HTTPS:** Always use HTTPS to communicate with the Cloudflare API.
*   **TLS:** Use TLS 1.2 or higher.
*   **Certificate Validation:** Validate the Cloudflare API certificate.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

*   **Validate Resource Attributes:** Write unit tests to validate the attributes of individual resources.
*   **Check for Expected Errors:** Write unit tests to check for expected errors.
*   **Use `terraform show`:** Use `terraform show` to inspect the generated Terraform configuration and verify that it is correct.

### 5.2 Integration Testing

*   **Deploy to a Test Environment:** Deploy your Terraform code to a test environment and verify that it works as expected.
*   **Automated Testing:** Automate your integration tests using a CI/CD pipeline.
*   **Check Cloudflare Resources:** After applying a Terraform configuration in a test environment, use the Cloudflare API or UI to verify that the resources have been created and configured correctly.

### 5.3 End-to-End Testing

*   **Simulate Real-World Traffic:** Simulate real-world traffic to your Cloudflare resources to verify that they are functioning correctly.
*   **Monitor Performance:** Monitor the performance of your Cloudflare resources to identify any bottlenecks or issues.

### 5.4 Test Organization

*   **Separate Test Directory:** Create a separate directory for your Terraform tests.
*   **Test Naming Conventions:** Follow consistent naming conventions for your tests.

### 5.5 Mocking and Stubbing

*   **Mock Data Sources:** Mock data sources to isolate your tests from external dependencies.
*   **Stub API Calls:** Stub API calls to control the behavior of the Cloudflare API during testing.

## 6. Common Pitfalls and Gotchas

*   **Rate Limiting:** Be aware of Cloudflare's API rate limits. Implement retry logic to handle rate limit errors.
*   **Resource Dependencies:** Understand the dependencies between Cloudflare resources. Create resources in the correct order to avoid errors.
*   **State Drift:** Regularly check for state drift and reconcile any differences between your Terraform state and your Cloudflare resources.
*   **Idempotency:** Ensure that your Terraform code is idempotent. Applying the same configuration multiple times should have the same result.
*   **Cloudflare Provider Version:** Lock the Cloudflare provider version in your `terraform` block in `main.tf`. Upgrading the provider can sometimes introduce breaking changes, so testing the upgrade in a non-production environment is suggested first.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Terraform CLI:** The official Terraform command-line interface.
*   **Terraform Language Server:** Provides code completion, syntax highlighting, and other features for Terraform code in your IDE.
*   **IDE Extensions:** Use IDE extensions for Terraform to improve your development experience (e.g., VS Code Terraform extension).
*   **cf-terraforming:** Use `cf-terraforming` tool to import existing Cloudflare resources into Terraform state.

### 7.2 Build Configuration

*   **Terraform Block:** Configure the required providers and Terraform version in the `terraform` block in `main.tf`.
*   **Provider Configuration:** Configure the Cloudflare provider with your API key and account ID.

### 7.3 Linting and Formatting

*   **`terraform fmt`:** Use `terraform fmt` to automatically format your Terraform code.
*   **`terraform validate`:** Use `terraform validate` to check for syntax errors and other issues.
*   **`tflint`:** Consider using `tflint` or other linting tools to enforce coding standards and best practices.

### 7.4 Deployment Best Practices

*   **CI/CD Pipeline:** Automate your Terraform deployments using a CI/CD pipeline.
*   **Plan Previews:** Always review the Terraform plan before applying changes.
*   **Apply with Auto-Approve (Carefully):** Use the `-auto-approve` flag with caution. Only use it in automated environments where you have a high degree of confidence in your code.
*   **Blue/Green Deployments:** Consider using blue/green deployments to minimize downtime during deployments.

### 7.5 CI/CD Integration

*   **Terraform Cloud/Enterprise:** Use Terraform Cloud or Enterprise to manage your Terraform workflows and collaborate with your team.
*   **GitHub Actions:** Use GitHub Actions to automate your Terraform deployments.
*   **GitLab CI:** Use GitLab CI to automate your Terraform deployments.
*   **Jenkins:** Use Jenkins to automate your Terraform deployments.

## 8. Angle Brackets, Square Brackets, and Curly Braces

*   **Angle Brackets (`<` and `>`):** Use angle brackets as placeholders for variables that the user must enter, except in URLs, where you should use curly braces.
    *   Example: `https://<user-specified domain>.cloudflare.com`
*   **Square Brackets (`[` and `]`):** Use square brackets to enclose optional items.
    *   Example: `tag=dns query [search tag=malware]`
*   **Curly Braces (`{` and `}`):** Use curly braces in code samples or string literals, such as placeholders in URLs.
    *   Example: `https://api.cloudflare.com/client/v4/organizations/{organization_identifier}/invites`

## 9. Cloudflare's Convention Symbols

*   The `>` symbol leads you through nested menu items and dialog box options to a final action.
    *   Example: `Options > Settings > General` directs you to pull down the Options menu, select the Settings item, and select General from the last dialog box. Do not use bold formatting for the `>` symbol.
*   Tip icon: This icon denotes a tip, which alerts you to advisory information.
*   Note icon: This icon denotes a note, which alerts you to important information.
*   Info icon: This icon denotes info, which alerts you to important information.
*   Notice icon: This icon denotes a notice, which alerts you to take precautions to avoid data loss, loss of signal integrity, or degradation of performance.
*   Caution icon: This icon denotes a caution, which advises you to take precautions to avoid injury.
*   Blue text: Text in this color indicates a link.
*   Bold: Use bold when referring to a clickable action or to highlight a title or name in the UI. Bold text denotes items that you must select or click in the software, identifiers in the UI, or parameter names. Do not use bold for programs. In nested menus, use bold for the word not the symbol.
    *   Example: `Dashboard > This > That`
*   Italics: Use italics when referring to an option that customers can select from, like in dropdown menus. Do not use italics when referring to the state of a toggle - for example, enabled/disabled should not be italicized.
*   Monospace: Text in this font denotes text or characters that you should enter from the keyboard, sections of code, programming examples, and syntax examples. This font is also used for the proper names of drives, paths, directories, programs, subprograms, devices, functions, operations, variables, files, API commands, and extensions.

By adhering to these best practices, you can create more efficient, secure, and maintainable Terraform code for managing your Cloudflare infrastructure.