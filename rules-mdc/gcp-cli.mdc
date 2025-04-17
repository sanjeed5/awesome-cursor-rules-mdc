---
description: Provides guidelines for using gcp-cli, including best practices for scripting, configuration management, security, and performance. Focuses on automation, predictable output, and secure authentication within Google Cloud environments.
globs: *.sh,*.yaml,*.tf,*.py
---
- **Scripting Best Practices**:
  - **Use the `--quiet` flag**: Suppress prompts for non-interactive script execution.
  - **Leverage output formatting**: Employ `--format=json`, `--format=yaml`, `--format=csv`, or `--format=text` for predictable, parsable output.
  - **Handle exit statuses**: Ensure scripts check exit codes for error handling. A non-zero exit status indicates an error.
  - **Avoid relying on standard error messages**: These are subject to change and shouldn't be parsed for critical logic.

- **Configuration Management**:
  - **Utilize named configurations**: Create multiple configurations using `gcloud config configurations create` for managing different environments (development, staging, production) or projects.
  - **Set properties**: Define the active account (`gcloud config set core/account`), default project (`gcloud config set core/project`), and default region/zone (`gcloud config set compute/region` and `gcloud config set compute/zone`) either globally or per-command.

- **Authentication and Authorization**:
  - **Service accounts for production**: Use service accounts with appropriate IAM roles for scripts running in production environments. Create and manage service accounts via the Google Cloud Console.
  - **User accounts for interactive sessions**: Use user accounts for interactive sessions and local development.
  - **Activate service accounts securely**: Use `gcloud auth activate-service-account --key-file [KEY_FILE]` to activate service accounts, ensuring the key file is securely managed.
  - **Avoid storing service account keys in code**: Employ environment variables or secret management solutions to store and retrieve service account keys.
  - **Impersonate Service Accounts (if applicable)**: If a user needs to act as a service account, use `gcloud config set auth/impersonate_service_account SERVICE_ACCT_EMAIL` after authenticating with `gcloud auth login`.

- **Filtering and Formatting Output**:
  - **Filter output**: Use the `--filter` flag to narrow down results based on specific criteria.
  - **Format output**: Use the `--format` flag to control the output format (JSON, YAML, CSV, Text, List).
  - **Utilize projections**: Combine `--format` with projections to extract specific fields from the output.
  - **Example: List instances in a specific zone**: `gcloud compute instances list --filter="zone:us-central1-a"`
  - **Example: List projects in JSON format**: `gcloud projects list --format="json" --filter="labels.env=test AND labels.version=alpha"`

- **Code Organization and Structure (Shell Scripts)**:
  - **Modularize scripts**: Break down large scripts into smaller, reusable functions.
  - **Use descriptive function names**: Improve readability and maintainability.
  - **Implement argument parsing**: Use `getopts` for robust argument handling.
  - **Add comments**: Explain the purpose and functionality of code blocks.
  - **Use consistent indentation**: Improves readability.

- **Code Organization and Structure (Terraform)**:
  - **Follow the Terraform Standard Directory Structure**: Organize configurations into modules.
  - **Use Modules**: Encapsulate reusable infrastructure components into modules.
  - **Variables**: Use variables to parameterize your configurations.
  - **Outputs**: Use outputs to expose important information about your infrastructure.
  - **Remote State Management**: Store Terraform state remotely (e.g., in Google Cloud Storage) for collaboration and versioning. Secure the bucket with appropriate IAM permissions.

- **Code Organization and Structure (Python)**:
  - **Package your gcp-cli interaction code into reusable modules and classes.**
  - **Follow PEP 8 Style Guide for Python Code**.
  - **Use Virtual Environments** for dependency management.

- **Common Patterns and Anti-patterns**:
  - **Pattern**: Use `xargs` or `parallel` to execute gcloud commands in parallel for faster processing.
  - **Anti-pattern**: Hardcoding credentials or sensitive information directly in scripts or configuration files. Use environment variables or secret management solutions instead.

- **Performance Considerations**:
  - **Use pagination**: When retrieving large datasets, use the `--page-size` flag to limit the number of results per page and iterate through the pages.
  - **Minimize API calls**: Batch operations where possible to reduce the number of API requests.
  - **Caching**: Cache API responses to avoid redundant requests. Implement caching mechanisms within your scripts or applications.

- **Security Best Practices**:
  - **Least Privilege**: Grant service accounts only the necessary IAM roles and permissions.
  - **Regularly rotate service account keys**: Implement a key rotation policy to minimize the impact of compromised keys.
  - **Use VPC Service Controls**: Restrict data exfiltration and unauthorized access to Google Cloud services.
  - **Audit Logging**: Enable audit logging to track API calls and resource changes.
  - **Input Validation**: Always validate user inputs and data received from external sources to prevent injection attacks.

- **Testing Approaches**:
  - **Unit tests**: Mock gcloud CLI calls to isolate and test individual components.
  - **Integration tests**: Verify the interaction between different Google Cloud services and your gcp-cli scripts.
  - **End-to-end tests**: Simulate real-world scenarios and validate the entire workflow.
  - **Use `gcloud config configurations activate` in testing** to isolate test environments.

- **Common Pitfalls and Gotchas**:
  - **Default project settings**: Ensure the correct project is configured before running gcloud commands.
  - **IAM propagation delays**: Be aware of potential delays when granting or revoking IAM permissions.
  - **API throttling**: Handle API rate limits gracefully by implementing retry mechanisms with exponential backoff.
  - **Version Compatibility**: Be aware of potential breaking changes when upgrading the gcloud CLI. Test your scripts after upgrades.

- **Tooling and Environment**:
  - **Use a dedicated development environment**: Isolate your development environment from production to avoid accidental changes.
  - **Use a version control system**: Track changes to your gcp-cli scripts and configuration files.
  - **Set up CI/CD pipelines**: Automate testing and deployment of your gcp-cli scripts.
  - **Recommended tools**: Shellcheck (for shell script linting), Terraform Validate (for Terraform configuration validation), Pylint (for python code linting).

- **Referenced Rules**:
  - @file python_best_practices.mdc
  - @file terraform_best_practices.mdc
  - @file shell_script_best_practices.mdc