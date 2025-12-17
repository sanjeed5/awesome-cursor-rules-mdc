---
description: Definitive guidelines for writing maintainable, secure, and scalable Terraform configurations. Focuses on structure, naming, state management, and automation.
globs: **/*.tf
---
# Terraform Best Practices

Terraform configurations are production-grade software. Treat them with the same rigor: readable, versioned, testable, and governed by automated policies. This guide provides opinionated, actionable best practices for our team's Terraform development, ensuring consistency, reliability, and security.

## 1. Code Organization and Structure

Organize your Terraform code for clarity, reusability, and environment isolation.

### 1.1 Project Layout

**Always** separate environments and reusable components.

❌ BAD: Monolithic `main.tf` or environment-specific resources mixed with modules.
```terraform
# environments/dev/main.tf
resource "aws_vpc" "dev_vpc" { # ... }
resource "aws_s3_bucket" "dev_app_bucket" { # ... }
# Duplicated for staging/prod
```

✅ GOOD: Dedicated directories for environments and shared modules.
```
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── backend.tf
│   │   └── terraform.tfvars
│   ├── prod/
│       ├── main.tf
│       ├── variables.tf
│       ├── backend.tf
│       └── terraform.tfvars
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── s3_bucket/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
```

### 1.2 Standard File Naming

Use HashiCorp's recommended file naming conventions for consistency.

*   `backend.tf`: Remote state configuration.
*   `main.tf`: Primary resource and module calls.
*   `outputs.tf`: All output blocks, alphabetically sorted.
*   `providers.tf`: `terraform` block (required version/providers) and `provider` blocks.
*   `variables.tf`: All variable blocks, alphabetically sorted.
*   `locals.tf`: Local values.

## 2. Naming Conventions

Consistency in naming is paramount for readability.

### 2.1 Resource and Variable Names

Use `lower_snake_case` for all resource, data source, variable, and output names. **Never** include the resource type in the name.

❌ BAD:
```terraform
resource "aws_s3_bucket" "s3_bucket_app_logs" { # ... }
variable "vpcName" { # ... }
```

✅ GOOD:
```terraform
resource "aws_s3_bucket" "app_logs" { # ... }
variable "vpc_name" { # ... }
```

### 2.2 Boolean Variables

Use positive boolean flags ending with `_enabled`.

❌ BAD:
```terraform
variable "disable_encryption" { type = bool }
```

✅ GOOD:
```terraform
variable "encryption_enabled" { type = bool }
```

### 2.3 Descriptions

**Every** `variable` and `output` block **must** include a `description`.

```terraform
variable "instance_type" {
  description = "The EC2 instance type to use for the server."
  type        = string
  default     = "t3.micro"
}

output "vpc_id" {
  description = "The ID of the created VPC."
  value       = aws_vpc.main.id
}
```

## 3. State Management

Remote state and locking are non-negotiable for team collaboration.

### 3.1 Remote State and Locking

**Always** use a remote backend (e.g., S3 with DynamoDB locking). **Never** rely on local state in shared environments.

```terraform
# backend.tf
terraform {
  backend "s3" {
    bucket         = "my-team-terraform-state"
    key            = "environments/prod/app/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock" # Must be pre-created
    encrypt        = true
  }
}
```

### 3.2 Workspaces

**Avoid** `terraform workspace` for environment isolation. Use separate backend configurations in distinct directories (e.g., `environments/dev`, `environments/prod`) instead. Workspaces are primarily for ephemeral, short-lived deployments or testing variations within a single configuration.

## 4. Modules: Reusability and Abstraction

Modules are the cornerstone of scalable Terraform.

### 4.1 Module Design

Design modules to be single-purpose, composable, and reusable. Abstract implementation details.

❌ BAD: A module that creates a VPC, an RDS instance, and an S3 bucket.
✅ GOOD: Separate modules for `vpc`, `rds_instance`, and `s3_bucket`.

### 4.2 Module Inputs and Outputs

*   **Inputs**: Parameterize only what needs to vary. Use `nullable = false` for variables that should never be `null` (especially lists, maps, objects defaulting to empty).
*   **Outputs**: Provide meaningful outputs with descriptions. **Never** pass outputs directly through input variables; reference resource attributes to ensure implicit dependencies.

```terraform
# modules/vpc/variables.tf
variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
}

variable "tags" {
  description = "A map of tags to assign to all resources in the VPC."
  type        = map(string)
  default     = {}
  nullable    = false # Ensure tags is always a map, even if empty
}

# modules/vpc/outputs.tf
output "vpc_id" {
  description = "The ID of the created VPC."
  value       = aws_vpc.main.id
}
```

## 5. Common Patterns and Anti-patterns

### 5.1 Loops and Conditionals (`for_each`, `count`)

Use `for_each` for creating multiple instances of a resource when you have a map or set of strings, as it provides better state management and readability than `count`. Use `count` for simple numeric iteration or conditional resource creation (when `count` is 0 or 1).

❌ BAD: Using `count` with a map for resource creation.
```terraform
resource "aws_instance" "app_servers" {
  count = length(var.instance_configs) # var.instance_configs is a map
  # ...
  tags = {
    Name = var.instance_configs[count.index].name
  }
}
```

✅ GOOD: Using `for_each` with a map for resource creation.
```terraform
resource "aws_instance" "app_servers" {
  for_each = var.instance_configs # var.instance_configs is a map of objects
  # ...
  tags = {
    Name = each.value.name
  }
}
```

### 5.2 Dynamic Blocks

Use dynamic blocks to construct repeatable nested blocks based on a complex variable.

```terraform
resource "aws_security_group" "web" {
  # ...
  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}
```

### 5.3 Tagging Resources

**Always** tag all resources with meaningful metadata (e.g., `Project`, `Environment`, `Owner`, `CostCenter`). Use a `default_tags` block in your provider configuration.

```terraform
provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Project     = "MyApplication"
      Environment = "prod"
      ManagedBy   = "Terraform"
    }
  }
}
```

## 6. Security and Configuration Management

### 6.1 Secrets Management

**Never** hardcode sensitive data. Fetch secrets dynamically from a dedicated secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault).

❌ BAD:
```terraform
resource "aws_db_instance" "app_db" {
  password = "SuperSecretPassword123!" # NEVER DO THIS
}
```

✅ GOOD:
```terraform
data "aws_secretsmanager_secret_version" "db_creds" {
  secret_id = "prod/my-app/db-credentials"
}

locals {
  db_credentials = jsondecode(data.aws_secretsmanager_secret_version.db_creds.secret_string)
}

resource "aws_db_instance" "app_db" {
  password = local.db_credentials.password
}
```

### 6.2 Policy as Code

Integrate Policy as Code (e.g., Open Policy Agent (OPA) or HashiCorp Sentinel) into your CI/CD pipeline to enforce organizational compliance and security policies before deployment.

## 7. Common Pitfalls and Gotchas

### 7.1 Formatting and Validation

**Always** run `terraform fmt` and `terraform validate` before committing. Integrate these into Git pre-commit hooks and CI/CD pipelines.

### 7.2 Linting

Use a linter like TFLint to catch style violations, potential errors, and enforce best practices early.

### 7.3 Version Pinning

**Always** pin versions for Terraform, providers, and modules to ensure reproducible deployments and prevent unexpected changes.

```terraform
# providers.tf
terraform {
  required_version = "~> 1.5" # Pin Terraform CLI version
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Pin AWS provider version
    }
  }
}

# main.tf (for modules)
module "vpc" {
  source  = "cloudposse/vpc/aws"
  version = "2.0.0" # Pin module version
  # ...
}
```

### 7.4 Lifecycle Management

Use the `lifecycle` block judiciously, especially `prevent_destroy` for critical resources and `create_before_destroy` for zero-downtime updates.

```terraform
resource "aws_s3_bucket" "critical_data" {
  bucket = "my-critical-data-bucket"
  # ...
  lifecycle {
    prevent_destroy = true # Protect critical resources from accidental deletion
  }
}
```

## 8. CI/CD Integration

Automate your Terraform workflow with CI/CD.

### 8.1 Automated Plan Checks

**Every** pull request **must** trigger a `terraform plan` and post the output as a comment or check. This ensures visibility and prevents unexpected changes.

### 8.2 Automated Apply

Implement automated `terraform apply` on merge to `main` (or environment branches) only after successful plan, validation, and policy checks. This requires robust state locking and access controls.