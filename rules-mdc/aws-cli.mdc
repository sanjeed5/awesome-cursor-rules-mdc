---
description: Definitive guidelines for secure, reproducible, and efficient use of the AWS CLI, emphasizing modern DevOps practices and automation.
globs: **/*
---
# aws-cli Best Practices

The AWS CLI is your primary interface for programmatic interaction with AWS. Adhering to these best practices ensures your CLI usage is secure, reproducible, and efficient, aligning with modern DevOps principles.

## 1. Code Organization and Structure

Organize your AWS CLI interactions into well-defined scripts or functions. Avoid one-off commands for anything beyond simple queries.

### 1.1. Use Named Profiles for Environment Segregation
Always define multiple named profiles for different environments (e.g., `dev`, `test`, `prod`) and projects. This prevents accidental operations in the wrong account.

❌ BAD: Relying on `default` profile or constantly switching credentials.
```bash
# Easy to accidentally target production
aws s3 rm s3://my-prod-bucket --recursive
```

✅ GOOD: Explicitly using named profiles.
```bash
# ~/.aws/config
# [profile dev]
# region = us-east-1
# output = json

# [profile prod]
# region = us-east-1
# output = json

# In your script or terminal
aws s3 ls --profile dev
aws s3 rm s3://my-dev-bucket --recursive --profile dev
```

### 1.2. Pin AWS CLI Version in CI/CD
Ensure consistent behavior by pinning the AWS CLI version in automated environments. Use `awscli-v2` for modern features and stability.

❌ BAD: Relying on the latest available version in CI/CD.
```yaml
# .github/workflows/deploy.yml
# ...
# - name: Install AWS CLI
#   run: pip install awscli # Installs latest, potentially breaking changes
```

✅ GOOD: Using a specific, tested version (e.g., via Docker).
```yaml
# .github/workflows/deploy.yml
# ...
- name: Deploy with AWS CLI v2
  uses: docker://amazon/aws-cli:2.15.30 # Pin to a specific, validated version
  with:
    args: s3 sync ./app s3://my-bucket-name --delete --profile prod
```

## 2. Common Patterns and Anti-patterns

Adopt patterns that promote automation, idempotency, and machine-readability.

### 2.1. Prefer JSON Output and `jq` for Parsing
Always output in `json` format and use `jq` for robust, idempotent parsing. Avoid parsing human-readable formats like `text` or `table`.

❌ BAD: Parsing `text` output is brittle and prone to breakage from CLI updates.
```bash
# Output format can change, breaking script
aws ec2 describe-instances --output text | grep "running" | awk '{print $NF}'
```

✅ GOOD: Using `json` and `jq` for reliable, structured parsing.
```bash
# Get running instance IDs
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].InstanceId' --output json | jq -r '.[] | .[]'
```

### 2.2. Embed Validation Steps
Include explicit validation steps in your scripts to catch misconfigurations or permission issues early, preventing costly failures.

❌ BAD: Assuming credentials are correct and permissions are sufficient.
```bash
# Script proceeds even if credentials are bad or lack permissions
aws s3 cp local.txt s3://my-bucket/
```

✅ GOOD: Validating caller identity before critical operations.
```bash
if ! aws sts get-caller-identity --profile prod > /dev/null 2>&1; then
  echo "ERROR: Failed to authenticate with 'prod' profile. Check credentials and IAM role."
  exit 1
fi
aws s3 cp local.txt s3://my-bucket/ --profile prod
```

## 3. Performance Considerations

Optimize CLI commands for speed and resource usage, especially when dealing with large datasets.

### 3.1. Filter at the Source with `--query` and `--filters`
Reduce network traffic and processing by filtering results directly in the CLI command rather than fetching all data and post-processing with `jq`.

❌ BAD: Fetching all data then filtering locally.
```bash
aws ec2 describe-instances --output json | jq -r '.Reservations[].Instances[] | select(.State.Name=="running") | .InstanceId'
```

✅ GOOD: Filtering on the AWS side using `--filters` and `--query`.
```bash
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].InstanceId' --output json | jq -r '.[] | .[]'
```

### 3.2. Leverage Pagination and Batch Operations
For commands that return many results, use `--page-size` and `--max-items` for controlled pagination. For write operations, use batch commands when available to reduce API calls.

✅ GOOD: Efficiently listing many S3 objects.
```bash
# List 1000 objects at a time, then continue, up to 5000 total
aws s3api list-objects-v2 --bucket my-large-bucket --page-size 1000 --max-items 5000 --output json
```

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that lead to security vulnerabilities or operational failures.

### 4.1. Never Hardcode Credentials
Store credentials securely outside of scripts and version control. Hardcoding is the weakest link in cloud security.

❌ BAD: Hardcoding `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in scripts.
```bash
export AWS_ACCESS_KEY_ID="AKIA..." # NEVER DO THIS
export AWS_SECRET_ACCESS_KEY="wJalr..." # NEVER DO THIS
aws s3 ls
```

✅ GOOD: Using named profiles, IAM roles, or temporary credentials.
```bash
# Credentials managed in ~/.aws/credentials or assumed via IAM role
aws s3 ls --profile dev
```

### 4.2. Validate Configuration Files
Regularly validate your `~/.aws/config` and `~/.aws/credentials` files, especially after manual edits or when troubleshooting.

✅ GOOD: Using the built-in validation.
```bash
aws configure list --profile dev # Verifies profile exists and lists its configuration
```

## 5. Configuration Management

Centralize and secure your AWS CLI configuration.

### 5.1. Use `~/.aws/config` and `~/.aws/credentials`
These are the standard, secure locations for CLI configuration and credentials. Keep them separate for clarity and security.

```ini
# ~/.aws/config (for profile-specific settings like region, output, role assumption)
[profile dev]
region = us-east-1
output = json
role_arn = arn:aws:iam::123456789012:role/DevRole
source_profile = default

# ~/.aws/credentials (for actual access keys, typically for base profile)
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = wJalr...
```

### 5.2. Store Secrets in AWS Secrets Manager or Parameter Store
For sensitive values (API keys, database passwords) used by CLI scripts, retrieve them dynamically at runtime. Never embed them in scripts or config files directly.

✅ GOOD: Fetching secrets at runtime.
```bash
DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id my-db-secret --query SecretString --output text --profile dev)
echo "Connecting with password: $DB_PASSWORD"
```

## 6. Environment Variables

Leverage environment variables for dynamic configuration without modifying scripts, enabling portable automation.

### 6.1. Explicitly Set `AWS_PROFILE` and `AWS_REGION`
Control which profile and region your commands target without requiring `--profile` and `--region` flags on every command. This is crucial for CI/CD.

❌ BAD: Forgetting `--profile` or `--region` and relying on defaults.
```bash
aws s3 ls # Might use default profile or wrong region
```

✅ GOOD: Setting environment variables for script context.
```bash
# In a script or shell session
export AWS_PROFILE=prod
export AWS_REGION=us-west-2
aws s3 ls # Will use 'prod' profile in 'us-west-2'
```

## 7. Logging

Implement robust logging for auditability, debugging, and compliance.

### 7.1. Log CLI Commands and Outputs
Capture command invocations and their responses, especially in automated scripts. This provides a clear audit trail.

✅ GOOD: Logging to a file or standard output.
```bash
LOG_FILE="aws_script_$(date +%Y%m%d_%H%M%S).log"
echo "Executing: aws s3 ls --profile dev" | tee -a "$LOG_FILE"
aws s3 ls --profile dev | tee -a "$LOG_FILE"
```

### 7.2. Integrate with CloudWatch Logs for CI/CD
In automated pipelines, ensure CLI output is captured and sent to CloudWatch Logs for centralized monitoring, alerting, and long-term retention.

## 8. Testing Approaches

Ensure the reliability and correctness of your AWS CLI scripts through systematic testing.

### 8.1. Local Testing with Mocks
For complex scripts, use tools like `moto` (for Python-based AWS interactions) to mock AWS services locally. This enables faster unit testing without incurring AWS costs or hitting actual endpoints.

### 8.2. CI/CD Integration Tests
Run your CLI scripts as part of your CI/CD pipeline against dedicated, isolated non-production AWS accounts. This validates end-to-end functionality and permissions.

✅ GOOD: Automated pipeline step.
```yaml
# .github/workflows/test.yml
# ...
- name: Run AWS CLI integration tests
  run: |
    ./scripts/validate_s3_bucket.sh --profile test
    ./scripts/check_lambda_status.sh --profile test
  env:
    AWS_PROFILE: test # Ensure test profile is used for integration tests
```

### 8.3. Policy-as-Code Validation
Before deployment, use tools like AWS Config conformance packs or Open Policy Agent (OPA) to validate that IAM policies and resource configurations (often managed via IaC and CLI) adhere to security and operational best practices.