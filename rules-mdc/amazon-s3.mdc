---
description: This guide provides definitive best practices for interacting with Amazon S3, focusing on security, performance, cost optimization, and maintainable code in modern AWS applications.
globs: **/*
---
# amazon-s3 Best Practices

Amazon S3 is the backbone of object storage in AWS. Adhering to these guidelines ensures your S3 implementations are secure, performant, cost-effective, and maintainable.

## 1. Security First

Security is paramount. Always assume your S3 buckets contain sensitive data and configure them defensively.

### 1.1. Encryption at Rest

Always encrypt objects at rest. Default to AWS-managed keys (SSE-S3) or KMS (SSE-KMS) for stronger control.

❌ **BAD: Unencrypted objects or relying on client-side encryption without server-side enforcement.**
```python
import boto3

s3 = boto3.client('s3')
s3.put_object(Bucket='my-unencrypted-bucket', Key='data.txt', Body=b'sensitive data')
# Data is stored unencrypted by default if bucket policy doesn't enforce it.
```

✅ **GOOD: Enforce server-side encryption (SSE-S3 or SSE-KMS).**
```python
import boto3

s3 = boto3.client('s3')
# Option 1: SSE-S3 (AWS-managed keys)
s3.put_object(
    Bucket='my-secure-bucket',
    Key='data.txt',
    Body=b'sensitive data',
    ServerSideEncryption='AES256' # SSE-S3
)

# Option 2: SSE-KMS (Customer Master Key)
# Ensure 'my-kms-key-id' exists and bucket has permissions.
s3.put_object(
    Bucket='my-secure-bucket',
    Key='more-data.txt',
    Body=b'more sensitive data',
    ServerSideEncryption='aws:kms',
    SSEKMSKeyId='arn:aws:kms:us-east-1:123456789012:key/my-kms-key-id'
)
```
> **Note on SSE-C:** As of April 2026, SSE-C is disabled for new buckets by default. Only enable it via `PutBucketEncryption` API *after* bucket creation if explicitly required for legacy or specific compliance needs. Prefer SSE-S3 or SSE-KMS.

### 1.2. Access Control: Policies over ACLs

Manage access exclusively through IAM and Bucket Policies. Disable ACLs for simplified, auditable permissions.

❌ **BAD: Relying on ACLs for granular object access.**
```python
# Avoid this pattern. ACLs are disabled by default for new buckets.
s3.put_object_acl(
    Bucket='my-bucket',
    Key='object.txt',
    GrantRead='id="some-aws-account-id"'
)
```

✅ **GOOD: Disable ACLs and use IAM/Bucket Policies.**
```python
# Ensure S3 Object Ownership is set to "Bucket owner enforced" to disable ACLs.
# This is the default for new buckets.
# Example Bucket Policy (IaC recommended):
# This policy grants read-only access to a specific IAM role for a specific prefix.
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789012:role/MyReadOnlyRole"
            },
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::my-secure-bucket/data/*"]
        }
    ]
}
# Apply this via IaC (e.g., Terraform, CloudFormation) or boto3.put_bucket_policy
```

### 1.3. Block Public Access

Always enable S3 Block Public Access at the account or organization level. Buckets should never be public unless explicitly required for static website hosting (and even then, use CloudFront with OAC).

✅ **GOOD: Enable S3 Block Public Access.**
```python
# This is typically configured via AWS Console, AWS CLI, or IaC at the account level.
# Example for a specific bucket (via boto3, but IaC is preferred):
s3.put_public_access_block(
    Bucket='my-secure-bucket',
    PublicAccessBlockConfiguration={
        'BlockPublicAcls': True,
        'IgnorePublicAcls': True,
        'BlockPublicPolicy': True,
        'RestrictPublicBuckets': True
    }
)
```

## 2. Code Organization & Data Governance

Structured naming, versioning, and lifecycle rules are crucial for manageability and cost control.

### 2.1. Bucket & Object Naming

Adopt clear, hierarchical naming conventions, especially for data lakes. This simplifies governance and policy application.

❌ **BAD: Generic or flat naming.**
```
my-app-bucket/customerdata_20230101.csv
my-app-bucket/logs_server1_access.log
```

✅ **GOOD: Hierarchical, environment-aware naming.**
```
# Bucket name (globally unique, often includes org/env/purpose)
# e.g., myorg-prod-datalake-raw-us-east-1

# Object keys (within the bucket)
# For data lakes:
# <environment>/<layer>/<source>/<dataset>/<year>/<month>/<day>/<file>
# e.g., prod/raw/ecommerce/orders/2025/01/15/order_batch_1.json
# e.g., dev/curated/analytics/user_sessions/2025/01/15/session_summary.parquet

# For application assets:
# <environment>/<application>/<asset_type>/<id>/<file>
# e.g., prod/webapp/images/user-123/profile.jpg
# e.g., dev/backend/configs/service-a/config.yaml
```

### 2.2. Versioning & Tagging

Enable versioning on all critical buckets. Use object tags for cost allocation, lifecycle management, and audit trails.

❌ **BAD: No versioning, untagged objects.**
```python
# Accidental deletion or overwrite means data loss.
s3.delete_object(Bucket='my-bucket', Key='important-report.csv')
# Oops, gone forever.
```

✅ **GOOD: Enable versioning and tag objects.**
```python
# Terraform example for bucket versioning:
# resource "aws_s3_bucket" "my_bucket" {
#   bucket = "myorg-prod-data"
#   versioning {
#     enabled = true
#   }
#   # ... other configurations
# }

# Python example for tagging an object:
s3.put_object(
    Bucket='myorg-prod-data',
    Key='reports/monthly/sales-2025-01.csv',
    Body=b'sales data',
    Tagging='department=finance&project=sales-reporting'
)
```

### 2.3. Lifecycle Management

Automate object transitions to cheaper storage classes (e.g., Glacier, Intelligent-Tiering) or deletion based on access patterns. Use S3 Inventory to gain insights.

✅ **GOOD: Define lifecycle rules.**
```python
# Terraform example for a lifecycle rule:
# resource "aws_s3_bucket_lifecycle_configuration" "my_bucket_lifecycle" {
#   bucket = aws_s3_bucket.my_bucket.id
#
#   rule {
#     id     = "archive_old_logs"
#     status = "Enabled"
#
#     filter {
#       prefix = "logs/"
#     }
#
#     transition {
#       days          = 30
#       storage_class = "STANDARD_IA" # Infrequent Access
#     }
#     transition {
#       days          = 90
#       storage_class = "GLACIER_FLEXIBLE_RETRIEVAL"
#     }
#     expiration {
#       days = 365
#     }
#   }
#   # ... other rules
# }
```

## 3. Performance Optimization

Design your S3 interactions for maximum throughput and minimal latency.

### 3.1. Multipart Uploads

For files larger than 100MB, always use multipart uploads. This improves upload speed, resilience, and allows for parallelization.

❌ **BAD: Uploading large files in a single PUT operation.**
```python
# This can be slow and prone to failure for large files.
s3.put_object(Bucket='my-bucket', Key='large-video.mp4', Body=open('large-video.mp4', 'rb'))
```

✅ **GOOD: Use `transfer_manager` or `upload_fileobj` for multipart uploads.**
```python
import boto3
from boto3.s3.transfer import TransferConfig

s3 = boto3.client('s3')
config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10, multipart_chunksize=1024 * 25) # 25MB parts

with open('large-video.mp4', 'rb') as data:
    s3.upload_fileobj(data, 'my-bucket', 'large-video.mp4', Config=config)
```

### 3.2. Request Rate & Prefixes

Distribute high-throughput workloads across multiple S3 prefixes. S3 scales performance based on prefixes. Aim for at least 3-4 x 10^4 requests per second per prefix.

❌ **BAD: Concentrating all high-volume reads/writes on a single prefix.**
```
my-bucket/logs/all_access_logs_YYYY-MM-DD.log # Single hot prefix
```

✅ **GOOD: Spread workloads across multiple prefixes.**
```
my-bucket/logs/webserver1/access_logs_YYYY-MM-DD.log
my-bucket/logs/webserver2/access_logs_YYYY-MM-DD.log
my-bucket/logs/appserver1/access_logs_YYYY-MM-DD.log
# Or use hash-based prefixes for even distribution:
# my-bucket/data/hash_prefix_A/object_id.json
# my-bucket/data/hash_prefix_B/object_id.json
```

## 4. Infrastructure as Code (IaC) with Terraform

When using S3 as a Terraform backend, follow these critical best practices.

❌ **BAD: Using local state or deprecated DynamoDB locking.**
```terraform
# backend "s3" {
#   bucket = "myorg-tf-states"
#   key    = "myapp/prod.tfstate"
#   region = "us-east-1"
#   # Missing versioning, encryption, and using deprecated DynamoDB for locking
#   dynamodb_table = "terraform-locks" # DEPRECATED!
# }
```

✅ **GOOD: Enable versioning, encryption, and S3 native locking.**
```terraform
terraform {
  backend "s3" {
    bucket         = "myorg-tf-states"
    key            = "myapp/prod.tfstate"
    region         = "us-east-1"
    encrypt        = true             # Always encrypt state file
    versioning_enabled = true         # Critical for state recovery
    use_lockfile   = true             # RECOMMENDED: S3 native locking (Terraform 1.10+)
  }
}
```

## 5. Testing Approaches

Thoroughly test your S3 interactions, especially security and data integrity.

### 5.1. Local Integration Testing

Use tools like LocalStack to simulate S3 locally, allowing for rapid integration testing without incurring AWS costs or affecting production resources.

✅ **GOOD: Use LocalStack for local S3 emulation.**
```python
# Example with boto3 and LocalStack (ensure LocalStack is running)
s3_local = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)
s3_local.create_bucket(Bucket='test-bucket')
s3_local.put_object(Bucket='test-bucket', Key='test-object.txt', Body=b'hello')
```

### 5.2. Policy Validation

Programmatically validate IAM and bucket policies to ensure least privilege and prevent unintended access.

✅ **GOOD: Use AWS IAM Policy Simulator or `boto3` for validation.**
```python
# Example: Check if an IAM role can perform a specific S3 action
iam_client = boto3.client('iam')
response = iam_client.simulate_principal_policy(
    PolicySourceArn='arn:aws:iam::123456789012:role/MyReadOnlyRole',
    ActionNames=['s3:GetObject'],
    ResourceArns=['arn:aws:s3:::my-secure-bucket/data/report.csv']
)
# Assert that 'Effect' is 'Allow' for the action.
```