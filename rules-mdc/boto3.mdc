---
description: This rule file guides developers on writing idiomatic, high-performance, and type-safe boto3 code, emphasizing client reuse, proper pagination, batch operations, and robust testing strategies.
globs: **/*.py
---
# boto3 Best Practices

Boto3 is the definitive Python SDK for AWS. Mastering it means writing code that is performant, maintainable, and resilient. This guide outlines the essential patterns and anti-patterns for modern `boto3` development.

## 1. Client/Resource Management & Reuse

Never instantiate `boto3` clients or resources inside performance-critical loops or frequently called functions. Reusing client objects reduces credential churn, improves connection pooling, and significantly boosts performance.

### ✅ GOOD: Module-level or Dependency Injected Clients

Instantiate clients/resources once per module or inject them via a wrapper class.

```python
# my_aws_service.py
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.service_resource import S3ServiceResource

# Module-level client/resource for simple cases
S3_CLIENT: S3Client = boto3.client("s3", region_name="us-east-1")
S3_RESOURCE: S3ServiceResource = boto3.resource("s3", region_name="us-east-1")

class S3Manager:
    """Manages S3 operations with a reusable client."""
    def __init__(self, s3_client: S3Client = S3_CLIENT):
        self._s3_client = s3_client

    def list_my_buckets(self) -> list[str]:
        response = self._s3_client.list_buckets()
        return [b["Name"] for b in response.get("Buckets", [])]

# In another part of your application:
manager = S3Manager()
buckets = manager.list_my_buckets()
```

### ❌ BAD: Repeated Client Instantiation

Avoid creating new clients/resources for every operation.

```python
# my_aws_service_bad.py
import boto3

def get_bucket_names_bad() -> list[str]:
    # This creates a new client every time the function is called
    s3_client = boto3.client("s3", region_name="us-east-1")
    response = s3_client.list_buckets()
    return [b["Name"] for b in response.get("Buckets", [])]

# This is inefficient if called frequently
for _ in range(100):
    get_bucket_names_bad()
```

## 2. Choose Clients vs. Resources Wisely

`boto3` offers two interfaces:
*   **Clients**: Low-level, 1:1 mapping to AWS API calls. Provides fine-grained control.
*   **Resources**: Higher-level, object-oriented abstraction. Offers automatic pagination, attribute access, and simpler syntax for common operations.

### ✅ GOOD: Leverage Resources for Simplicity and Automation

Use resources when available and when their higher-level abstractions simplify your code (e.g., S3, EC2, DynamoDB).

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import Bucket, S3ServiceResource

S3_RESOURCE: S3ServiceResource = boto3.resource("s3")

def delete_all_objects_in_bucket(bucket_name: str) -> None:
    bucket: Bucket = S3_RESOURCE.Bucket(bucket_name)
    # Resource handles pagination automatically for collection iteration
    for obj in bucket.objects.all():
        obj.delete()
    print(f"Deleted all objects from {bucket_name}")

# Example usage:
# delete_all_objects_in_bucket("my-test-bucket")
```

### ❌ BAD: Over-complicating with Clients when Resources are Better

Manually handling tasks that resources automate.

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

S3_CLIENT: S3Client = boto3.client("s3")

def delete_all_objects_in_bucket_bad(bucket_name: str) -> None:
    # Requires manual pagination and object key extraction
    paginator = S3_CLIENT.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        objects_to_delete = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects_to_delete:
            S3_CLIENT.delete_objects(Bucket=bucket_name, Delete={"Objects": objects_to_delete})
    print(f"Deleted all objects from {bucket_name}")
```

## 3. Efficient Pagination

AWS APIs often return results in pages. Always use `boto3`'s built-in paginators or resource collections to iterate through all results.

### ✅ GOOD: Using Paginators or Resource Collections

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_ec2.client import EC2Client
    from mypy_boto3_ec2.service_resource import EC2ServiceResource, Instance

EC2_CLIENT: EC2Client = boto3.client("ec2")
EC2_RESOURCE: EC2ServiceResource = boto3.resource("ec2")

def list_all_instance_ids_client() -> list[str]:
    paginator = EC2_CLIENT.get_paginator("describe_instances")
    instance_ids = []
    for page in paginator.paginate():
        for reservation in page.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_ids.append(instance["InstanceId"])
    return instance_ids

def list_all_instance_ids_resource() -> list[str]:
    # Resource collections handle pagination automatically
    instance_ids = [instance.id for instance in EC2_RESOURCE.instances.all()]
    return instance_ids

# Example usage:
# print(list_all_instance_ids_client())
# print(list_all_instance_ids_resource())
```

### ❌ BAD: Manual Pagination Loops

This is error-prone and verbose.

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_ec2.client import EC2Client

EC2_CLIENT: EC2Client = boto3.client("ec2")

def list_all_instance_ids_manual_bad() -> list[str]:
    instance_ids = []
    next_token = None
    while True:
        kwargs = {"NextToken": next_token} if next_token else {}
        response = EC2_CLIENT.describe_instances(**kwargs)
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_ids.append(instance["InstanceId"])
        next_token = response.get("NextToken")
        if not next_token:
            break
    return instance_ids
```

## 4. Batch Operations

Always prefer batch operations when available to minimize API calls and improve efficiency.

### ✅ GOOD: Using Batch APIs

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_sqs.client import SQSClient

SQS_CLIENT: SQSClient = boto3.client("sqs")

def send_batch_messages(queue_url: str, messages: list[str]) -> None:
    entries = [{"Id": str(i), "MessageBody": msg} for i, msg in enumerate(messages)]
    # SQS allows up to 10 messages per batch
    for i in range(0, len(entries), 10):
        batch = entries[i:i+10]
        SQS_CLIENT.send_message_batch(QueueUrl=queue_url, Entries=batch)
    print(f"Sent {len(messages)} messages to {queue_url}")

# Example usage:
# send_batch_messages("https://sqs.us-east-1.amazonaws.com/123456789012/my-queue", ["msg1", "msg2", "msg3"])
```

### ❌ BAD: Looping Single Operations

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_sqs.client import SQSClient

SQS_CLIENT: SQSClient = boto3.client("sqs")

def send_individual_messages_bad(queue_url: str, messages: list[str]) -> None:
    for msg in messages:
        # Each message sends a separate API call
        SQS_CLIENT.send_message(QueueUrl=queue_url, MessageBody=msg)
    print(f"Sent {len(messages)} messages to {queue_url}")
```

## 5. Type Hinting with `boto3-stubs`

`boto3`'s dynamic nature makes static analysis challenging. Use `boto3-stubs` to enable precise type checking, catch errors early, and improve IDE autocomplete.

### ✅ GOOD: Fully Type-Hinted Boto3 Code

Install `boto3-stubs` (e.g., `pip install 'boto3-stubs[s3]' mypy`).

```python
from typing import TYPE_CHECKING
import boto3

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import BucketTypeDef, ListBucketsOutputTypeDef

S3_CLIENT: S3Client = boto3.client("s3")

def get_s3_bucket_names() -> list[str]:
    response: ListBucketsOutputTypeDef = S3_CLIENT.list_buckets()
    buckets: list[BucketTypeDef] = response.get("Buckets", [])
    return [b["Name"] for b in buckets]

# Mypy will now correctly validate types and provide completions.
```

### ❌ BAD: Untyped Boto3 Code

This makes code harder to read, refactor, and debug.

```python
import boto3

s3_client = boto3.client("s3")

def get_s3_bucket_names_untyped():
    response = s3_client.list_buckets()
    # No type information here, relies on runtime inspection
    return [b["Name"] for b in response.get("Buckets", [])]
```

## 6. Robust Error Handling

Boto3 operations can fail. Always wrap calls in `try...except` blocks and catch specific `botocore.exceptions` for graceful degradation.

### ✅ GOOD: Specific Exception Handling

```python
import boto3
from botocore.exceptions import ClientError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

S3_CLIENT: S3Client = boto3.client("s3")

def download_s3_object(bucket_name: str, key: str, local_path: str) -> bool:
    try:
        S3_CLIENT.download_file(bucket_name, key, local_path)
        print(f"Successfully downloaded {key} to {local_path}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Object {key} not found in bucket {bucket_name}.")
        elif e.response["Error"]["Code"] == "403":
            print(f"Permission denied to access {key} in bucket {bucket_name}.")
        else:
            print(f"An unexpected error occurred: {e}")
        return False
    except Exception as e:
        print(f"A general error occurred: {e}")
        return False

# Example usage:
# download_s3_object("my-test-bucket", "non-existent-file.txt", "/tmp/local.txt")
```

### ❌ BAD: Generic Exception Handling or No Handling

```python
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

S3_CLIENT: S3Client = boto3.client("s3")

def download_s3_object_bad(bucket_name: str, key: str, local_path: str) -> None:
    try:
        S3_CLIENT.download_file(bucket_name, key, local_path)
    except Exception as e: # Too broad, hides specific issues
        print(f"An error occurred: {e}")
    # Or worse, no try/except at all.
```

## 7. Testing with `moto`

For unit and integration tests, `moto` provides a mock AWS environment, allowing you to test `boto3` interactions without hitting actual AWS services.

### ✅ GOOD: Using `moto` Decorators

```python
import boto3
import unittest
from moto import mock_aws

class TestS3Operations(unittest.TestCase):
    @mock_aws
    def test_create_and_list_bucket(self):
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "my-test-bucket"

        # Create a bucket
        s3_client.create_bucket(Bucket=bucket_name)

        # List buckets and assert
        response = s3_client.list_buckets()
        self.assertIn(bucket_name, [b["Name"] for b in response["Buckets"]])

        # Upload an object
        s3_client.put_object(Bucket=bucket_name, Key="test.txt", Body="Hello Moto!")

        # Download and verify
        response = s3_client.get_object(Bucket=bucket_name, Key="test.txt")
        self.assertEqual(response["Body"].read().decode("utf-8"), "Hello Moto!")

if __name__ == "__main__":
    unittest.main()
```

### ❌ BAD: Mocking `boto3` Internals Manually

While `unittest.mock` is powerful, mocking `boto3` at a low level is tedious and fragile.

```python
import boto3
import unittest
from unittest.mock import patch

class TestS3OperationsManualMock(unittest.TestCase):
    @patch("boto3.client")
    def test_create_bucket_manual_mock(self, mock_boto_client):
        # Configure the mock client to return specific values
        mock_s3_client = mock_boto_client.return_value
        mock_s3_client.list_buckets.return_value = {"Buckets": []}
        mock_s3_client.create_bucket.return_value = {} # Simulate success

        s3_client = boto3.client("s3") # This will now return our mock
        s3_client.create_bucket(Bucket="my-test-bucket")

        mock_s3_client.create_bucket.assert_called_once_with(Bucket="my-test-bucket")
        # This approach quickly becomes complex for more involved interactions.
```