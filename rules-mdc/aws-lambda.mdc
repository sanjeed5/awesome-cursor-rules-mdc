---
description: Definitive guidelines for writing secure, performant, and maintainable AWS Lambda functions using modern best practices.
globs: **/*
---
# aws-lambda Best Practices

This guide outlines the definitive best practices for developing AWS Lambda functions. Adhere to these principles to build reliable, cost-effective, and secure serverless applications.

## Code Organization and Structure

**1. Single Responsibility Principle:** Each Lambda function must perform one distinct task. Decompose complex workflows into smaller, focused functions orchestrated by services like AWS Step Functions.

**2. Initialize Outside the Handler:** Leverage execution environment reuse by initializing SDK clients, database connections, and heavy dependencies in the global scope.

❌ BAD:
```python
import boto3

def handler(event, context):
    s3 = boto3.client('s3') # Initialized on every invocation
    # ...
```

✅ GOOD:
```python
import boto3

s3 = boto3.client('s3') # Initialized once per execution environment

def handler(event, context):
    # s3 client is reused across invocations
    # ...
```

**3. Use Lambda Layers for Shared Code:** Keep deployment packages small and promote code reuse for common libraries and dependencies.

**4. Configuration via Environment Variables:** Never hardcode operational parameters. Use environment variables for dynamic configuration. For sensitive data, use AWS Secrets Manager or AWS Systems Manager Parameter Store.

**5. Structured JSON Logging:** Output logs in JSON format to CloudWatch. This makes logs easily queryable and analyzable. Use `aws-lambda-powertools` Logger utility.

❌ BAD:
```python
print(f"Processing event: {event}")
```

✅ GOOD:
```python
from aws_lambda_powertools import Logger
logger = Logger()

@logger.inject_lambda_context
def handler(event, context):
    logger.info("Processing event", event=event)
    # ...
```

**6. Infrastructure as Code (AWS CDK v2):** Define your Lambda functions and their surrounding infrastructure (API Gateway, DynamoDB, IAM roles) using AWS CDK v2. This ensures version control, repeatability, and safe deployments.

✅ GOOD:
```python
# Example using AWS CDK v2 (Python)
from aws_cdk import Stack, aws_lambda as lambda_
from constructs import Construct

class MyLambdaStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        lambda_.Function(self, "MyHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda"), # 'lambda' directory contains index.py
            environment={
                "TABLE_NAME": "MyTable",
            },
            # Add more configurations like memory, timeout, IAM roles
        )
```

## Common Patterns and Anti-patterns

**1. Implement Idempotency:** Design functions to produce the same result even if invoked multiple times with the same input. Use `aws-lambda-powertools` for robust idempotency.

❌ BAD:
```python
# No idempotency check, might process duplicate events
def handler(event, context):
    order_id = event['detail']['orderId']
    # Process order, could be duplicated
    # ...
```

✅ GOOD:
```python
from aws_lambda_powertools.utilities.idempotency import idempotent
from aws_lambda_powertools.utilities.idempotency.persistence import DynamoDBPersistenceLayer
import os

persistence_layer = DynamoDBPersistenceLayer(table_name=os.environ["IDEMPOTENCY_TABLE"])

@idempotent(persistence_layer=persistence_layer)
def handler(event, context):
    order_id = event['detail']['orderId']
    # Process order, guaranteed to run once
    # ...
```

**2. Stateless Functions:** Functions must not rely on mutable local state between invocations. Any mutable data belongs in external, durable storage (e.g., DynamoDB, S3).

**3. Orchestrate with Step Functions:** For complex, multi-step workflows, use AWS Step Functions to manage state and coordination. Keep individual Lambda functions simple.

## Performance Considerations

**1. Optimize Memory and CPU:** Memory allocation directly impacts CPU. Use the AWS Lambda Power Tuning tool to find the optimal memory configuration for your function.

**2. Keep Deployment Packages Small:** Minimize cold start times by reducing package size. Use Lambda Layers and only include necessary dependencies.

**3. Use Keep-Alive for Persistent Connections:** Prevent idle connections from being purged, especially for HTTP clients.

✅ GOOD:
```python
import http.client
import ssl

# Global scope for connection reuse
http_client = http.client.HTTPSConnection("example.com", context=ssl._create_unverified_context())

def handler(event, context):
    http_client.request("GET", "/api/data")
    response = http_client.getresponse()
    # ...
```

## Common Pitfalls and Gotchas

**1. Over-privileged IAM Roles:** Apply the principle of least privilege. Grant only the minimum necessary permissions to your function's execution role.

❌ BAD:
```json
{
  "Effect": "Allow",
  "Action": "s3:*", // Too broad
  "Resource": "*"
}
```

✅ GOOD:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject"
  ],
  "Resource": "arn:aws:s3:::my-specific-bucket/*" // Specific bucket and actions
}
```

**2. Lack of Input Validation:** Always validate and sanitize all incoming event data to prevent injection attacks and unexpected behavior. Use libraries like Pydantic or `aws-lambda-powertools` Parser utility.

**3. Recursive Invocations:** Avoid functions calling themselves directly or indirectly, as this can lead to infinite loops and escalated costs.

**4. Unhandled Errors:** Ensure all potential errors are caught and handled gracefully. Throw explicit exceptions for unrecoverable issues. Use `async/await` for asynchronous operations.

## Testing Approaches

**1. Unit Testing:** Focus on testing the core business logic of your function in isolation, mocking AWS service interactions.

**2. Integration Testing:** Test your function's interactions with actual AWS services. Use ephemeral resources provisioned by CDK.

**3. End-to-End Testing:** Simulate real-world scenarios, testing the entire flow from event source to final outcome.

**4. Local Development and Testing:** Utilize tools like AWS SAM CLI or LocalStack to test functions locally before deployment.