---
description: This guide provides opinionated, actionable best practices for developing and deploying applications on AWS, emphasizing CDK v2, serverless patterns, and Well-Architected principles.
globs: **/*
---
# AWS Best Practices

This document outlines our definitive AWS development standards, rooted in the AWS Well-Architected Framework and modern DevOps guidance. We mandate CDK v2 for infrastructure-as-code and prioritize serverless patterns. Adhere to these guidelines to build secure, reliable, performant, and cost-optimized AWS applications.

## 1. Code Organization and Structure (AWS CDK v2)

Always structure your AWS CDK applications for clarity, reusability, and maintainability.

### 1.1 Use AWS CDK v2 Exclusively

CDK v1 is deprecated. All new development and migrations **must** use CDK v2.

❌ **BAD: Using CDK v1**
```typescript
// cdk.json
{
  "app": "npx ts-node bin/my-app.ts",
  "context": {
    "@aws-cdk/core:enableStackNameDuplicates": "true",
    // ... other v1 context
  }
}
```

✅ **GOOD: Using CDK v2**
```typescript
// cdk.json
{
  "app": "npx ts-node --prefer-ts-exts bin/my-app.ts",
  "context": {
    "@aws-cdk/aws-ecr:disableCrossAccountPush": true,
    "@aws-cdk/aws-iam:minimizePolicies": true,
    "@aws-cdk/core:stackRelativeExports": true,
    "@aws-cdk/aws-lambda:recognizeVersionProps": true,
    // ... other v2 context
  }
}
```

### 1.2 Modularize with Constructs

Encapsulate related resources, runtime code, and configuration into reusable CDK Constructs. This promotes componentization and reduces boilerplate.

```typescript
// lib/api-construct.ts
import { Construct } from 'aws-cdk-lib';
import { RestApi, LambdaIntegration } from 'aws-cdk-lib/aws-apigateway';
import { NodejsFunction } from 'aws-cdk-lib/aws-lambda-nodejs';
import { Runtime } from 'aws-cdk-lib/aws-lambda';
import { Table } from 'aws-cdk-lib/aws-dynamodb';

interface ApiConstructProps {
  tableName: string;
  table: Table;
}

export class ApiConstruct extends Construct {
  constructor(scope: Construct, id: string, props: ApiConstructProps) {
    super(scope, id);

    const handler = new NodejsFunction(this, 'MyApiHandler', {
      runtime: Runtime.NODEJS_20_X,
      entry: 'src/lambda/api-handler.ts',
      environment: {
        TABLE_NAME: props.tableName,
      },
    });

    props.table.grantReadWriteData(handler);

    const api = new RestApi(this, 'MyApi', {
      restApiName: 'MyServiceApi',
    });

    api.root.addMethod('GET', new LambdaIntegration(handler));
  }
}
```

### 1.3 Design Logical Stacks

Organize your application into logical stacks (e.g., `DataStack`, `ApiStack`, `MonitoringStack`). Each stack should represent a deployable unit.

```typescript
// bin/my-app.ts
import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DataStack } from '../lib/data-stack';
import { ApiStack } from '../lib/api-stack';

const app = new App();

const dataStack = new DataStack(app, 'MyServiceDataStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

new ApiStack(app, 'MyServiceApiStack', {
  table: dataStack.table, // Pass dependencies explicitly
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

app.synth();
```

## 2. Common Patterns and Anti-patterns

Adopt proven patterns and rigorously avoid anti-patterns.

### 2.1 Lambda Execution Environment Reuse

Initialize SDK clients and reusable resources (e.g., database connections) outside the Lambda handler. This significantly improves performance and reduces cost.

❌ **BAD: Initializing resources inside the handler**
```typescript
// src/lambda/api-handler.ts
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';

export const handler = async (event: any) => {
  const client = new DynamoDBClient({}); // New client on every invocation
  // ... use client
  return { statusCode: 200, body: 'OK' };
};
```

✅ **GOOD: Initializing resources outside the handler**
```typescript
// src/lambda/api-handler.ts
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';

const client = new DynamoDBClient({}); // Initialized once per execution environment

export const handler = async (event: any) => {
  // ... use client
  return { statusCode: 200, body: 'OK' };
};
```

### 2.2 Use Environment Variables for Configuration

Never hardcode secrets or environment-specific configurations. Use Lambda environment variables or AWS Systems Manager Parameter Store.

❌ **BAD: Hardcoding bucket name**
```typescript
const S3_BUCKET = 'my-hardcoded-bucket-name-prod'; // Don't do this!
```

✅ **GOOD: Using environment variables**
```typescript
// Lambda code
const S3_BUCKET = process.env.S3_BUCKET_NAME;

// CDK definition
new NodejsFunction(this, 'MyFunction', {
  // ...
  environment: {
    S3_BUCKET_NAME: 'my-dynamic-bucket-name', // Set via CDK
  },
});
```

### 2.3 Write Idempotent Lambda Functions

Design your Lambda functions to produce the same result even if invoked multiple times with the same input. This is critical for reliable distributed systems. Use AWS Powertools for Idempotency.

```typescript
// Example using Powertools for Lambda (TypeScript)
import { DynamoDBPersistenceLayer } from '@aws-lambda-powertools/idempotency/lib/persistence';
import { idempotent } from '@aws-lambda-powertools/idempotency';

const persistenceLayer = new DynamoDBPersistenceLayer({
  tableName: process.env.IDEMPOTENCY_TABLE_NAME!,
});

export const handler = idempotent(
  async (event: any) => {
    // Your business logic here
    console.log('Processing event:', event.detail.id);
    // Simulate some work
    await new Promise(resolve => setTimeout(resolve, 100));
    return { status: 'processed', id: event.detail.id };
  },
  { persistenceStore: persistenceLayer }
);
```

## 3. Performance Considerations

Optimize your AWS resources for efficiency and cost.

### 3.1 Tune Lambda Memory and CPU

Memory allocation directly impacts CPU and network performance. Use CloudWatch metrics (`Max Memory Used`) and tools like AWS Lambda Power Tuning to find the optimal memory configuration.

❌ **BAD: Defaulting to 128MB for CPU-intensive tasks**
```typescript
new NodejsFunction(this, 'CpuIntensiveFunction', {
  runtime: Runtime.NODEJS_20_X,
  memorySize: 128, // Likely under-provisioned
  // ...
});
```

✅ **GOOD: Performance testing and tuning memory**
```typescript
new NodejsFunction(this, 'CpuIntensiveFunction', {
  runtime: Runtime.NODEJS_20_X,
  memorySize: 1024, // Tuned based on performance testing
  // ...
});
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes that lead to security vulnerabilities, operational issues, or unexpected costs.

### 4.1 Enforce Least Privilege IAM

Grant only the necessary permissions to your AWS resources. Use `cdk-nag` to enforce this policy.

❌ **BAD: Over-permissive IAM policy**
```typescript
// Granting full S3 access when only read is needed
bucket.grantReadWrite(lambdaFunction); // Too broad!
```

✅ **GOOD: Least privilege IAM**
```typescript
// Granting specific S3 read access
bucket.grantRead(lambdaFunction); // Only read access
```

### 4.2 Avoid Recursive Lambda Invocations

A Lambda function should never directly or indirectly invoke itself or trigger a process that leads to its own re-invocation. This creates infinite loops and massive bills.

❌ **BAD: Recursive invocation pattern**
```typescript
// Inside a Lambda handler processing SQS messages
// If processing fails, put message back to the SAME queue
// This can lead to an infinite loop if the error persists
sqs.sendMessage({ MessageBody: JSON.stringify(event) });
```

✅ **GOOD: Use Dead-Letter Queues (DLQs) for failures**
```typescript
// CDK definition for Lambda with DLQ
import { SqsDlq } from 'aws-cdk-lib/aws-lambda-event-sources';
import { Queue } from 'aws-cdk-lib/aws-sqs';

const dlq = new Queue(this, 'MyFunctionDLQ');
const myFunction = new NodejsFunction(this, 'MyFunction', {
  // ...
  deadLetterQueue: dlq, // Failed invocations go here
  deadLetterQueueEnabled: true,
});
```

## 5. Testing Approaches

Implement robust testing for your AWS infrastructure and application code.

### 5.1 Mandate `cdk-nag` for Policy Enforcement

Use `cdk-nag` to automatically validate your CDK constructs against security and operational best practices (e.g., AWS Solutions Security Hub, NIST 800-53). Integrate it into your CI/CD pipeline.

```typescript
// bin/my-app.ts
import { App, Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks, NagSuppressions } from 'cdk-nag';
import { MyServiceStack } from '../lib/my-service-stack';

const app = new App();
const stack = new MyServiceStack(app, 'MyServiceStack');

// Apply the AWS Solutions checks to the entire app
Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

// Suppress specific rules with justification if absolutely necessary
NagSuppressions.addStackSuppressions(stack, [
  { id: 'AwsSolutions-IAM4', reason: 'Managed policy for specific use case.' },
  { id: 'AwsSolutions-L1', reason: 'Lambda runtime is managed by AWS.' },
]);

app.synth();
```

### 5.2 Implement Unit and Integration Tests

Write unit tests for your Lambda functions and CDK constructs. Use integration tests for end-to-end validation of your deployed services.

```typescript
// test/my-api.test.ts (CDK Unit Test)
import { App, Stack } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { ApiConstruct } from '../lib/api-construct';
import { Table } from 'aws-cdk-lib/aws-dynamodb';

test('ApiConstruct creates expected resources', () => {
  const app = new App();
  const stack = new Stack(app, 'TestStack');
  const table = new Table(stack, 'TestTable', { tableName: 'TestTable' });

  new ApiConstruct(stack, 'TestApi', { tableName: 'TestTable', table });

  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::Lambda::Function', {
    Runtime: 'nodejs20.x',
    Environment: {
      Variables: {
        TABLE_NAME: 'TestTable',
      },
    },
  });

  template.hasResourceProperties('AWS::ApiGateway::RestApi', {
    Name: 'MyServiceApi',
  });
});
```