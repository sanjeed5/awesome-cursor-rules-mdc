---
description: Provides definitive best practices and actionable code examples for interacting with AWS DynamoDB using the AWS SDK v3, focusing on data modeling, performance, security, and common pitfalls.
globs: **/*
---
# aws-dynamodb Best Practices

This guide outlines the definitive best practices for interacting with AWS DynamoDB, ensuring optimal performance, security, and maintainability. We leverage AWS SDK v3, modern data modeling, and infrastructure-as-code principles.

## 1. Data Modeling: Single-Table Design is King

Embrace single-table design to minimize operational overhead and maximize query flexibility. Only use Global Secondary Indexes (GSIs) when your access patterns *cannot* be satisfied by the primary key.

### ✅ GOOD: Single-Table Design with Composite Primary Keys
Model your primary key (Partition Key + Sort Key) to serve your most frequent access patterns. Use generic attribute names like `PK` and `SK` to allow for diverse item types within the same table.

```typescript
// Example: User and Order data in a single table
// PK: USER#<userId>, SK: #METADATA (for user details)
// PK: USER#<userId>, SK: ORDER#<orderId> (for user's orders)

interface UserItem {
  PK: string; // e.g., USER#123
  SK: string; // e.g., #METADATA
  userId: string;
  username: string;
  email: string;
}

interface OrderItem {
  PK: string; // e.g., USER#123
  SK: string; // e.g., ORDER#abc-123
  orderId: string;
  userId: string;
  orderDate: string;
  totalAmount: number;
}

// Access Pattern: Get User by userId -> Query: PK = 'USER#<userId>', SK = '#METADATA'
// Access Pattern: Get all Orders for a User -> Query: PK = 'USER#<userId>', SK begins_with 'ORDER#'
```

### ❌ BAD: Multi-Table Design for Related Entities
Creating separate tables for every entity type (e.g., `UsersTable`, `OrdersTable`) leads to increased operational complexity and often requires more complex cross-table queries or application-level joins.

```typescript
// Don't do this unless absolutely necessary for strict isolation
// UsersTable: PK = userId
// OrdersTable: PK = orderId, GSI on userId
```

## 2. Code Organization and Structure

Organize your DynamoDB interactions into dedicated data access layers (DALs) or repositories. Use the AWS SDK v3's modular clients and `async/await`.

### ✅ GOOD: Dedicated Data Access Layer with Typed Clients
Encapsulate all DynamoDB operations, ensuring consistent error handling, logging, and parameter construction. Use the DocumentClient for simpler JSON interaction.

```typescript
// src/data/userRepository.ts
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand, PutCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({ region: process.env.AWS_REGION });
const ddbDocClient = DynamoDBDocumentClient.from(client);
const TABLE_NAME = process.env.DYNAMODB_TABLE_NAME || "YourAppTable";

export class UserRepository {
  static async getUserById(userId: string): Promise<UserItem | undefined> {
    const params = {
      TableName: TABLE_NAME,
      Key: { PK: `USER#${userId}`, SK: '#METADATA' },
    };
    const { Item } = await ddbDocClient.send(new GetCommand(params));
    return Item as UserItem | undefined;
  }
}
```

### ❌ BAD: Inline DynamoDB Calls or Generic Clients
Avoid scattering DynamoDB logic throughout your application or using untyped, generic clients. This makes maintenance, testing, and refactoring extremely difficult.

```typescript
// src/handlers/createUser.ts - Bad practice
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});
const ddbDocClient = DynamoDBDocumentClient.from(client);

export const handler = async (event: any) => {
  // ...
  await ddbDocClient.send(new PutCommand({
    TableName: "MyHardcodedTable", // Hardcoded table name
    Item: { PK: `USER#${event.userId}`, SK: '#METADATA', /* ... */ },
  }));
};
```

## 3. Performance Considerations

Optimize for read/write capacity, minimize data transfer, and leverage eventual consistency where appropriate.

### ✅ GOOD: Use On-Demand Capacity or Provisioned with Auto-Scaling
Choose `ON_DEMAND` for unpredictable workloads. For predictable traffic, use `PROVISIONED` with `AutoScaling` to manage costs and performance.

```yaml
# CloudFormation (example)
Resources:
  MyDynamoDBTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${Environment}-MyTable-v1"
      AttributeDefinitions:
        - AttributeName: PK
          AttributeType: S
        - AttributeName: SK
          AttributeType: S
      KeySchema:
        - AttributeName: PK
          KeyType: HASH
        - AttributeName: SK
          KeyType: RANGE
      BillingMode: PAY_PER_REQUEST # ON_DEMAND is default for new tables
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true }
      SSESpecification: { SSEEnabled: true } # Encryption at rest
```

### ❌ BAD: Fixed Provisioned Capacity for Spiky Workloads
Manually setting fixed provisioned capacity for unpredictable workloads leads to either throttling (under-provisioned) or wasted cost (over-provisioned).

```yaml
# CloudFormation (example of bad practice for spiky workloads)
Resources:
  MyDynamoDBTable:
    Type: AWS::DynamoDB::Table
    Properties:
      # ...
      BillingMode: PROVISIONED
      ProvisionedThroughput:
        ReadCapacityUnits: 10 # Fixed, will throttle or waste money
        WriteCapacityUnits: 10 # Fixed, will throttle or waste money
```

### ✅ GOOD: Leverage `Query` and `GetItem` with Strong Typing
`GetItem` for single item retrieval by primary key. `Query` for items with the same partition key and an optional sort key condition. Always use `ConsistentRead: