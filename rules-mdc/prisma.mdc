---
description: Enforce best practices for Prisma ORM, ensuring type-safe, performant, and maintainable database interactions in modern TypeScript applications.
globs: **/*.prisma
---
# prisma Best Practices

Prisma is the definitive Type-first ORM for modern TypeScript applications. Adhere to these guidelines for optimal performance, maintainability, and type safety, leveraging the latest TypeScript/WASM engine for a superior developer experience.

## 1. Code Organization & Client Management

Always use `prisma.config.ts` for configuration and ensure a single, globally accessible `PrismaClient` instance.

### ✅ GOOD: Type-Safe Configuration (`prisma.config.ts`)

Configure Prisma CLI with full type safety using `defineConfig` or `satisfies PrismaConfig`. This is the standard for Prisma ORM v7+.

```typescript
// prisma.config.ts
import 'dotenv/config';
import { defineConfig, env } from 'prisma/config'; // v7+ config import

export default defineConfig({
  schema: 'prisma/schema.prisma',
  migrations: {
    path: 'prisma/migrations',
    seed: 'tsx prisma/seed.ts', // Use tsx for TypeScript seed scripts
  },
  datasource: {
    url: env('DATABASE_URL'), // Always use environment variables for sensitive data
  },
});
```

### ✅ GOOD: Singleton PrismaClient Instance

Prevent connection pool exhaustion and memory leaks, especially in hot-reloading development environments like Next.js.

```typescript
// lib/prisma.ts
import { PrismaClient } from '@prisma/client'; // Import generated client

const globalForPrisma = global as unknown as { prisma: PrismaClient };

export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: ['query', 'info', 'warn', 'error'], // Essential for observability and debugging
  });

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma; // Persist client across hot reloads in dev
}
```

❌ **BAD**: Instantiating `new PrismaClient()` in every request or module.

```typescript
// ❌ BAD: Creates multiple PrismaClient instances
async function getPosts() {
  const prisma = new PrismaClient(); // This will exhaust your connection pool
  await prisma.post.findMany();
}
```

## 2. Data Modeling & Schema Design

Design your `schema.prisma` for clarity, consistency, and database efficiency.

### ✅ GOOD: Naming Conventions & Essential Fields

Use singular, PascalCase for models and camelCase for fields. Leverage enums for fixed domains. Include `createdAt`, `updatedAt`, and `deletedAt` for robust auditing and soft deletes.

```prisma
// prisma/schema.prisma
model User {
  id        String   @id @default(uuid()) // Prefer UUIDs for distributed systems
  email     String   @unique
  name      String?
  role      Role     @default(USER)     // Use enums for constrained values
  posts     Post[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  deletedAt DateTime? // Crucial for soft deletes and data integrity
}

model Post {
  id        Int      @id @default(autoincrement())
  title     String
  content   String?
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

enum Role {
  USER
  ADMIN
  EDITOR
}
```

### ✅ GOOD: Composite Keys & Indexes

Define composite unique constraints and indexes for efficient lookups on multiple fields.

```prisma
model OrderItem {
  orderId   String
  productId String
  quantity  Int

  @@id([orderId, productId]) // Composite primary key
  @@index([productId])       // Index for efficient lookups by product
}
```

## 3. Query Optimization & Performance

Avoid over-fetching, use bulk operations, and leverage Prisma's built-in optimizations.

### ✅ GOOD: Selective Data Fetching (`select`/`include`)

Only fetch the data you explicitly need. This drastically reduces network overhead and memory usage.

```typescript
// ✅ GOOD: Fetch only ID and email
const users = await prisma.user.findMany({
  select: {
    id: true,
    email: true,
  },
});

// ✅ GOOD: Include related posts only when necessary
const userWithPosts = await prisma.user.findUnique({
  where: { id: 'some-uuid' },
  include: {
    posts: {
      select: {
        id: true,
        title: true,
      },
    },
  },
});
```

❌ **BAD**: Over-fetching all fields and relations.

```typescript
// ❌ BAD: Fetches ALL user fields and ALL post fields, even if not used
const users = await prisma.user.findMany({
  include: { posts: true },
});
```

### ✅ GOOD: Bulk Operations

For large data sets, prefer `createMany()`/`createManyAndReturn()` and `updateMany()` over individual operations. Batch thousands of records for optimal performance.

```typescript
// ✅ GOOD: Bulk create multiple users
const newUsers = await prisma.user.createMany({
  data: [
    { email: 'alice@example.com', name: 'Alice' },
    { email: 'bob@example.com', name: 'Bob' },
  ],
  skipDuplicates: true, // Prevents errors on existing records
});

// ✅ GOOD: Batch writes for very large datasets (e.g., 1000 records per batch)
async function bulkInsert(data: UserCreateInput[]) {
  const batchSize = 1000;
  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    await prisma.user.createMany({ data: batch });
  }
}
```

❌ **BAD**: Individual inserts in a loop for many records.

```typescript
// ❌ BAD: N separate database queries for N records
for (const user of manyUsers) {
  await prisma.user.create({ data: user });
}
```

### ✅ GOOD: Solve N+1 with `include` or Fluent API

Leverage Prisma's dataloader for `findUnique` (which batches queries) or use `include` for related data to avoid the N+1 problem.

```typescript
// ✅ GOOD: Using include to fetch related data in a single query
const usersWithPosts = await prisma.user.findMany({
  include: {
    posts: true,
  },
});

// ✅ GOOD: Fluent API for specific related data (batches findUnique calls)
const userPosts = await prisma.user.findUnique({
  where: { id: 'some-uuid' },
})?.posts({
  where: { published: true },
});
```

## 4. Migration Patterns

Maintain a clean, consistent, and reliable migration history.

### ✅ GOOD: Descriptive Migration Names

Use `npx prisma migrate dev --name <descriptive-name>` for clear, self-documenting migration history.

```bash
npx prisma migrate dev --name add-user-profile-fields
```

❌ **BAD**: Generic names like `init` or `migration` provide no context.

### ✅ GOOD: Never Edit Applied Migrations

Once a migration is applied to *any* environment (dev, staging, production), *never* modify its SQL. This prevents schema drift, data corruption, and deployment issues. If changes are needed, create a new migration.

## 5. Editor Tooling

Maximize productivity and maintain code quality with the official Prisma VS Code extension and Prettier.

### ✅ GOOD: Prisma VS Code Extension & Prettier

Install the official [Prisma VS Code extension](https://marketplace.visualstudio.com/items?itemName=Prisma.prisma) and `prettier-plugin-prisma`. Configure your `settings.json` for automatic formatting on save.

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "[prisma]": {
    "editor.defaultFormatter": "Prisma.prisma"
  }
}
```

This ensures consistent schema formatting, linting, and provides features like go-to-definition and quick-fixes.