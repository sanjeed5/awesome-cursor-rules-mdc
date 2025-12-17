---
description: This guide provides definitive, opinionated best practices for using Drizzle ORM to build robust, type-safe, and performant database applications in TypeScript.
globs: **/*.{ts,js}
---
# Drizzle ORM Best Practices

Drizzle ORM is our go-to for type-safe SQL. Follow these guidelines religiously to ensure maintainable, performant, and secure database interactions.

## 1. Code Organization and Structure

Organize your schema logically. Each domain should have its own file, and all schemas should be re-exported from a central `index.ts` for Drizzle Kit.

### Schema File Structure

**✅ GOOD: Modular Schema Files**
Keep each logical domain in its own file under `src/db/schema/`.

```typescript
// src/db/schema/users.ts
import { pgTable, serial, text, varchar } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  fullName: text('full_name').notNull(),
  email: varchar('email', { length: 256 }).unique().notNull(),
});

// src/db/schema/posts.ts
import { pgTable, serial, text, timestamp, varchar } from 'drizzle-orm/pg-core';
import { users } from './users'; // Import related schema

export const posts = pgTable('posts', {
  id: serial('id').primaryKey(),
  title: varchar('title', { length: 256 }).notNull(),
  content: text('content'),
  authorId: serial('author_id').references(() => users.id).notNull(), // Foreign key
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// src/db/schema/index.ts (for Drizzle Kit and client)
import * as usersSchema from './users';
import * as postsSchema from './posts';

export const schema = {
  ...usersSchema,
  ...postsSchema,
};
```

**❌ BAD: Monolithic Schema File**
Avoid dumping all tables into a single file. It hinders discoverability and tree-shaking.

```typescript
// src/db/schema.ts
import { pgTable, serial, text, varchar, timestamp } from 'drizzle-orm/pg-core';

export const users = pgTable('users', { /* ... */ });
export const posts = pgTable('posts', { /* ... */ });
// ... many more tables
```

### Naming Conventions

**✅ GOOD: `camelCase` for TypeScript, `snake_case` for Database**
This aligns with community standards and keeps your code idiomatic. Declare enums with `pgEnum` and reuse them.

```typescript
import { pgTable, serial, text, varchar, pgEnum } from 'drizzle-orm/pg-core';

export const userRoleEnum = pgEnum('user_role', ['admin', 'editor', 'viewer']);

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  fullName: text('full_name').notNull(), // TS: fullName, DB: full_name
  emailAddress: varchar('email_address', { length: 256 }).unique().notNull(), // TS: emailAddress, DB: email_address
  role: userRoleEnum('role').default('viewer').notNull(),
});
```

**❌ BAD: Inconsistent Naming**
Mixing casing styles leads to confusion and potential errors.

```typescript
export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  full_name: text('full_name').notNull(), // TS: full_name (camelCase expected)
  email_address: varchar('email_address', { length: 256 }).unique().notNull(),
});
```

## 2. Data Modeling

Define your schema precisely. Leverage Drizzle's type inference by explicitly marking constraints and properties.

### Primary Keys, Nullability, and Defaults

Always define primary keys, explicitly mark `notNull()`, and use `default()` for consistent data.

**✅ GOOD: Explicit Constraints**

```typescript
import { pgTable, serial, text, timestamp, boolean } from 'drizzle-orm/pg-core';

export const tasks = pgTable('tasks', {
  id: serial('id').primaryKey(), // Explicit Primary Key
  title: text('title').notNull(), // Required field
  description: text('description'), // Optional field
  isComplete: boolean('is_complete').default(false).notNull(), // Default value, required
  createdAt: timestamp('created_at').defaultNow().notNull(), // Default to current timestamp
  updatedAt: timestamp('updated_at').defaultNow().onUpdateFn(() => new Date()).notNull(), // Auto-update timestamp
});
```

**❌ BAD: Missing Constraints**
Omitting `primaryKey()` or `notNull()` can lead to runtime errors and unexpected database behavior.

```typescript
export const tasks = pgTable('tasks', {
  id: serial('id'), // Missing .primaryKey()
  title: text('title'), // Missing .not