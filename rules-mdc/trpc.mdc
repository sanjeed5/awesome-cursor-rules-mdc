---
description: Definitive guidelines for building robust, type-safe, and maintainable tRPC APIs, emphasizing modularity, Zod validation, and efficient error handling.
globs: **/*.{ts,tsx}
---
# tRPC Best Practices

These rules provide a definitive guide for building tRPC APIs within our team. Adhere to these principles for consistent, high-quality, and type-safe backend development.

## 1. Modularize Routers by Feature

Organize your API into distinct, feature-specific routers. This improves maintainability, readability, and allows for easier scaling.

❌ **BAD: Monolithic Router**
```typescript
// src/server/api/root.ts
export const appRouter = t.router({
  getUser: publicProcedure.query(() => { /* ... */ }),
  createUser: protectedProcedure.mutation(() => { /* ... */ }),
  getPost: publicProcedure.query(() => { /* ... */ }),
  createPost: protectedProcedure.mutation(() => { /* ... */ }),
  // ... hundreds of procedures
});
```

✅ **GOOD: Modular Routers**
```typescript
// src/server/api/routers/user.ts
export const userRouter = t.router({
  getUser: publicProcedure.query(() => { /* ... */ }),
  createUser: protectedProcedure.mutation(() => { /* ... */ }),
});

// src/server/api/routers/post.ts
export const postRouter = t.router({
  getPost: publicProcedure.query(() => { /* ... */ }),
  createPost: protectedProcedure.mutation(() => { /* ... */ }),
});

// src/server/api/root.ts
export const appRouter = t.router({
  user: userRouter,
  post: postRouter,
});
```

## 2. Centralize Context Definition

Define your `createTRPCContext` function once to provide shared resources (e.g., database client, session, headers) to all procedures. This ensures consistency and avoids prop-drilling or global state.

```typescript
// src/server/api/context.ts
import { type inferAsyncReturnType } from '@trpc/server';
import { type CreateNextContextOptions } from '@trpc/server/adapters/next';
import { db } from '../db'; // Your database client
import { getServerAuthSession } from '../auth'; // Your auth session helper

export const createTRPCContext = async (opts: CreateNextContextOptions) => {
  const { req, res } = opts;
  const session = await getServerAuthSession({ req, res });

  return {
    db,
    session,
    headers: opts.req.headers,
  };
};

export type Context = inferAsyncReturnType<typeof createTRPCContext>;
```

## 3. Enforce Input Validation with Zod

Always validate procedure inputs using Zod schemas. This provides runtime validation mirroring compile-time types, preventing malformed data from reaching business logic and improving API robustness.

❌ **BAD: No Input Validation**
```typescript
export const postRouter = t.router({
  createPost: protectedProcedure
    .input(z.any()) // Avoid z.any()
    .mutation(async ({ ctx, input }) => {
      // Manual validation or no validation
      if (typeof input.title !== 'string' || input.title.length < 3) {
        throw new Error('Invalid title');
      }
      return ctx.db.post.create({ data: input });
    }),
});
```

✅ **GOOD: Zod for Input Validation**
```typescript
import { z } from 'zod';

export const postRouter = t.router({
  createPost: protectedProcedure
    .input(z.object({
      title: z.string().min(3, "Title must be at least 3 characters"),
      content: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      // Input is guaranteed to be type-safe and validated
      return ctx.db.post.create({ data: input });
    }),
});
```

## 4. Implement Middleware for Authorization

Use tRPC middleware to enforce authentication and authorization rules declaratively. This keeps procedure logic clean and ensures consistent access control.

```typescript
// src/server/api/trpc.ts
import { initTRPC, TRPCError } from '@trpc/server';
import superjson from 'superjson';
import { ZodError } from 'zod';
import { type Context } from './context';

const t = initTRPC.context<Context>().create({
  transformer: superjson,
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError: error.cause instanceof ZodError ? error.cause.flatten() : null,
      },
    };
  },
});

export const createTRPCRouter = t.router;
export const publicProcedure = t.procedure;

// Middleware to check for authenticated session
const enforceUserIsAuthed = t.middleware(({ ctx, next }) => {
  if (!ctx.session || !ctx.session.user) {
    throw new TRPCError({ code: 'UNAUTHORIZED' });
  }
  return next({
    ctx: {
      // Infers the `session` as non-nullable
      session: { ...ctx.session, user: ctx.session.user },
    },
  });
});

export const protectedProcedure = t.procedure.use(enforceUserIsAuthed);

// Example: Admin-only procedure
const enforceUserIsAdmin = enforceUserIsAuthed.unstable_pipe(({ ctx, next }) => {
  if (ctx.session.user.role !== 'ADMIN') { // Assuming a 'role' on user
    throw new TRPCError({ code: 'FORBIDDEN', message: 'Only admins can perform this action.' });
  }
  return next();
});

export const adminProcedure = t.procedure.use(enforceUserIsAdmin);
```

## 5. Standardize Error Formatting

Configure `errorFormatter` in `initTRPC.create` to provide consistent and helpful error responses, especially for Zod validation errors.

```typescript
// src/server/api/trpc.ts (excerpt)
const t = initTRPC.context<Context>().create({
  transformer: superjson,
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        // Flatten Zod errors for client-side consumption
        zodError: error.cause instanceof ZodError ? error.cause.flatten() : null,
      },
    };
  },
});
```

## 6. Use Superjson for Data Serialization

Always use `superjson` as your data transformer. It correctly serializes and deserializes complex types (like `Date`, `Map`, `Set`) between client and server, maintaining type fidelity.

```typescript
// src/server/api/trpc.ts (excerpt)
import superjson from 'superjson';

const t = initTRPC.context<Context>().create({
  transformer: superjson, // Essential for type-safe serialization
  // ...
});
```

## 7. Prefer Relative Imports for tRPC Types

When importing `AppRouter` or other tRPC-related types into your client-side code, use relative paths instead of TypeScript path aliases. tRPC's type inference can sometimes struggle with alias resolution, leading to `any` types.

❌ **BAD: Path Alias for tRPC Types**
```typescript
// src/utils/trpc.ts
import { type AppRouter } from '@/server/api/root'; // May resolve to 'any'
```

✅ **GOOD: Relative Path for tRPC Types**
```typescript
// src/utils/trpc.ts
import { type AppRouter } from '../../server/api/root'; // Ensures correct type inference
```

## 8. Implement Robust Client-Side Invalidation

When using tRPC with `@tanstack/react-query`, ensure proper cache invalidation after mutations. This prevents stale data from being displayed and ensures UI consistency.

```typescript
// src/components/PostForm.tsx
import { api } from '@/utils/api'; // Your tRPC client instance

function PostForm() {
  const utils = api.useUtils(); // Access TanStack Query client
  const createPost = api.post.createPost.useMutation({
    onSuccess: () => {
      // Invalidate all queries under the 'post' router to refetch lists/details
      utils.post.invalidate();
      // For more granular control:
      // utils.post.getPosts.invalidate();
    },
  });

  // ... form submission logic
  const handleSubmit = (data: { title: string; content?: string }) => {
    createPost.mutate(data);
  };

  return (/* ... */);
}
```

## 9. Document Procedures with JSDoc

Add comprehensive JSDoc comments to each procedure. This improves developer experience by providing immediate context and usage instructions in the IDE.

```typescript
// src/server/api/routers/post.ts
export const postRouter = t.router({
  /**
   * Fetches all posts from the database.
   * @returns A list of posts.
   */
  getPosts: publicProcedure
    .query(async ({ ctx }) => {
      return ctx.db.post.findMany();
    }),

  /**
   * Creates a new post.
   * @param input - The post data (title, content).
   * @returns The newly created post.
   * @throws TRPCError if user is unauthorized.
   */
  createPost: protectedProcedure
    .input(z.object({
      title: z.string().min(3),
      content: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      return ctx.db.post.create({ data: { ...input, authorId: ctx.session.user.id } });
    }),
});
```

## 10. Enable TypeScript Strict Mode

Always develop with TypeScript's `strict` mode enabled in `tsconfig.json`. This catches a wide range of common programming errors at compile time, maximizing tRPC's type-safety benefits.

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true, // Essential for robust type-safety
    // ... other options
  },
  // ...
}
```