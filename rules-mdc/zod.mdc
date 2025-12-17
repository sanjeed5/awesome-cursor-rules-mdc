---
description: This guide outlines definitive best practices for using Zod in TypeScript projects, ensuring robust runtime validation, superior type safety, and maintainable schemas.
globs: **/*.{ts,tsx}
---
# Zod Best Practices

Zod is the definitive TypeScript-first schema validation library. It provides static type inference directly from runtime schemas, catching data-shape errors at both compile time and runtime. Adhere to these guidelines to build resilient, type-safe applications with Zod 4.

## Core Setup & Principles

### 1. Enable Strict TypeScript

Zod's type inference relies on `strict: true` in your `tsconfig.json`. This is non-negotiable for reliable type safety.

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true, // ✅ ALWAYS enable strict mode
    // ...
  }
}
```

### 2. Standardize Zod 4 Adoption

Ensure your project and any Zod-dependent libraries use the modern Zod 4 core.

```json
// package.json
{
  "dependencies": {
    "zod": "^3.25.0 || ^4.0.0" // ✅ Use this peer dependency range
  }
}
```

### 3. Leverage Type Inference (`z.infer`)

Always infer types from your schemas. This eliminates duplication and keeps types synchronized with validation logic.

```typescript
import { z } from 'zod';

const userSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
});

// ❌ BAD: Manually defining types
interface User {
  id: string;
  name: string;
  email: string;
}

// ✅ GOOD: Infer types directly from the schema
type User = z.infer<typeof userSchema>;

function processUser(user: User) {
  console.log(user.email);
}
```

## Data Validation & Error Handling

### 4. Always Parse Untrusted Data

Never trust external data. Use `schema.parse()` or `schema.safeParse()` immediately upon receiving data from APIs, forms, or environment variables.

```typescript
import { z } from 'zod';

const idSchema = z.string().uuid();

// ❌ BAD: Assuming data is valid
function processId(id: string) {
  // This 'id' could be anything if not validated
  console.log(id.toUpperCase());
}

// ✅ GOOD: Use safeParse for external/untrusted data
function handleRequest(rawId: unknown) {
  const result = idSchema.safeParse(rawId);
  if (!result.success) {
    console.error('Invalid ID received:', result.error.issues);
    return { status: 400, message: 'Invalid ID' };
  }
  // Data is now type-safe and validated
  processId(result.data);
  return { status: 200, message: 'ID processed' };
}

// ✅ GOOD: Use parse for trusted data or when immediate error throwing is desired
function loadConfig(envVar: unknown) {
  try {
    const config = z.object({ API_KEY: z.string().min(1) }).parse(envVar);
    console.log('Config loaded:', config.API_KEY);
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Invalid environment config:', error.issues);
      process.exit(1); // Critical error, exit application
    }
    throw error;
  }
}
```

### 5. Customize Error Messages for UX

Provide clear, user-friendly error messages directly within your schemas.

```typescript
import { z } from 'zod';

const loginSchema = z.object({
  email: z.string().email({ message: 'Invalid email address.' }), // ✅ Custom message
  password: z.string().min(8, { message: 'Password must be at least 8 characters.' }), // ✅ Custom message
});

// Integrate with form libraries (e.g., React Hook Form)
// const { register, handleSubmit, formState: { errors } } = useForm({
//   resolver: zodResolver(loginSchema),
// });
```

## Schema Design & Organization

### 6. Keep Schemas Modular & Domain-Specific

Organize schemas into small, focused files based on their domain. Avoid monolithic schema files.

```
// ❌ BAD: monolithic schemas.ts
// schemas.ts
export const userSchema = z.object({ /* ... */ });
export const productSchema = z.object({ /* ... */ });
export const orderSchema = z.object({ /* ... */ });

// ✅ GOOD: modular, domain-specific files
// schemas/user.schema.ts
export const userSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
});

// schemas/product.schema.ts
export const productSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(3),
  price: z.number().positive(),
});

// schemas/index.ts (optional re-export)
export * from './user.schema';
export * from './product.schema';
```

### 7. Use Transforms and Refinements for Data Normalization

Normalize and sanitize data as part of validation.

```typescript
import { z } from 'zod';

// ❌ BAD: Manual post-validation processing
const emailInputSchemaBad = z.string().email();
const processEmailBad = (email: string) => email.toLowerCase().trim();

// ✅ GOOD: Integrate transforms directly into the schema
const emailInputSchema = z
  .string()
  .trim() // ✅ Trim whitespace
  .toLowerCase() // ✅ Convert to lowercase
  .email({ message: 'Invalid email address' });

const postIdSchema = z.string().uuid().transform((val) => `post-${val}`); // ✅ Transform data type or format

const passwordSchema = z.string()
  .min(8)
  .refine(val => /[A-Z]/.test(val), { message: 'Password must contain an uppercase letter' }); // ✅ Refine with custom logic
```

### 8. Prefer `z.coerce` for Type Coercion

Use `z.coerce` for explicit type conversion of known string inputs (e.g., from query parameters, environment variables).

```typescript
import { z } from 'zod';

// ❌ BAD: Manual parsing or loose validation
const pageParamBad = z.string(); // Will be '1' not 1
const pageNumberBad = parseInt(pageParamBad.parse('10'));

// ✅ GOOD: Coerce directly in the schema
const querySchema = z.object({
  page: z.coerce.number().int().positive().default(1), // ✅ Coerces '10' to 10
  isActive: z.coerce.boolean().default(false), // ✅ Coerces 'true'/'false' to boolean
});

const { page, isActive } = querySchema.parse({ page: '5', isActive: 'true' });
console.log(page, typeof page); // 5, 'number'
console.log(isActive, typeof isActive); // true, 'boolean'
```

### 9. Differentiate `optional()`, `nullable()`, and `default()`

Understand the nuances for robust schema design.

```typescript
import { z } from 'zod';

const userProfileSchema = z.object({
  username: z.string(),
  // ✅ optional(): field can be present or absent (undefined)
  bio: z.string().optional(),
  // ✅ nullable(): field can be present with a value or null
  website: z.string().url().nullable(),
  // ✅ default(): provides a fallback value if field is undefined
  status: z.enum(['active', 'inactive']).default('active'),
});

type UserProfile = z.infer<typeof userProfileSchema>;
/*
{
  username: string;
  bio?: string | undefined;
  website: string | null;
  status: "active" | "inactive";
}
*/
```

## Testing & Maintenance

### 10. Write Schema-Focused Unit Tests

Thoroughly test your Zod schemas with both valid and invalid data to prevent regressions.

```typescript
// tests/schemas/user.schema.test.ts
import { userSchema } from '../../schemas/user.schema';

describe('userSchema', () => {
  it('should validate a valid user object', () => {
    const validUser = {
      id: 'a1b2c3d4-e5f6-7890-1234-567890abcdef',
      name: 'John Doe',
      email: 'john.doe@example.com',
    };
    expect(() => userSchema.parse(validUser)).not.toThrow(); // ✅ Validates successfully
  });

  it('should invalidate an invalid email', () => {
    const invalidUser = {
      id: 'a1b2c3d4-e5f6-7890-1234-567890abcdef',
      name: 'John Doe',
      email: 'invalid-email',
    };
    const result = userSchema.safeParse(invalidUser);
    expect(result.success).toBe(false); // ✅ Fails validation
    if (!result.success) {
      expect(result.error.issues[0].message).toBe('Invalid email'); // ✅ Specific error message
    }
  });
});
```