---
description: This guide provides opinionated, actionable best practices for building scalable, maintainable full-stack applications using AWS Amplify Gen 2, focusing on TypeScript-first development and efficient cloud resource management.
globs: **/*
---
# AWS Amplify Gen 2 Best Practices

Amplify Gen 2 is a TypeScript-first, code-first full-stack platform. Embrace its opinionated nature to build robust applications efficiently.

## 1. Code Organization & Backend Structure

Always define your backend resources (Auth, Data, Functions) in their dedicated TypeScript files within the `amplify/` directory. This ensures type safety and leverages Amplify's convention-over-configuration.

❌ **BAD: Mixing resource definitions**
```typescript
// amplify/backend.ts
import { defineBackend } from '@aws-amplify/backend';
import { a } from '@aws-amplify/backend/data';
import { defineAuth } from '@aws-amplify/backend/auth';

const backend = defineBackend({
  auth: defineAuth({
    loginWith: { email: true },
  }),
  data: a.defineData({
    // Data schema defined directly here
    Todo: a.model({
      content: a.string(),
    }),
  }),
});
```

✅ **GOOD: Dedicated resource files**
```typescript
// amplify/backend.ts
import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource.js';
import { data } from './data/resource.js';

const backend = defineBackend({
  auth,
  data,
});

// amplify/auth/resource.ts
import { defineAuth } from '@aws-amplify/backend/auth';
export const auth = defineAuth({
  loginWith: { email: true },
});

// amplify/data/resource.ts
import { a } from '@aws-amplify/backend/data';
export const data = a.defineData({
  Todo: a.model({
    content: a.string(),
  }),
});
```

## 2. Backend Functions: Single Responsibility & Modularity

Adhere strictly to the Single Responsibility Principle for Amplify functions. Avoid monolithic functions or chaining Lambdas, which increase complexity and reduce maintainability. Share common logic via separate TypeScript files.

❌ **BAD: Monolithic or chained functions**
```typescript
// amplify/functions/myFunction.ts
export const handler = async (event: any) => {
  // ... extensive logic for user creation
  // ... extensive logic for sending welcome email
  // ... extensive logic for processing data
  // ... then calls another Lambda function
};
```

✅ **GOOD: Modular functions with shared utilities**
```typescript
// amplify/functions/createUser.ts
import { createUserInDB } from '../shared/userUtils.js';
import { sendWelcomeEmail } from '../shared/emailService.js';

export const handler = async (event: any) => {
  const userData = JSON.parse(event.body);
  const user = await createUserInDB(userData);
  await sendWelcomeEmail(user.email);
  return { statusCode: 200, body: JSON.stringify(user) };
};

// amplify/shared/userUtils.ts
export const createUserInDB = async (data: any) => { /* ... */ };

// amplify/shared/emailService.ts
export const sendWelcomeEmail = async (email: string) => { /* ... */ };
```

## 3. Data Modeling & Authorization

Define your data models with clear relationships and granular authorization rules directly in `amplify/data/resource.ts`. Leverage `a.authorization` for fine-grained access control.

```typescript
// amplify/data/resource.ts
import { a } from '@aws-amplify/backend/data';

export const data = a.defineData({
  Post: a.model({
    title: a.string().required(),
    content: a.string(),
    owner: a.string(), // Automatically populated by Amplify Auth
    comments: a.hasMany('Comment', 'postId'),
  }).authorization((allow) => [
    allow.owner(), // Owner can perform all operations
    allow.public.read(), // Public can read posts
  ]),
  Comment: a.model({
    content: a.string().required(),
    postId: a.id(),
    post: a.belongsTo('Post', 'postId'),
    author: a.string(),
  }).authorization((allow) => [
    allow.owner(),
    allow.authenticated.read(), // Authenticated users can read comments
  ]),
});
```

## 4. Frontend Integration & UI Components

Always use the `@aws-amplify/ui-react` library for common UI patterns like authentication. Configure Amplify early in your application lifecycle with `amplify_outputs.json`.

```typescript
// src/main.tsx (or _app.tsx for Next.js)
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import '@aws-amplify/ui-react/styles.css';
import { Amplify } from 'aws-amplify';
import outputs from '../amplify_outputs.json'; // Ensure this path is correct

Amplify.configure(outputs);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// src/App.tsx
import { withAuthenticator, Button } from '@aws-amplify/ui-react';

function App({ signOut, user }: { signOut: () => void; user: any }) {
  return (
    <div>
      <h1>Hello, {user?.username}!</h1>
      <Button onClick={signOut}>Sign Out</Button>
      {/* Your application content */}
    </div>
  );
}

export default withAuthenticator(App);
```

## 5. Data Fetching & State Management

Use `generateClient` for type-safe data operations. For real-time updates, prefer `client.models.Model.observeQuery()` over repeated `list()` calls.

```typescript
// src/App.tsx (or a data-fetching component)
import { useEffect, useState } from 'react';
import { generateClient } from 'aws-amplify/data';
import type { Schema } from '../amplify/data/resource'; // Import generated types

const client = generateClient<Schema>();

function TodoList() {
  const [todos, setTodos] = useState<Schema['Todo'][]>([]);

  useEffect(() => {
    // Use observeQuery for real-time updates
    const sub = client.models.Todo.observeQuery().subscribe({
      next: ({ items }) => setTodos([...items]),
      error: (error) => console.error('Subscription error:', error),
    });
    return () => sub.unsubscribe();
  }, []);

  const createTodo = async () => {
    await client.models.Todo.create({ content: `New Todo ${Date.now()}` });
  };

  return (
    <div>
      <button onClick={createTodo}>Add Todo</button>
      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>{todo.content}</li>
        ))}
      </ul>
    </div>
  );
}
```

## 6. UI Builder (Figma-to-React) Best Practices

When using Amplify Studio's UI Builder, guide designers to follow these rules for optimal conversion:

*   **Mark Frames as Components:** Only Figma components are converted.
*   **Use Auto Layout:** Treat Figma Auto Layout as CSS Flexbox for responsiveness.
*   **Code Interactive States:** Manually implement hover/active states in code; Studio doesn't convert them.
*   **Consistent Variants:** Figma variants must have identical child element structures.
*   **"Hug Contents" / "Fill Container":** Use these constraints for proper resizing.
*   **Manual Font Integration:** Fonts are not automatically exported; include them via standard React build pipelines.

## 7. Development & Deployment Workflow

Leverage Amplify's full-stack Git deployments and per-developer cloud sandboxes.

*   **Local Sandboxes:** Use `ampx sandbox` for rapid local iteration with high-fidelity cloud environments.
*   **Fullstack Branching:** Map Git branches 1:1 to Amplify environments (dev, staging, prod) for zero-config CI/CD.
*   **Monorepos/Multi-repos:** For complex setups, separate frontend and backend repositories and use webhooks to trigger frontend builds on backend updates.

❌ **BAD: Manual backend deployments for local dev**
```bash
# Don't manually deploy to shared dev environments for every change
amplify push --env dev
```

✅ **GOOD: Per-developer sandboxes for local dev**
```bash
# Use a dedicated sandbox for your local development
ampx sandbox
```