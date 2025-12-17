---
description: This guide outlines definitive best practices for building robust, performant, and maintainable applications with Remix, emphasizing co-location, type safety, and efficient data handling.
globs: **/*.{js,jsx,ts,tsx}
---
# Remix Best Practices

Remix provides a powerful full-stack framework built on React Router v7. Adhering to these guidelines ensures a consistent, high-quality codebase that leverages Remix's strengths for server-rendered applications.

## 1. Code Organization and Structure

**Always co-locate route logic and UI.** Each route must live in its own folder, containing its `route.tsx` component, `loader.ts`, `action.ts`, and optional `meta.ts` or `error-boundary.tsx`. This keeps related concerns together and simplifies navigation.

❌ BAD:
```tsx
// app/data/posts.ts
export async function getPost(id: string) { /* ... */ }

// app/components/PostDetail.tsx
import { getPost } from '~/data/posts';
function PostDetail() {
  // Client-side fetch or prop drilling
}

// app/routes/posts.$postId.tsx
import PostDetail from '~/components/PostDetail';
export default function PostRoute() {
  return <PostDetail />;
}
```

✅ GOOD:
```tsx
// app/routes/posts.$postId/route.tsx
import { json, useLoaderData } from "@remix-run/node";
import type { LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ params }: LoaderFunctionArgs) {
  const post = await db.getPost(params.postId); // Direct server call
  if (!post) throw new Response("Not Found", { status: 404 });
  return json({ post });
}

export default function PostDetailRoute() {
  const { post } = useLoaderData<typeof loader>();
  return (
    <div>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </div>
  );
}

// app/routes/posts.$postId/error-boundary.tsx
import { isRouteErrorResponse, useRouteError } from "@remix-run/react";
export function ErrorBoundary() {
  const error = useRouteError();
  if (isRouteErrorResponse(error)) {
    return <div>Error: {error.status} {error.statusText}</div>;
  }
  return <div>Unknown Error!</div>;
}
```

**Maintain a clear `src` directory hierarchy.** Separate concerns into logical folders like `app/components`, `app/features`, `app/hooks`, `app/utils`, `app/services`, etc.

❌ BAD:
```
app/
├── index.tsx
├── post-list.tsx
├── user-profile.tsx
├── api.ts
└── helpers.ts
```

✅ GOOD:
```
app/
├── components/       // Reusable UI components
├── features/         // Domain-specific components/logic
├── hooks/            // Custom React hooks
├── routes/           // File-based routes (as above)
├── services/         // Database/API interaction logic
├── utils/            // Generic utility functions
└── entry.client.tsx
└── entry.server.tsx
```

**Enforce consistent naming conventions.** Use PascalCase for React components and types, and kebab-case for file names.

❌ BAD: `mycomponent.jsx`, `MyComponent.ts`, `userProfile.tsx`
✅ GOOD: `MyComponent.tsx`, `user-profile.ts`, `UserProfile.tsx`

## 2. Common Patterns and Anti-patterns

**Use Remix `Form` for all data mutations.** This leverages Remix's built-in revalidation and pending UI states for a robust user experience.

❌ BAD:
```tsx
import { useState } from 'react';
function NewPostForm() {
  const [title, setTitle] = useState('');
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetch('/posts', { method: 'POST', body: JSON.stringify({ title }) });
  };
  return <form onSubmit={handleSubmit}>...</form>;
}
```

✅ GOOD:
```tsx
// app/routes/posts.new/route.tsx
import { Form, redirect } from "@remix-run/react";
import type { ActionFunctionArgs } from "@remix-run/node";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const title = formData.get("title");
  await db.createPost({ title });
  return redirect("/posts");
}

export default function NewPostForm() {
  return (
    <Form method="post">
      <input type="text" name="title" />
      <button type="submit">Create Post</button>
    </Form>
  );
}
```

**Utilize Resource Routes for API-only endpoints.** For data fetching or mutations not tied to a specific UI page, create a resource route.

❌ BAD: Creating a separate API server or `app/api/posts.ts` that isn't a route.
✅ GOOD:
```tsx
// app/routes/api.posts.$postId.ts
import { json } from "@remix-run/node";
import type { LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ params }: LoaderFunctionArgs) {
  const post = await db.getPost(params.postId);
  return json(post);
}
```

## 3. Performance Considerations

**Perform all heavy data fetching in `loader` and `action` functions.** These run exclusively on the server, keeping your client bundle small and fast.

❌ BAD:
```tsx
// app/routes/dashboard.tsx
import { useEffect, useState } from 'react';
function Dashboard() {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch('/api/dashboard-data').then(res => res.json()).then(setData);
  }, []);
  return <div>{data ? 'Loaded' : 'Loading...'}</div>;
}
```

✅ GOOD:
```tsx
// app/routes/dashboard.tsx
import { json, useLoaderData } from "@remix-run/node";
export async function loader() {
  const data = await db.getDashboardData(); // Server-side data fetch
  return json(data);
}
export default function Dashboard() {
  const data = useLoaderData<typeof loader>();
  return <div>{data ? 'Loaded' : 'Loading...'}</div>;
}
```

**Defer non-critical data with `defer` and `Await` for faster initial page loads.** This allows critical data to render immediately while slower data streams in.

❌ BAD: Waiting for all data to resolve before rendering.
```tsx
// loader that waits for everything
export async function loader() {
  const critical = await getCriticalData();
  const slow = await getSlowData(); // Blocks initial render
  return json({ critical, slow });
}
```

✅ GOOD:
```tsx
// loader that streams slow data
import { defer, Await, json } from "@remix-run/node";
import { Suspense } from "react";

export async function loader() {
  const critical = await getCriticalData();
  const slowPromise = getSlowData(); // Returns a promise
  return defer({ critical, slow: slowPromise });
}

export default function MyRoute() {
  const { critical, slow } = useLoaderData<typeof loader>();
  return (
    <div>
      <h1>Critical: {critical.message}</h1>
      <Suspense fallback={<p>Loading slow data...</p>}>
        <Await resolve={slow}>
          {(resolvedSlow) => <p>Slow: {resolvedSlow.message}</p>}
        </Await>
      </Suspense>
    </div>
  );
}
```

## 4. Common Pitfalls and Gotchas

**Avoid client-side specific APIs in `loader` or `action` functions.** These functions run on the server, so `window`, `localStorage`, `document`, etc., are unavailable.

❌ BAD:
```tsx
export async function loader() {
  const token = window.localStorage.getItem('token'); // Will fail on server
  // ...
}
```

✅ GOOD: Pass necessary data via cookies, headers, or form data.
```tsx
export async function loader({ request }: LoaderFunctionArgs) {
  const cookieHeader = request.headers.get("Cookie");
  const token = getAuthTokenFromCookie(cookieHeader); // Server-safe
  // ...
}
```

**Treat `useLoaderData` and `useActionData` as immutable.** Remix revalidates data, so direct mutations won't persist and can lead to unexpected behavior.

❌ BAD:
```tsx
const { items } = useLoaderData<typeof loader>();
items.push("new item"); // Don't mutate loader data directly
```

✅ GOOD: Create a new array or object if you need to modify data client-side.
```tsx
const { items } = useLoaderData<typeof loader>();
const [clientItems, setClientItems] = useState(items);
// ... setClientItems([...clientItems, "new item"])
```

## 5. Hooks Best Practices

**Use `useNavigation` for pending UI states.** This provides a seamless user experience during transitions.

❌ BAD: Manually managing `isLoading` state with `useState` for form submissions.
✅ GOOD:
```tsx
import { Form, useNavigation } from "@remix-run/react";
export default function MyForm() {
  const navigation = useNavigation();
  const isSubmitting = navigation.state === "submitting";
  return (
    <Form method="post">
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Saving..." : "Save"}
      </button>
    </Form>
  );
}
```

**Use `useFetcher` for non-navigation data mutations or fetches.** This is ideal for background updates, like liking a post, without triggering a full page navigation.

❌ BAD: Using `<Form>` for every small interaction that doesn't need a redirect.
✅ GOOD:
```tsx
import { useFetcher } from "@remix-run/react";
export default function LikeButton({ postId }: { postId: string }) {
  const fetcher = useFetcher();
  const isLiking = fetcher.state === "submitting";
  return (
    <fetcher.Form method="post" action="/api/like">
      <input type="hidden" name="postId" value={postId} />
      <button type="submit" disabled={isLiking}>
        {isLiking ? "Liking..." : "Like"}
      </button>
    </fetcher.Form>
  );
}
```

## 6. Type Safety with TypeScript

**Always use TypeScript with Remix.** Leverage `typeof loader` and `typeof action` for type-safe data access, ensuring robust code and improved developer experience.

❌ BAD:
```tsx
// No types, relies on runtime checks
function MyComponent() {
  const data = useLoaderData(); // data is 'any'
  // ...
}
```

✅ GOOD:
```tsx
import { json, useLoaderData } from "@remix-run/node";
export async function loader() {
  return json({ message: "Hello", count: 42 });
}
export default function MyComponent() {
  const data = useLoaderData<typeof loader>(); // data is { message: string; count: number; }
  return <p>{data.message} {data.count}</p>;
}
```

## 7. Security

**Implement rate-limiting for critical endpoints.** Protect your `action` and `resource` routes from abuse. This is typically done at the edge (Cloudflare, Vercel) or with server-side middleware.

❌ BAD: No protection for public-facing mutation endpoints.
✅ GOOD:
```ts
// app/routes/signup.ts
import { json, ActionFunctionArgs } from "@remix-run/node";
import { rateLimit } from "~/utils/rate-limiter.server"; // Custom server utility

export async function action({ request }: ActionFunctionArgs) {
  await rateLimit(request, "signup", { limit: 5, window: 60 }); // 5 requests per minute
  // ... signup logic
  return json({ success: true });
}
```