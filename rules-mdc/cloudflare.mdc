---
description: Definitive guidelines for building secure, performant, and maintainable applications on Cloudflare's developer platform, emphasizing tiny bundles, edge-first design, and robust security.
globs: **/*
---
# cloudflare Best Practices

Cloudflare's developer platform thrives on speed, security, and global distribution. Our focus is on building tiny, observable, and secure bundles that leverage the edge. This guide outlines the definitive best practices for our team.

## 1. Code Organization and Structure

**Prioritize Workers & Pages for all new applications.** Use the official `wrangler` CLI for project scaffolding and deployment.

*   **Project Initialization:**
    *   ✅ GOOD: Always start with `wrangler generate`.
        ```bash
        npx wrangler generate my-worker --type=module --ts
        # For full-stack apps with Pages:
        npx create-cloudflare@latest my-fullstack-app --framework next
        ```
*   **Monorepos:** For projects with multiple Workers or Pages apps, use `pnpm` or `yarn workspaces`.
    *   ✅ GOOD: Centralize dependencies and build scripts.
        ```json
        // package.json
        {
          "workspaces": ["apps/*", "packages/*"]
        }
        ```
*   **Environment Variables:** Manage secrets securely via `wrangler.toml` or the Cloudflare dashboard. Use `.dev.vars` for local development.
    *   ❌ BAD: Storing sensitive data directly in code or `.env` files that aren't `.dev.vars`.
    *   ✅ GOOD: Define variables in `wrangler.toml` and reference them in your Worker.
        ```toml
        # wrangler.toml
        name = "my-worker"
        main = "src/index.ts"
        compatibility_date = "2025-12-01"

        [vars]
        MY_API_KEY = "your-api-key-value" # For non-sensitive defaults or local testing

        [secrets] # For sensitive production secrets
        # MY_DB_URL = "set via `wrangler secret put MY_DB_URL`"
        ```
    *   ✅ GOOD: Use `.dev.vars` for local secrets, and integrate with CLI tools.
        ```bash
        # .dev.vars
        DATABASE_URL="postgresql://user:pass@localhost:5432/mydb"
        # Use with Prisma:
        dotenv -e .dev.vars -- npx prisma migrate dev
        ```

## 2. Common Patterns and Anti-patterns

**Design for the edge: stateless Workers, stateful Durable Objects.**

*   **Edge-first Design:** Workers are stateless and globally distributed. Avoid in-memory state that needs to persist across requests or instances.
    *   ❌ BAD: Relying on global variables for user sessions or shared data.
        ```typescript
        let requestCount = 0; // This will reset or be inconsistent
        export default {
          async fetch(request, env, ctx) {
            requestCount++;
            return new Response(`Requests: ${requestCount}`);
          },
        };
        ```
    *   ✅ GOOD: Use KV for simple key-value storage, D1 for relational data, or Durable Objects for strong consistency and real-time state.
        ```typescript
        // Using KV for simple counters
        export default {
          async fetch(request, env, ctx) {
            let count = parseInt(await env.REQUEST_COUNTER.get("total_requests") || "0");
            await env.REQUEST_COUNTER.put("total_requests", String(count + 1));
            return new Response(`Requests: ${count + 1}`);
          },
        };
        ```
*   **Serverless SQL (D1, Hyperdrive):** Use D1 for new, natively serverless relational databases. For existing regional databases, use Hyperdrive for edge acceleration.
    *   ✅ GOOD: D1 for new projects.
        ```typescript
        // src/index.ts
        import { DrizzleD1Database, drizzle } from 'drizzle-orm/d1';
        interface Env { D1_BINDING: D1Database; }

        export default {
          async fetch(request: Request, env: Env) {
            const db = drizzle(env.D1_BINDING);
            const result = await db.select().from(users).all();
            return new Response(JSON.stringify(result));
          },
        };
        ```
*   **Object Storage (R2):** Always use R2 for large file storage. Benefit from zero egress fees and global distribution.
    *   ✅ GOOD: Efficiently store and retrieve assets.
        ```typescript
        // src/index.ts
        interface Env { R2_BUCKET: R2Bucket; }
        export default {
          async fetch(request: Request, env: Env) {
            const object = await env.R2_BUCKET.get("my-file.txt");
            if (object) return new Response(object.body);
            return new Response("Not found", { status: 404 });
          },
        };
        ```
*   **Cloudflare One & Tunnels:** Securely expose internal services without opening firewall ports. Integrate SSO for private apps.
    *   ✅ GOOD: Use `cloudflared` for secure access to Azure, AWS, GCP, or on-prem resources.
        ```bash
        # On your private VM
        cloudflared tunnel create Azure-01
        # Configure config.yml and start service
        ```
    *   ✅ GOOD: Present private web apps on Cloudflare-owned domains with SSO.
        ```yaml
        # Cloudflare Access Policy (example)
        name: "Require SSO for Internal App"
        application: "internal-app.cloudflare.com"
        rules:
          - action: "allow"
            identity_provider_ids: ["your-sso-provider-id"]
            emails: ["@your-domain.com"]
        ```

## 3. Performance Considerations

**Bundle size and cold start latency are paramount.**

*   **Bundle Size Optimization:** Keep Worker bundles as small as possible. Tree-shake dependencies aggressively.
    *   ❌ BAD: Importing large, unoptimized libraries.
    *   ✅ GOOD: Use `esbuild` for bundling and ensure minimal imports.
        ```typescript
        // Example: Import only specific functions
        import { get } from 'lodash-es/get'; // Instead of import * as _ from 'lodash';
        ```
*   **Prisma Optimization:** For ORMs, use Prisma with `engineType: "client"` and an edge-compatible driver adapter. This avoids large Rust binaries.
    *   ❌ BAD: Default Prisma setup with Rust query engines in Workers.
    *   ✅ GOOD: Configure `schema.prisma` and use an adapter.
        ```prisma
        // schema.prisma
        generator client {
          provider = "prisma-client-js"
          engineType = "client" # Crucial for Workers
        }
        datasource db {
          provider = "postgresql"
          url      = env("DATABASE_URL")
        }
        ```
        ```typescript
        // src/index.ts
        import { PrismaClient } from '@prisma/client/edge';
        import { withAccelerate } from '@prisma/extension-accelerate'; // For Prisma Accelerate

        const prisma = new PrismaClient({
          datasourceUrl: env.DATABASE_URL,
        }).$extends(withAccelerate()); // If using Accelerate
        ```
*   **Caching:** Leverage Cloudflare's CDN for static assets and API responses.
    *   ✅ GOOD: Set appropriate `Cache-Control` headers.
        ```typescript
        return new Response(JSON.stringify(data), {
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'public, max-age=3600, s-maxage=86400',
          },
        });
        ```

## 4. Common Pitfalls and Gotchas

**Understand the Workers runtime: single-threaded, event-driven, no Node.js APIs.**

*   **Blocking I/O:** Workers are single-threaded and event-driven. All I/O operations *must* be `await`ed.
    *   ❌ BAD: Synchronous file system operations or long-running CPU-bound tasks.
    *   ✅ GOOD: Asynchronous operations for network requests, KV, D1, etc.
        ```typescript
        // All I/O should be awaited
        const response = await fetch('https://api.example.com');
        const data = await env.KV_NAMESPACE.get('key');
        ```
*   **Missing Security Policies:** Always configure WAF, Bot Fight Mode, and Rate Limiting.
    *   ❌ BAD: Deploying public APIs without edge security.
    *   ✅ GOOD: Implement granular rate limiting for login endpoints, API abuse, and `cf_clearance` cookie reuse.
        ```yaml
        # Example Rate Limiting Rule (conceptual, configured in Cloudflare Dashboard)
        # Match: (http.request.uri.path eq "/login" and http.request.method eq "POST")
        # Counting characteristics: IP
        # Rate: 4 requests / 1 minute
        # Action: Managed Challenge
        ```

## 5. Testing Approaches

**Test early, test often, and test at the edge.**

*   **Unit Testing:** Use standard JavaScript/TypeScript testing frameworks (`Vitest`, `Jest`) for pure functions and business logic.
    *   ✅ GOOD: Isolate and test small components.
        ```typescript
        // my-logic.test.ts
        import { expect, test } from 'vitest';
        import { calculateDiscount } from './my-logic';
        test('calculates discount correctly', () => {
          expect(calculateDiscount(100, 0.1)).toBe(90);
        });
        ```
*   **Integration Testing (Miniflare):** Use `Miniflare` for local emulation of the Workers runtime, including bindings (KV, D1, R2). This is critical for testing Worker logic with dependencies.
    *   ✅ GOOD: Simulate the Cloudflare environment locally.
        ```typescript
        // worker.test.ts
        import { Miniflare } from 'miniflare';
        import { test, expect, beforeAll, afterAll } from 'vitest';

        let mf: Miniflare;
        beforeAll(async () => {
          mf = new Miniflare({
            modules: true,
            scriptPath: 'dist/index.mjs', // Your compiled worker
            bindings: { MY_VAR: 'test-value' },
            kvNamespaces: ['REQUEST_COUNTER'],
            d1Databases: ['D1_BINDING'],
            r2Buckets: ['R2_BUCKET'],
          });
        });
        afterAll(async () => await mf.dispose());

        test('Worker responds with correct text', async () => {
          const res = await mf.dispatchFetch('http://localhost/');
          expect(await res.text()).toBe('Hello World!');
        });
        ```
*   **End-to-End Testing:** Deploy to a staging environment (`wrangler deploy --env staging`) as part of your CI/CD pipeline. Use tools like Playwright or Cypress to verify full application flows.
    *   ✅ GOOD: Validate the entire stack in a production-like environment.
        ```bash
        # CI/CD step
        npx wrangler deploy --env staging
        npx playwright test --project=staging
        ```