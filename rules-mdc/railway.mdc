---
description: This guide provides definitive best practices for developing and deploying applications on Railway, focusing on stateless services, reproducible builds with Nixpacks, secure configuration, and optimal performance.
globs: **/*
---
# railway Best Practices

Railway is a powerful PaaS for modern applications. Adhere to these guidelines to ensure your services are performant, reliable, and easily deployable.

## 1. Code Organization and Structure

Organize your codebase for clarity and Railway's build system. For monorepos, explicitly define service directories.

### ✅ GOOD: Explicit Service Definition (Monorepo)

Use `railway.json` to define services, ensuring Railway builds and deploys the correct subdirectories.

```json
// railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "services": [
    {
      "name": "backend",
      "path": "./backend",
      "buildCommand": "npm install && npm run build",
      "startCommand": "npm start"
    },
    {
      "name": "frontend",
      "path": "./frontend",
      "buildCommand": "npm install && npm run build",
      "startCommand": "npm start"
    }
  ]
}
```

## 2. Common Patterns and Anti-patterns

Embrace statelessness and graceful shutdowns for scalable, resilient services.

### ❌ BAD: Stateful Services

Storing session data or temporary files directly on the server instance. This breaks horizontal scaling and leads to data loss on redeployments or scale-downs.

```typescript
// app.ts (BAD)
let inMemoryCache = {}; // Will be lost on scale-down/redeploy
```

### ✅ GOOD: Stateless Design

Persist state in external services (databases, Redis, S3). Your application must be able to start, stop, and scale without losing data or affecting other instances.

```typescript
// app.ts (GOOD)
import { createClient } from 'redis'; // Use an external Redis instance

const redisClient = createClient({
  url: process.env.REDIS_URL // Connect to Railway-provisioned Redis
});

async function getFromCache(key: string) {
  await redisClient.connect();
  const value = await redisClient.get(key);
  await redisClient.disconnect();
  return value;
}
```

### ✅ GOOD: Graceful Shutdowns (Node.js Example)

Handle `SIGTERM` signals to ensure your application cleans up resources (e.g., close database connections, flush logs) before termination, preventing data corruption or dropped requests.

```typescript
// server.ts
import express from 'express';
// Assume connectToDatabase and disconnectFromDatabase exist
import { connectToDatabase, disconnectFromDatabase } from './db'; 

const app = express();
const PORT = process.env.PORT || 3000;
let server: any;

async function startServer() {
  await connectToDatabase();
  server = app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

function handleShutdown() {
  console.log('SIGTERM received. Shutting down gracefully...');
  server.close(async () => {
    console.log('HTTP server closed.');
    await disconnectFromDatabase();
    process.exit(0);
  });

  // Force close if server hasn't closed after a timeout
  setTimeout(() => {
    console.error('Forcing shutdown after timeout.');
    process.exit(1);
  }, 10000); // 10 seconds
}

process.on('SIGTERM', handleShutdown);
process.on('SIGINT', handleShutdown); // Also handle Ctrl+C for local dev

startServer();
```

## 3. Performance Considerations

Optimize for low latency and efficient resource usage.

### ✅ GOOD: Private Networking for Internal Services

Always use Railway's private network for communication between services (e.g., app to database, app to cache). This reduces latency and eliminates egress costs.

```typescript
// database.ts
// ❌ BAD: Public endpoint (introduces latency, egress costs)
// const DB_HOST = 'your-db-public-url.railway.app';

// ✅ GOOD: Private network endpoint (fast, free internal traffic)
const DB_HOST = process.env.PGHOST || 'postgres.railway.internal'; // Or redis.railway.internal, mongo.railway.internal
const DB_PORT = process.env.PGPORT || 5432;
```

### ✅ GOOD: Region Selection

Deploy your application in a region geographically close to your primary user base to minimize latency.

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that lead to deployment failures or runtime issues.

### ❌ BAD: Missing Health Checks

Railway needs a `/health` endpoint to determine if your service is running correctly. Without it, deployments may fail or services restart unnecessarily.

### ✅ GOOD: Implement a Basic Health Endpoint

```typescript
// app.ts
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});
```

## 5. Configuration Management

Treat your configuration as code and manage secrets securely.

### ✅ GOOD: Config-as-Code with `railway.yaml`

Define environment variables, secrets, and service configurations directly in version control. This ensures reproducibility across environments.

```yaml
# railway.yaml
name: My Awesome Project
services:
  - name: web
    repo: .
    buildCommand: npm install && npm run build
    startCommand: npm start
    variables:
      NODE_ENV: production
    secrets:
      API_KEY: ${{ secrets.API_KEY }} # Reference secrets securely
```

### ✅ GOOD: Use Nixpacks for Reproducible Builds

Leverage Nixpacks for declarative, reproducible builds. Only use custom Dockerfiles if Nixpacks cannot infer your build process or if you have highly specialized requirements.

```toml
# nixpacks.toml (Example for a Python app needing specific system packages)
[phases.setup]
nixPkgs = ["python3", "postgresql"] # Add system-level dependencies here
```

## 6. Environment Variables

Manage environment variables and secrets securely.

### ❌ BAD: Hardcoding Secrets or Committing `.env`

Never commit sensitive information directly to your repository. This is a major security vulnerability.

```typescript
// config.ts (BAD)
const API_KEY = "super-secret-key-123";
```

### ✅ GOOD: Use Railway Secrets

Store secrets directly in Railway's dashboard or via `railway variables set --project <id> API_KEY=...` and reference them in `railway.yaml`. Access them via `process.env` at runtime.

```typescript
// config.ts (GOOD)
const API_KEY = process.env.API_KEY; // Railway injects this securely
```

## 7. Logging

Centralize logs to Railway's platform for easy debugging and monitoring.

### ✅ GOOD: Log to Standard Output (Stdout/Stderr)

Railway automatically collects logs written to `stdout` and `stderr`. Use structured logging (JSON) for easier parsing and integration with external log analysis tools.

```typescript
// logger.ts
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
});

// ✅ GOOD: Structured JSON log for easy parsing
logger.info({ event: 'user_login', userId: 'abc-123', ip: '192.168.1.1' }, 'User logged in');

// ❌ BAD: Unstructured log (hard to parse programmatically)
// console.log('User logged in');
```

## 8. Testing Approaches

Integrate automated testing into your CI/CD pipeline.

### ✅ GOOD: Automated CI/CD with GitHub Actions

Run unit, integration, and end-to-end tests automatically on every push or pull request *before* deploying to Railway. This ensures code quality and prevents regressions.

```yaml
# .github/workflows/deploy.yml
name: Deploy to Railway

on:
  push:
    branches:
      - main

jobs:
  build_and_deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      - name: Install dependencies
        run: npm ci
      - name: Run tests
        run: npm test # ✅ Ensure tests pass before deployment
      - name: Deploy to Railway
        if: success() # Only deploy if tests pass
        uses: railwayapp/github-action@v3
        with:
          token: ${{ secrets.RAILWAY_TOKEN }}
          service: backend # Or use railway.json for multiple services
```