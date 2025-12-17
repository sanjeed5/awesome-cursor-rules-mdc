---
description: This guide outlines essential Heroku best practices, focusing on the Twelve-Factor App methodology, runtime principles, and modern cloud-native development patterns for building robust, scalable, and maintainable applications.
globs: **/*
---
# Heroku Best Practices

Developing on Heroku requires adherence to specific patterns to leverage its Platform-as-a-Service (PaaS) strengths. This guide focuses on the **Twelve-Factor App** methodology and Heroku's runtime principles to ensure your applications are scalable, resilient, and maintainable.

## 1. Code Organization and Structure

Maintain a clean, version-controlled codebase and clearly define your application's processes.

### 1.1. `Procfile` for Process Types

Explicitly declare your application's process types using a `Procfile`. This allows Heroku to run different types of dynos (web, worker, etc.) and scale them independently.

❌ **BAD**: Relying on default `web` process or running multiple concerns in one dyno.
```
# No Procfile, or a single 'web' process doing everything
web: npm start
```

✅ **GOOD**: Separate web and worker processes.
```Procfile
web: node src/server.js
worker: node src/worker.js
release: node src/db-migrate.js
```
*   **Context**: Use `web` for HTTP traffic, `worker` for background jobs, and `release` for one-off tasks like database migrations *before* the app starts.

### 1.2. `app.json` for Declarative Setup

Use `app.json` for Heroku Button deployments, Review Apps, and to declare add-ons and environment variables.

❌ **BAD**: Manual setup of add-ons and config vars for each environment.
```
# No app.json, requiring manual configuration
```

✅ **GOOD**: Declarative `app.json` for consistent deployments.
```json
// app.json
{
  "name": "My Awesome App",
  "description": "A description of my awesome app.",
  "repository": "https://github.com/myorg/my-awesome-app",
  "keywords": ["node", "express", "heroku"],
  "addons": [
    "heroku-postgresql",
    "heroku-redis"
  ],
  "env": {
    "NODE_ENV": {
      "description": "The environment the app is running in.",
      "value": "production"
    },
    "LOG_LEVEL": {
      "description": "Logging verbosity.",
      "value": "info"
    }
  },
  "scripts": {
    "postdeploy": "npm run db:seed"
  },
  "formation": {
    "web": {
      "quantity": 1,
      "size": "standard-1x"
    },
    "worker": {
      "quantity": 1,
      "size": "standard-1x"
    }
  }
}
```
*   **Context**: `app.json` ensures consistency across environments and simplifies onboarding for new developers.

## 2. Common Patterns and Anti-patterns

Adhere to the Twelve-Factor App principles for robust cloud-native applications.

### 2.1. Configuration in the Environment

Store all configuration that varies between deploys (database credentials, API keys) in environment variables, not in code.

❌ **BAD**: Hardcoding secrets or committing config files.
```javascript
// config.js
const DB_URL = "postgres://user:pass@host:port/db"; // BAD: hardcoded secret
const API_KEY = "super-secret-key"; // BAD: hardcoded secret
```

✅ **GOOD**: Fetch configuration from environment variables.
```javascript
// config.js
const DB_URL = process.env.DATABASE_URL;
const API_KEY = process.env.EXTERNAL_API_KEY;

if (!DB_URL || !API_KEY) {
  console.error("Missing critical environment variables!");
  process.exit(1);
}
```
*   **Context**: Heroku automatically injects add-on credentials (e.g., `DATABASE_URL`, `REDIS_URL`) as environment variables. Always use `process.env.PORT` for the server port.

### 2.2. Stateless Processes

Ensure your application processes are stateless and share-nothing. Any data that needs to persist must be stored in a backing service.

❌ **BAD**: Storing user sessions or uploaded files on the local filesystem.
```javascript
// server.js
const express = require('express');
const session = require('express-session');
const fs = require('fs');

const app = express();
app.use(session({ secret: 'my-secret', resave: false, saveUninitialized: false })); // BAD: default session store is in-memory
app.post('/upload', (req, res) => {
  fs.writeFileSync(`/tmp/${req.file.name}`, req.file.buffer); // BAD: writing to ephemeral filesystem
});
```

✅ **GOOD**: Use external backing services for state.
```javascript
// server.js
const express = require('express');
const session = require('express-session');
const RedisStore = require('connect-redis').default;
const { createClient } = require('redis');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3'); // Or any cloud storage

const app = express();

// Use Redis for session storage
const redisClient = createClient({ url: process.env.REDIS_URL });
redisClient.connect().catch(console.error);
app.use(session({
  store: new RedisStore({ client: redisClient }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false
}));

// Upload files to S3
const s3 = new S3Client({ region: process.env.AWS_REGION });
app.post('/upload', async (req, res) => {
  const command = new PutObjectCommand({
    Bucket: process.env.S3_BUCKET_NAME,
    Key: req.file.name,
    Body: req.file.buffer
  });
  await s3.send(command);
  res.send('File uploaded to S3');
});
```
*   **Context**: Heroku's filesystem is ephemeral and unique to each dyno. Data written to it will be lost on dyno restart or scale events.

## 3. Performance Considerations

Optimize for Heroku's dyno model to achieve high performance and scalability.

### 3.1. Background Jobs

Offload long-running tasks to worker dynos to keep web dynos responsive.

❌ **BAD**: Performing heavy computations or sending emails directly in a web request.
```javascript
// web_route.js
app.post('/process-data', async (req, res) => {
  const result = await performHeavyComputation(req.body.data); // Blocks web dyno
  await sendEmail(req.user.email, result); // Blocks web dyno
  res.json({ status: 'completed', result });
});
```

✅ **GOOD**: Queue background jobs for worker dynos.
```javascript
// web_route.js
const { Queue } = require('bullmq'); // Example with BullMQ (requires Redis)
const myQueue = new Queue('my-processing-queue', { connection: { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT } });

app.post('/process-data', async (req, res) => {
  await myQueue.add('process-task', { data: req.body.data, userEmail: req.user.email });
  res.json({ status: 'processing', message: 'Task queued successfully.' });
});

// worker.js (running on a worker dyno)
const { Worker } = require('bullmq');
const worker = new Worker('my-processing-queue', async job => {
  const result = await performHeavyComputation(job.data.data);
  await sendEmail(job.data.userEmail, result);
}, { connection: { host: process.env.REDIS_HOST, port: process.env.REDIS_PORT } });
```
*   **Context**: Web dynos should respond quickly (ideally < 500ms). Use a queueing add-on (e.g., Heroku Redis with BullMQ/Celery) for any task that takes longer.

### 3.2. Database Optimization

Use Heroku Postgres efficiently. Optimize queries, add indexes, and consider connection pooling.

❌ **BAD**: Opening a new database connection for every request, N+1 queries.
```javascript
// In a Node.js Express app without connection pooling
app.get('/users/:id', async (req, res) => {
  const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
  await client.connect();
  const user = await client.query('SELECT * FROM users WHERE id = $1', [req.params.id]);
  // ... potentially more queries for related data without proper joins
  await client.end();
  res.json(user.rows[0]);
});
```

✅ **GOOD**: Use a connection pool and optimize queries.
```javascript
// In a Node.js Express app with pg-pool
const { Pool } = require('pg');
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false } // Required for Heroku Postgres
});

app.get('/users/:id', async (req, res) => {
  const user = await pool.query('SELECT * FROM users WHERE id = $1', [req.params.id]);
  res.json(user.rows[0]);
});
```
*   **Context**: Heroku Postgres is a managed service. Focus on application-level database performance. Always use SSL when connecting to Heroku Postgres.

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to ensure smooth operation on Heroku.

### 4.1. Binding to `PORT` Environment Variable

Your web process *must* listen on the port specified by the `PORT` environment variable. Heroku dynamically assigns this port.

❌ **BAD**: Hardcoding a port (e.g., `3000`).
```javascript
// server.js
const PORT = 3000; // BAD: hardcoded port
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
```

✅ **GOOD**: Use `process.env.PORT`.
```javascript
// server.js
const PORT = process.env.PORT || 5000; // Use Heroku's PORT, fallback for local dev
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
```
*   **Context**: Heroku routes traffic to your web dyno on a dynamically assigned port. Failing to bind to `process.env.PORT` will cause your app to crash.

### 4.2. Handling Graceful Shutdown

Heroku sends a `SIGTERM` signal to dynos before shutting them down. Your app should catch this signal and perform cleanup.

❌ **BAD**: Abruptly exiting on `SIGTERM`.
```javascript
// server.js
process.on('SIGTERM', () => {
  console.log('Received SIGTERM. Exiting immediately.');
  process.exit(0); // BAD: no time for cleanup
});
```

✅ **GOOD**: Implement graceful shutdown.
```javascript
// server.js
const server = app.listen(PORT, () => console.log(`Listening on ${PORT}`));

process.on('SIGTERM', () => {
  console.log('Received SIGTERM. Shutting down gracefully...');
  server.close(() => {
    console.log('HTTP server closed.');
    // Close database connections, flush logs, etc.
    // Ensure all pending requests are handled or rejected.
    process.exit(0);
  });

  // Force close if server doesn't close within a timeout
  setTimeout(() => {
    console.error('Forcefully shutting down after timeout.');
    process.exit(1);
  }, 10000); // 10 seconds grace period
});
```
*   **Context**: Heroku gives your dyno 10 seconds to shut down. This is crucial for preventing data loss and ensuring a smooth user experience during deploys or scaling events.

## 5. Testing Approaches

Integrate testing into your Heroku development workflow, especially with Heroku Flow.

### 5.1. Automated CI/CD with Heroku Flow

Leverage Heroku Flow for automated testing, linting, and deployment. This ensures code quality and prevents regressions.

❌ **BAD**: Manual testing and deployment, no automated checks.
```
# No CI/CD setup, manual deploys to production
```

✅ **GOOD**: Configure Heroku CI for automated testing on every push.
```json
// app.json (excerpt for Heroku CI)
{
  "scripts": {
    "test": "npm test"
  },
  "environments": {
    "test": {
      "addons": ["heroku-postgresql:mini"],
      "env": {
        "NODE_ENV": "test"
      }
    }
  }
}
```
*   **Context**: Heroku CI runs your `test` script (defined in `package.json` or `app.json`) on a fresh dyno for every push to GitHub, providing immediate feedback.

### 5.2. Review Apps for Feature Testing

Use Review Apps to automatically spin up a disposable Heroku app for every pull request, enabling easy testing and collaboration.

❌ **BAD**: Developers testing features locally or on a shared staging environment.
```
# No Review Apps, PRs are reviewed without a dedicated deploy
```

✅ **GOOD**: Enable Review Apps in Heroku Dashboard.
```json
// app.json (ensure this file exists and is configured)
{
  "name": "My Awesome App",
  "description": "A description of my awesome app.",
  "repository": "https://github.com/myorg/my-awesome-app",
  "keywords": ["node", "express", "heroku"],
  "addons": [
    "heroku-postgresql:hobby-dev"
  ],
  "env": {
    "NODE_ENV": "development"
  }
}
```
*   **Context**: Review Apps provide an isolated, production-like environment for each feature, making code reviews more effective and catching environment-specific bugs early.