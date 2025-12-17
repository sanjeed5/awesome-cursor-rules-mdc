---
description: Definitive guidelines for building, deploying, and operating applications on AWS ECS, emphasizing immutable containers, secure secrets management, and robust operational patterns.
globs: **/*
---
# aws-ecs Best Practices

This guide outlines the definitive best practices for developing and deploying applications on Amazon Elastic Container Service (ECS). Adhere to these principles to ensure your applications are secure, performant, and maintainable.

## 1. Code Organization and Structure

### 1.1. Immutable Container Images

Always build immutable container images. Any change to code or dependencies *must* trigger a new image build and deployment. This ensures reproducibility and simplifies rollbacks.

❌ BAD: Modifying container contents at runtime (e.g., downloading dependencies on startup).
```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package.json .
RUN npm install # Installs dependencies during build
COPY . .
CMD ["sh", "-c", "npm install && npm start"] # Re-installs on every run!
```

✅ GOOD: Bake all dependencies into the image during the build process.
```dockerfile
# Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev # Install production dependencies
COPY . .
RUN npm run build # Build application assets

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist # Copy built assets
EXPOSE 3000
CMD ["node", "dist/main.js"] # Run the built application
```

### 1.2. Single Process Per Container

Each container must run only one foreground application process. This allows ECS to effectively manage the container's lifecycle, health checks, and resource allocation.

❌ BAD: Using process managers like `supervisord` to run multiple applications.
```dockerfile
# Dockerfile
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
# supervisord.conf might start Nginx, a Python app, and a cron job.
```

✅ GOOD: Decouple services into separate containers and task definitions.
```dockerfile
# Dockerfile for a single application
CMD ["python", "app.py"]
```
*Context*: If services are tightly coupled (e.g., a web server and a sidecar agent that *must* run on the same host), use a multi-container task definition. Otherwise, deploy them as separate services.

## 2. Common Patterns and Anti-patterns

### 2.1. Graceful Shutdown Handling (SIGTERM)

Your application must gracefully handle `SIGTERM` signals. ECS sends `SIGTERM` to allow your application to finish in-flight requests and clean up before `SIGKILL` is sent.

❌ BAD: Ignoring `SIGTERM`, leading to abrupt shutdowns and potential data loss.
```javascript
// Node.js example
process.on('SIGINT', () => { /* Handles Ctrl+C, but not SIGTERM from orchestrator */ });
// No SIGTERM handler, process exits immediately on SIGTERM.
```

✅ GOOD: Implement a `SIGTERM` handler to allow for graceful exit.
```javascript
// Node.js example
process.on('SIGTERM', async () => {
  console.log('SIGTERM received. Initiating graceful shutdown...');
  // Stop accepting new connections
  server.close(() => {
    console.log('HTTP server closed.');
    // Perform any cleanup (e.g., close DB connections, flush logs)
    process.exit(0);
  });
  // Force exit after a timeout if cleanup takes too long
  setTimeout(() => {
    console.error('Graceful shutdown timed out. Forcing exit.');
    process.exit(1);
  }, 10000); // 10 seconds timeout
});
```

### 2.2. Secrets Management

Never hard-code secrets into container images or environment variables directly in Dockerfiles. Use AWS Secrets Manager or AWS Systems Manager Parameter Store, injected at runtime via the task definition.

❌ BAD: Hardcoding secrets or passing them as plain environment variables.
```dockerfile
# Dockerfile
ENV DB_PASSWORD=mysecretpassword
```
```json
// Task Definition (bad practice)
{
  "environment": [
    { "name": "DB_PASSWORD", "value": "mysecretpassword" }
  ]
}
```

✅ GOOD: Reference secrets from Secrets Manager in your task definition.
```json
// Task Definition
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest",
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:my-db-secret-abcdef:DB_PASSWORD::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/my-app",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "ecs"
        },
        "secretOptions": [
          {
            "name": "awslogs-secret-token",
            "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:my-log-auth-token:token::"
          }
        ]
      }
    }
  ]
}
```

### 2.3. Logging

Containers must write logs to `stdout` and `stderr`. Decouple log handling from your application code using the `awslogs` driver.

❌ BAD: Writing logs to local files within the container.
```python
# Python app
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
```

✅ GOOD: Log to `stdout`/`stderr` and configure `awslogs` in the task definition.
```python
# Python app
import logging
logging.basicConfig(level=logging.INFO) # Logs to stderr by default
logging.info("Application started.")
```
```json
// Task Definition (excerpt)
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest",
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/my-app",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## 3. Performance Considerations

### 3.1. Resource Limits

Always define CPU and memory limits in your task definitions. This prevents resource exhaustion on the underlying host and ensures fair scheduling.

❌ BAD: Omitting resource limits, leading to potential noisy neighbor issues or task failures.
```json
// Task Definition (no limits)
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest"
      // No cpu or memory specified
    }
  ]
}
```

✅ GOOD: Explicitly set CPU and memory limits.
```json
// Task Definition
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest",
      "cpu": 256,   // 0.25 vCPU
      "memory": 512 // 512 MB
    }
  ],
  "cpu": "256", // Total CPU for the task
  "memory": "512" // Total Memory for the task
}
```
*Context*: For Fargate, `cpu` and `memory` must be defined at the task level, and container limits must be less than or equal to the task limits.

### 3.2. Fargate First

Prioritize AWS Fargate for most workloads. It abstracts away server management, offering a serverless compute engine that scales automatically and provides strong task isolation. Only use EC2 launch types if you have specialized hardware requirements (e.g., GPUs), need privileged access, or have specific licensing needs.

## 4. Common Pitfalls and Gotchas

### 4.1. IAM Task Roles

Always assign a dedicated IAM role to each task (`taskRoleArn`). This enforces the principle of least privilege, granting containers only the permissions they need to interact with other AWS services.

❌ BAD: Relying on the EC2 instance profile for permissions (if using EC2 launch type), which grants all tasks on that instance the same, potentially overly broad, permissions.
```json
// Task Definition (missing taskRoleArn)
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest"
    }
  ]
}
```

✅ GOOD: Define a specific `taskRoleArn` for the task.
```json
// Task Definition
{
  "family": "my-app-task",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecs-my-app-task-role",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest"
    }
  ]
}
```
*Context*: `executionRoleArn` is required for tasks to pull images from ECR and send logs to CloudWatch. `taskRoleArn` is for your *application's* permissions.

### 4.2. Container Image Tagging

Use immutable, unique tags for container images, preferably based on the Git SHA or a semantic version. Never rely solely on the `latest` tag for production deployments. Enable immutable image tags in ECR.

❌ BAD: Using `latest` or mutable tags in production.
```dockerfile
# Dockerfile
# ...
docker build -t my-ecr-repo/my-app:latest .
```

✅ GOOD: Tag with Git SHA and enable ECR immutable tags.
```bash
# CI/CD Pipeline
GIT_SHA=$(git rev-parse --short HEAD)
docker build -t my-ecr-repo/my-app:"$GIT_SHA" .
docker push my-ecr-repo/my-app:"$GIT_SHA"
```
*Context*: Configure your ECR repository to enforce immutable tags.

## 5. Testing Approaches

### 5.1. Health Checks

Implement robust health checks (HTTP, TCP, or command) in your task definitions. This allows ECS to identify unhealthy containers and replace them, ensuring service availability.

❌ BAD: Omitting health checks, leading to traffic being routed to unhealthy containers.
```json
// Task Definition (no health check)
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest"
    }
  ]
}
```

✅ GOOD: Configure an HTTP health check.
```json
// Task Definition
{
  "containerDefinitions": [
    {
      "name": "my-app",
      "image": "my-ecr-repo/my-app:latest",
      "portMappings": [
        { "containerPort": 3000, "hostPort": 3000 }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"],
        "interval": 30,  // Check every 30 seconds
        "timeout": 5,    // Timeout after 5 seconds
        "retries": 3,    // 3 retries before marking unhealthy
        "startPeriod": 60 // Allow 60 seconds for startup
      }
    }
  ]
}
```

### 5.2. Automated CI/CD with AWS Copilot

Automate image builds, testing, and deployments using AWS Copilot or similar CI/CD pipelines (e.g., AWS CodePipeline, GitHub Actions). This ensures consistent, repeatable deployments and adheres to the Well-Architected Framework.

❌ BAD: Manual deployments or custom scripts that are not version-controlled.

✅ GOOD: Use AWS Copilot for opinionated, automated deployments.
```bash
# Initialize a new service
copilot init

# Deploy to a test environment
copilot deploy --name my-app --env test

# Deploy to production
copilot deploy --name my-app --env prod
```
*Context*: Copilot simplifies the creation of ECS services, including load balancers, CI/CD pipelines, and service discovery.