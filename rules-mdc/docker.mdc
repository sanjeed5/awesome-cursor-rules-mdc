---
description: Definitive guidelines for writing efficient, secure, and maintainable Dockerfiles and Docker Compose configurations, ensuring fast builds and reliable deployments.
globs: **/Dockerfile,**/docker-compose*.{yml,yaml}
---
# docker Best Practices

Docker is the cornerstone of modern container-first development. Treat your `Dockerfile` and `docker-compose.yml` as critical source code. These rules ensure your images are fast, secure, reproducible, and aligned with modern DevOps practices.

## 1. Optimize for Multi-Stage Builds

Always use multi-stage builds. This pattern isolates build-time dependencies from runtime, drastically reducing final image size and attack surface.

❌ BAD: Single-stage build with all dependencies
```dockerfile
FROM node:20
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["node", "dist/server.js"]
```

✅ GOOD: Multi-stage build for production
```dockerfile
# Stage 1: Build application artifacts
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production # Install only production dependencies for build
COPY . .
RUN npm run build

# Stage 2: Create minimal runtime image
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules # Copy only necessary runtime modules
USER node # Run as non-root user
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

## 2. Choose Minimal, Trusted Base Images

Start with the smallest possible base image from a trusted source (Docker Official Images, Verified Publishers). Alpine variants are often the best choice for minimal footprints.

❌ BAD: Large, generic base image
```dockerfile
FROM ubuntu:latest # Too large, many unnecessary packages and vulnerabilities
```

✅ GOOD: Minimal, specific base image
```dockerfile
FROM node:20-alpine # Official, small, specific to Node.js
# Or for static content:
FROM nginx:alpine
```

## 3. Leverage `.dockerignore`

Always use a `.dockerignore` file to exclude irrelevant files and directories from your build context. This significantly speeds up builds and prevents sensitive or unnecessary files from being copied into the image.

❌ BAD: No `.dockerignore` (sends `node_modules`, `.git`, `.env`, etc., to the daemon)

✅ GOOD: Comprehensive `.dockerignore`
```
# .dockerignore
.git
.gitignore
node_modules
npm-debug.log
Dockerfile
docker-compose*.yml
.env
README.md
tmp/
```

## 4. Optimize Layer Caching

Order your `Dockerfile` instructions from least to most frequently changing. Docker caches layers, so placing stable instructions first maximizes cache hits and speeds up subsequent builds.

❌ BAD: Copying all source code before installing dependencies
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . . # Invalidates cache if any file changes, forcing npm install every time
RUN npm install
```

✅ GOOD: Copying dependency files first
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./ # Only changes when dependencies change; cache hit if unchanged
RUN npm ci # Cache hit if package.json is unchanged
COPY . . # Only invalidates this layer if source code changes
```

## 5. Run as a Non-Root User

Never run your application as `root` inside the container. Create a dedicated non-root user and switch to it using the `USER` instruction to enforce the principle of least privilege.

❌ BAD: Running as root (default behavior)
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "app.js"] # Runs as root
```

✅ GOOD: Running as a non-root user
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install && \
    chown -R node:node /app # Ensure non-root user has permissions for app directory
USER node # Switch to non-root user
CMD ["node", "app.js"]
```

## 6. Distinguish `ARG` and `ENV`

Use `ARG` for build-time variables that do not persist in the final image. Use `ENV` for runtime environment variables that your application needs.

❌ BAD: Using `ENV` for build-only values or `ARG` for runtime values
```dockerfile
FROM ubuntu
ENV BUILD_VERSION=1.0.0 # Persists in image, potentially unnecessary
ARG PORT=8080 # Not available at runtime unless explicitly passed
```

✅ GOOD: Correct usage of `ARG` and `ENV`
```dockerfile
FROM node:20-alpine
ARG NODE_ENV=production # Build-time variable, not in final image
ENV APP_PORT=3000 # Runtime variable, available to the application
WORKDIR /app
COPY . .
RUN if [ "$NODE_ENV" = "production" ]; then npm ci --production; else npm ci; fi
EXPOSE ${APP_PORT}
CMD ["node", "app.js"]
```

## 7. Implement Health Checks

Define `HEALTHCHECK` instructions in your `Dockerfile` to allow Docker and orchestration tools to verify if your containerized service is still working correctly. This is crucial for reliable deployments.

❌ BAD: No health check, relying solely on process exit
```dockerfile
CMD ["node", "app.js"]
```

✅ GOOD: Robust health check
```dockerfile
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1
CMD ["node", "app.js"]
```

## 8. Log to `stdout`/`stderr`

Containerized applications must write logs to `stdout` and `stderr`. This allows Docker and orchestration tools to easily capture, aggregate, and manage logs without needing complex in-container log rotation or file access.

❌ BAD: Writing logs to files inside the container
```javascript
// app.js
fs.appendFileSync('/var/log/app.log', 'Application started\n');
```

✅ GOOD: Writing logs to `stdout`/`stderr`
```javascript
// app.js
console.log('Application started');
console.error('An error occurred');
```

## 9. Use Docker Compose for Local Development

For multi-service applications, always use `docker-compose.yml` to define and run your services. This ensures a consistent, reproducible development environment for all team members.

❌ BAD: Manually running multiple `docker run` commands for interconnected services
```bash
docker run -p 8080:80 my-web-app
docker run -p 5432:5432 my-database
```

✅ GOOD: Defining services in `docker-compose.yml`
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgres://user:password@db:5432/mydb
    depends_on:
      - db
  db:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

## 10. Manage Secrets Securely with Docker Compose

Never hardcode sensitive information (API keys, passwords) directly in `Dockerfile` or `docker-compose.yml`. For local development, use Docker Compose's `secrets` feature.

❌ BAD: Hardcoding secrets directly in the compose file
```yaml
# docker-compose.yml
services:
  web:
    environment:
      API_KEY: "supersecretkey123" # BAD! This is committed to source control.
```

✅ GOOD: Using Docker Compose secrets
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: my-web-app:latest
    secrets:
      - api_key_file
    environment:
      API_KEY_PATH: /run/secrets/api_key_file # Application reads from this path

secrets:
  api_key_file:
    file: ./secrets/api_key.txt # Create this file locally, ensure it's .gitignored
```
*(Note: For production, use a dedicated secret management system like Kubernetes Secrets, AWS Secrets Manager, or HashiCorp Vault.)*