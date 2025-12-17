---
description: Definitive guidelines for building, deploying, and managing cloud-native applications on DigitalOcean, focusing on secure, scalable, and cost-efficient practices.
globs: **/*
---
# digitalocean Best Practices

This guide outlines the definitive best practices for developing and deploying applications on DigitalOcean. Adhering to these rules ensures your projects are secure, performant, scalable, and maintainable, leveraging DigitalOcean's managed services and cloud-native patterns.

## 1. VPC-Centric Networking (Security & Simplicity)

**ALWAYS** isolate your infrastructure within Virtual Private Clouds (VPCs). This is the cornerstone of secure and efficient communication between your services. Connect all compute resources (Droplets, Kubernetes nodes, Functions) and managed services (Databases) within the same VPC.

*   **Connect Managed Databases via VPC**: Use the private connection string and add the VPC's CIDR block as the *only* trusted source. This drastically reduces the attack surface and simplifies IP management.

    ❌ BAD: Public database connections, individual IP trusted sources.
    ```sql
    -- Public connection string, exposed to the internet
    psql "postgresql://doadmin:PASSWORD@db-public-endpoint.db.ondigitalocean.com:25060/defaultdb?sslmode=require"
    ```

    ✅ GOOD: Private VPC connection, CIDR block trusted source.
    ```sql
    -- Private connection string, only accessible within the VPC
    psql "postgresql://doadmin:PASSWORD@db-private-endpoint.db.ondigitalocean.com:25060/defaultdb?sslmode=require"
    ```
    *Configuration*: In DigitalOcean Control Panel, for your database, add your VPC's CIDR (e.g., `10.108.0.0/20`) to Trusted Sources.

*   **All Compute in VPC**: Ensure all Droplets, Kubernetes nodes, and Functions are launched within the same VPC as your managed services.

## 2. Microservices-First Architecture

Embrace a microservices architecture for agility, scalability, and resilience. DigitalOcean provides excellent platforms for this.

*   **Managed Kubernetes (DOKS) for Orchestration**: For complex, containerized applications requiring fine-grained control and high availability.
*   **App Platform for Rapid CI/CD**: For web apps, APIs, and static sites needing fast, automated deployments with integrated CI/CD. Ideal for AI-assisted development workflows (e.g., with Claude Code).
*   **Premium CPU-Optimized Droplets**: For high-performance, resource-intensive workloads that benefit from dedicated CPU.

    ❌ BAD: Monolithic application on a single Droplet.
    ```dockerfile
    # Dockerfile for a large, multi-service monolith
    FROM node:18-alpine
    WORKDIR /app
    COPY . .
    RUN npm install
    EXPOSE 8080 5432 # Exposing database directly from app container
    CMD ["npm", "start"]
    ```

    ✅ GOOD: Decoupled services, deployed to appropriate platforms.
    ```dockerfile
    # Dockerfile for a single microservice (e.g., API)
    FROM node:20-alpine AS builder
    WORKDIR /app
    COPY package*.json ./
    RUN npm install --production
    COPY . .
    RUN npm run build # If applicable

    FROM node:20-alpine
    WORKDIR /app
    COPY --from=builder /app/node_modules ./node_modules
    COPY --from=builder /app/dist ./dist # Or similar for compiled assets
    EXPOSE 8080
    CMD ["node", "dist/index.js"] # Or your main entry point
    ```
    *Deployment*: Deploy the API to App Platform or DOKS, use a Managed PostgreSQL database.

## 3. Leverage Managed Services

**ALWAYS** prefer DigitalOcean's managed services over self-hosting when available. They provide automated backups, failover, scaling, and security patching, reducing operational overhead.

*   **Managed Databases**: Use PostgreSQL, MySQL, Redis, or MongoDB.
*   **Spaces (Object Storage)**: For static assets, backups, and media files. Integrate with CDN for performance.
*   **Container Registry**: Store private Docker images securely.
*   **Load Balancers**: Distribute traffic and provide SSL termination.

    ❌ BAD: Self-hosting PostgreSQL on a Droplet.
    ```bash
    # Manual setup, no automated failover or backups
    sudo apt update && sudo apt install postgresql postgresql-contrib
    # ... manual configuration, backup scripts, monitoring, etc.
    ```

    ✅ GOOD: Provision a Managed PostgreSQL Database.
    ```terraform
    resource "digitalocean_database_cluster" "primary_db" {
      name       = "my-app-db"
      engine     = "pg"
      version    = "15"
      size       = "db-s-2vcpu-4gb"
      region     = "nyc3"
      node_count = 1 # Start with 1, add standby nodes for HA
      vpc_uuid   = digitalocean_vpc.main.id # Ensure VPC is defined
    }
    ```

## 4. Disciplined Container Builds

Follow Docker's best practices to create small, secure, and efficient container images.

*   **Multi-stage Builds**: Separate build-time dependencies from runtime dependencies.
*   **Minimal Base Images**: Use `alpine` variants or other minimal images.
*   **Trusted Base Images**: Prefer Docker Official Images or Verified Publisher images.
*   **`.dockerignore`**: Exclude unnecessary files from the build context.
*   **Rebuild Images Often**: Ensure dependencies are up-to-date with security patches.

    ❌ BAD: Single-stage build with development tools, no `.dockerignore`.
    ```dockerfile
    FROM node:18 # Large base image
    WORKDIR /app
    COPY . . # Copies everything, including .git, node_modules (if present)
    RUN npm install # Installs dev dependencies too
    CMD ["npm", "start"]
    ```

    ✅ GOOD: Multi-stage build, minimal base, `.dockerignore` in place.
    ```dockerfile
    # .dockerignore example:
    # node_modules
    # .git
    # .env
    # Dockerfile
    # README.md

    # Build stage
    FROM node:20-alpine AS builder
    WORKDIR /app
    COPY package*.json ./
    RUN npm install --omit=dev # Only install production dependencies
    COPY . .
    RUN npm run build # Compile application if needed

    # Production stage
    FROM node:20-alpine
    WORKDIR /app
    COPY --from=builder /app/node_modules ./node_modules
    COPY --from=builder /app/dist ./dist # Copy compiled app
    EXPOSE 8080
    CMD ["node", "dist/index.js"]
    ```

## 5. Automated Workflows (CI/CD & IaC)

Automate everything possible. This reduces human error, increases deployment speed, and ensures consistency.

*   **CI/CD Pipelines**: Use DigitalOcean App Platform's built-in CI/CD or integrate with GitHub Actions for DOKS deployments.
*   **Infrastructure as Code (IaC)**: Manage all DigitalOcean resources with Terraform. This ensures your infrastructure is version-controlled, reproducible, and auditable.

    ❌ BAD: Manual resource creation via Control Panel.
    ```bash
    # Click-ops in the UI
    # Create Droplet -> Add Database -> Configure Load Balancer...
    ```

    ✅ GOOD: Define infrastructure with Terraform.
    ```terraform
    # main.tf
    resource "digitalocean_project" "my_project" {
      name        = "My Awesome Project"
      description = "Infrastructure for my awesome app."
      purpose     = "Web Application"
    }

    resource "digitalocean_vpc" "main" {
      name     = "my-app-vpc"
      region   = "nyc3"
      ip_range = "10.10.0.0/20"
    }

    resource "digitalocean_kubernetes_cluster" "main_cluster" {
      name    = "my-app-cluster"
      region  = "nyc3"
      version = "1.28.2-do.0"
      vpc_uuid = digitalocean_vpc.main.id
      node_pool {
        name       = "worker-pool"
        size       = "s-2vcpu-4gb"
        node_count = 2
      }
    }
    ```

## 6. Performance & Cost Optimization

Continuously monitor and optimize your resources.

*   **Autoscaling**: Implement autoscaling for Droplet pools and Kubernetes node pools to match demand and save costs.
*   **Load Balancers**: Use DigitalOcean Load Balancers to distribute traffic and handle SSL.
*   **Regular Architecture Reviews**: Leverage DigitalOcean's free Solutions Engineer reviews to identify inefficiencies and security gaps.
*   **Monitor Usage & Set Alerts**: Track CPU, memory, disk I/O, network traffic, and billing.

    ❌ BAD: Over-provisioned Droplets running at low utilization.
    ```bash
    # Manually scaled up to a large Droplet "just in case"
    digitalocean droplet create --size s-8vcpu-16gb --image ubuntu-22-04 --region nyc3 --name my-server
    ```

    ✅ GOOD: Use autoscaling groups or right-sized resources.
    ```terraform
    resource "digitalocean_kubernetes_node_pool" "autoscaling_pool" {
      cluster_id = digitalocean_kubernetes_cluster.main_cluster.id
      name       = "autoscaled-workers"
      size       = "s-2vcpu-4gb"
      min_nodes  = 1
      max_nodes  = 5
      auto_scale = true
    }
    ```

## 7. Testing & Observability

Integrate testing into your CI/CD and ensure robust monitoring.

*   **Automated Testing**: Include unit, integration, and end-to-end tests in your CI pipeline.
*   **Staging Environments**: Deploy to dedicated staging environments (e.g., via App Platform's branch deployments) before production.
*   **Logging & Monitoring**: Centralize logs and use DigitalOcean Monitoring for metrics and alerts.

    ❌ BAD: Manual testing only, no dedicated staging environment.
    ```bash
    # Deploy directly to production after local testing
    git push origin main # Triggers production deploy
    ```

    ✅ GOOD: Automated tests, staging environment, and feature branches.
    ```yaml
    # .github/workflows/ci-cd.yml (simplified)
    name: CI/CD Pipeline

    on:
      push:
        branches:
          - main
          - feature/*

    jobs:
      build-and-test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
          - name: Build Docker Image
            run: docker build -t my-app:${{ github.sha }} .
          - name: Run Unit Tests
            run: docker run my-app:${{ github.sha }} npm test

      deploy-staging:
        needs: build-and-test
        if: startsWith(github.ref, 'refs/heads/feature/')
        runs-on: ubuntu-latest
        steps:
          # ... deploy to App Platform staging environment
          - name: Deploy to App Platform Staging
            run: doctl apps create-deployment --app-id ${{ secrets.APP_PLATFORM_STAGING_ID }} --image my-app:${{ github.sha }}

      deploy-production:
        needs: build-and-test
        if: github.ref == 'refs/heads/main'
        runs-on: ubuntu-latest
        steps:
          # ... deploy to App Platform production environment
          - name: Deploy to App Platform Production
            run: doctl apps create-deployment --app-id ${{ secrets.APP_PLATFORM_PROD_ID }} --image my-app:${{ github.sha }}
    ```