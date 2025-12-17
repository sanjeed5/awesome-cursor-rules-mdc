---
description: This guide provides opinionated, actionable best practices for writing maintainable, secure, and performant CircleCI configurations and orbs.
globs: **/*
---
# circleci Best Practices

CircleCI is the backbone of our CI/CD. Adhere to these guidelines to ensure our pipelines are efficient, secure, and maintainable.

## 1. Code Organization & Reusability

### 1.1. Author and Consume Orbs for Reusability
Encapsulate common logic into orbs. This promotes DRY principles and consistent patterns across projects.

*   **Descriptive Naming**: Use clear `namespace/orb-name` slugs.
    ❌ BAD: `myorg/plugin`, `myorg/ci-tool`
    ✅ GOOD: `myorg/docker-publish`, `myorg/aws-deploy`
*   **Semantic Versioning**: Lock to specific orb versions.
    ❌ BAD: `myorg/aws-deploy@volatile`
    ✅ GOOD: `myorg/aws-deploy@1.2.3`
*   **Comprehensive Descriptions**: Every orb component (commands, jobs, executors, parameters) *must* have a clear `description`.

    ```yaml
    # .circleci/orb.yml
    commands:
      my-command:
        description: "Utilize this command to echo Hello to a step in the UI."
        steps:
          - run: echo "Hello"
    ```

### 1.2. Reuse Commands, Executors, and Jobs
Within your `config.yml` or orbs, define reusable components to avoid duplication.

❌ BAD:
```yaml
jobs:
  build-and-test:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - run: npm install
      - run: npm test
  another-job:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - run: npm install # Duplication
      - run: npm run build
```

✅ GOOD:
```yaml
# .circleci/config.yml
version: 2.1
commands:
  install_deps:
    description: "Install Node.js dependencies."
    steps:
      - run: npm install

jobs:
  build-and-test:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - install_deps
      - run: npm test
  build-app:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - install_deps
      - run: npm run build
```

### 1.3. Use Pass-Through Parameters in Jobs
When a job uses a command or executor that accepts parameters, expose those parameters at the job level for flexibility.

```yaml
# .circleci/orb.yml (partial)
executors:
  default:
    docker:
      - image: 'cimg/node:<<parameters.tag>>'
    parameters:
      tag:
        type: string
        default: '18.17'

jobs:
  test:
    description: "Simple job to test Node.js application."
    parameters:
      version: # Pass-through parameter for the executor's tag
        type: string
        default: '18.17'
    executor:
      name: default
      tag: << parameters.version >> # Pass parameter to executor
    steps:
      - checkout
      - run: npm test
```

## 2. Security & Configuration Management

### 2.1. Store Secrets in Contexts
**Never hard-code secrets or expose them directly in `config.yml`.** Use CircleCI Contexts for sensitive environment variables.

❌ BAD:
```yaml
jobs:
  deploy:
    steps:
      - run: echo "Deploying with API_KEY=$MY_API_KEY" # MY_API_KEY defined directly in config or as unmanaged env var
```

✅ GOOD:
```yaml
# In CircleCI UI, define a Context named 'my-app-secrets' with MY_API_KEY
jobs:
  deploy:
    docker:
      - image: cimg/base:stable
    steps:
      - run: echo "Deploying with API_KEY=$MY_API_KEY" # MY_API_KEY is loaded from 'my-app-secrets' context
workflows:
  build-and-deploy:
    jobs:
      - deploy:
          context: my-app-secrets # Apply the context to the job
```

## 3. Performance Optimization

### 3.1. Leverage Caching for Dependencies
Cache dependencies and build artifacts to significantly speed up pipeline execution.

❌ BAD:
```yaml
jobs:
  build:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - run: npm install # Downloads dependencies every time
      - run: npm test
```

✅ GOOD:
```yaml
jobs:
  build:
    docker:
      - image: cimg/node:18.17.1
    steps:
      - checkout
      - restore_cache: # Restore cache based on package-lock.json
          keys:
            - v1-dependencies-{{ checksum "package-lock.json" }}
            - v1-dependencies- # Fallback to latest cache
      - run: npm install
      - save_cache: # Save cache if dependencies changed
          paths:
            - node_modules
          key: v1-dependencies-{{ checksum "package-lock.json" }}
      - run: npm test
```

### 3.2. Parallelize Jobs and Tests
Distribute work across multiple containers to reduce overall build time, especially for large test suites.

```yaml
jobs:
  run-tests:
    docker:
      - image: cimg/node:18.17.1
    parallelism: 4 # Run 4 containers concurrently
    steps:
      - checkout
      - run:
          name: Run parallel tests
          command: |
            # Example: split tests using a tool like 'circleci tests split'
            npx jest --json --outputFile=test-results.json --coverage --testResultsProcessor="jest-junit" \
              $(circleci tests glob "src/**/*.test.js" | circleci tests split --split-by=timings)
      - store_test_results:
          path: test-results.json
```

### 3.3. Consolidate `run` Steps
Minimize UI noise and improve readability by combining related shell commands into a single `run` step.

❌ BAD:
```yaml
steps:
  - run: pip install example
  - run: example login $MY_TOKEN
  - run: example deploy my-app
```

✅ GOOD:
```yaml
steps:
  - run:
      name: Deploy application
      command: |
        pip install example
        example login $MY_TOKEN
        example deploy my-app
```

## 4. Robust Testing & Quality Gates

### 4.1. Implement Comprehensive Testing
Automate unit, integration, and security (SAST/DAST) tests. Enforce these tests as gatekeepers before deployment.

```yaml
workflows:
  build-test-deploy:
    jobs:
      - unit-test
      - integration-test:
          requires: [unit-test]
      - security-scan:
          requires: [integration-test]
      - deploy-staging:
          requires: [security-scan]
      - hold-for-prod:
          type: approval
          requires: [deploy-staging]
      - deploy-production:
          requires: [hold-for-prod]
```

### 4.2. Serverless CI/CD Considerations
For serverless applications, include specific steps for packaging, Infrastructure as Code (IaC) validation, and canary deployments.

```yaml
jobs:
  serverless-deploy:
    docker:
      - image: cimg/python:3.9
    steps:
      - checkout
      - run: pip install serverless # Install Serverless Framework
      - run: serverless deploy --stage dev # Deploy to dev
      - run: serverless deploy --stage prod --type canary # Canary deployment
```

## 5. Common Pitfalls & Best Practices

### 5.1. Conditionally Use `sudo`
Check if the user is already root before adding `sudo` to commands.

❌ BAD:
```yaml
steps:
  - run: sudo apt-get update
```

✅ GOOD:
```yaml
steps:
  - run:
      name: Update apt packages
      command: |
        if [[ $EUID == 0 ]]; then export SUDO=""; else export SUDO="sudo"; fi
        $SUDO apt-get update
```

### 5.2. Pin Docker Image Tags
Always use specific, stable Docker image tags, not `latest`.

❌ BAD: `cimg/node:latest`
✅ GOOD: `cimg/node:18.17.1`

### 5.3. Use Admonitions in Documentation
For orb descriptions or comments in `config.yml`, use CircleCI's admonition syntax for emphasis.

```yaml
# NOTE: This command requires AWS credentials configured in a context.
# CAUTION: Deploying to production will trigger a canary release.
```