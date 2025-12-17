---
description: This guide provides opinionated, actionable best practices for structuring, optimizing, and securing GitLab CI/CD pipelines, focusing on modern patterns and avoiding common pitfalls.
globs: **/*
---
# gitlab-ci Best Practices

GitLab CI/CD is the backbone of modern DevOps. This guide ensures your `.gitlab-ci.yml` files are robust, maintainable, performant, and secure. Follow these rules rigorously.

## 1. Code Organization and Structure

Your `.gitlab-ci.yml` is code. Treat it with the same discipline as application code.

### 1.1. Modularize with `include`

Break down large pipelines into smaller, focused YAML files. This improves readability, reusability, and reduces merge conflicts. Keep the root `.gitlab-ci.yml` as an orchestrator.

❌ **BAD: Monolithic `.gitlab-ci.yml`**
```yaml
# .gitlab-ci.yml (too long, hard to navigate)
stages:
  - build
  - test
  - deploy

build_frontend:
  stage: build
  script:
    - npm install
    - npm run build

test_frontend:
  stage: test
  script:
    - npm run test:unit

build_backend:
  stage: build
  script:
    - pip install -r requirements.txt
    - python setup.py build

test_backend:
  stage: test
  script:
    - pytest

deploy_staging:
  stage: deploy
  script:
    - deploy-script --env staging
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
```

✅ **GOOD: Modularized with `include`**
```yaml
# .gitlab-ci.yml (root orchestrator)
include:
  - local: '.gitlab/ci/stages.yml'
  - local: '.gitlab/ci/frontend.yml'
  - local: '.gitlab/ci/backend.yml'
  - local: '.gitlab/ci/deploy.yml'

# .gitlab/ci/stages.yml
stages:
  - build
  - test
  - security
  - deploy

# .gitlab/ci/frontend.yml
.frontend_template: &frontend_template
  image: node:20-alpine
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node-modules
    paths:
      - frontend/node_modules/
    policy: pull-push
  before_script:
    - cd frontend

build_frontend:
  <<: *frontend_template
  stage: build
  script:
    - npm install
    - npm run build
  artifacts:
    paths:
      - frontend/dist/
    expire_in: 1 day

test_frontend:
  <<: *frontend_template
  stage: test
  script:
    - npm run test:unit

# .gitlab/ci/backend.yml
.backend_template: &backend_template
  image: python:3.11-slim-buster
  cache:
    key: ${CI_COMMIT_REF_SLUG}-pip-cache
    paths:
      - backend/.venv/
    policy: pull-push
  before_script:
    - cd backend
    - python -m venv .venv
    - source .venv/bin/activate

build_backend:
  <<: *backend_template
  stage: build
  script:
    - pip install -r requirements.txt
    - python setup.py build
  artifacts:
    paths:
      - backend/dist/
    expire_in: 1 day

test_backend:
  <<: *backend_template
  stage: test
  script:
    - pytest

# .gitlab/ci/deploy.yml
deploy_staging:
  stage: deploy
  image: alpine/git:latest
  script:
    - echo "Deploying to staging..."
    - ./scripts/deploy.sh staging
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
  environment:
    name: staging
    url: https://staging.example.com
```

### 1.2. Use `extends` and YAML Anchors for Reusability

Avoid repetition. Define common configurations once and reuse them across jobs. `extends` is preferred for job-level inheritance, while YAML anchors (`&` and `*`) are excellent for smaller, inline reusable blocks.

❌ **BAD: Repetitive job definitions**
```yaml
build_job_a:
  image: node:20-alpine
  script:
    - npm install
    - npm run build
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node-modules
    paths:
      - node_modules/

build_job_b:
  image: node:20-alpine
  script:
    - npm install
    - npm run build:prod
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node-modules
    paths:
      - node_modules/
```

✅ **GOOD: Reusable with `extends` and anchors**
```yaml
.node_base: &node_base_config
  image: node:20-alpine
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node-modules
    paths:
      - node_modules/
    policy: pull-push

.build_template:
  extends: .node_base
  stage: build
  script:
    - npm install
    - npm run build

build_job_a:
  extends: .build_template

build_job_b:
  extends: .build_template
  script:
    - npm install # Overrides script from .build_template
    - npm run build:prod
```

### 1.3. Consistent Indentation

Always use **2 spaces** for indentation. Never use tabs. Inconsistent indentation is a leading cause of YAML parsing errors.

❌ **BAD: Mixed indentation or incorrect spacing**
```yaml
job_name:
  stage: build
    script: # 4 spaces, inconsistent
      - echo "Hello"
```

✅ **GOOD: Consistent 2-space indentation**
```yaml
job_name:
  stage: build
  script:
    - echo "Hello"
```

## 2. Common Patterns and Anti-patterns

Adopt modern GitLab CI/CD features for robust and efficient pipelines.

### 2.1. Use `rules` for Conditional Job Execution

The `rules` keyword is the definitive way to control when jobs run. It replaces the deprecated `only/except` syntax, offering far greater flexibility and clarity.

❌ **BAD: Deprecated `only/except`**
```yaml
deploy_prod:
  stage: deploy
  script:
    - deploy-to-prod.sh
  only:
    - main
  except:
    - branches@feature/*
```

✅ **GOOD: Modern `rules` syntax**
```yaml
deploy_prod:
  stage: deploy
  script:
    - deploy-to-prod.sh
  rules:
    - if: $CI_COMMIT_BRANCH == "main" && $CI_PIPELINE_SOURCE == "push"
      when: manual # Requires manual approval for production pushes
    - if: $CI_COMMIT_TAG # Run on tags automatically
      when: on_success
    - when: never # Do not run otherwise
```

### 2.2. Leverage `needs` for Explicit Job Dependencies

Use `needs` to define explicit dependencies between jobs. This allows jobs to run in parallel across stages, significantly speeding up pipelines by avoiding strict stage-based ordering.

❌ **BAD: Implicit stage-based dependencies**
```yaml
stages:
  - build
  - test
  - deploy

build_app:
  stage: build
  script:
    - make build
  artifacts:
    paths:
      - build/

test_app: # Waits for all 'build' jobs to complete
  stage: test
  script:
    - make test
```

✅ **GOOD: Explicit `needs` for parallel execution**
```yaml
stages:
  - build
  - test
  - deploy

build_frontend:
  stage: build
  script:
    - npm run build
  artifacts:
    paths:
      - frontend/dist/

build_backend:
  stage: build
  script:
    - make build
  artifacts:
    paths:
      - backend/dist/

test_frontend:
  stage: test
  needs: ["build_frontend"] # Only waits for build_frontend
  script:
    - npm run test:unit

test_backend:
  stage: test
  needs: ["build_backend"] # Only waits for build_backend
  script:
    - make test
```
In this example, `test_frontend` and `test_backend` can run as soon as their respective build jobs are done, potentially in parallel, even if they are in different stages according to the `stages` definition.

### 2.3. Name Jobs and Stages Clearly

Use descriptive, consistent naming conventions. This makes pipelines easier to understand and debug. A good pattern is `stage_name:job_description`.

❌ **BAD: Generic or unclear names**
```yaml
stages:
  - first
  - second
  - third

job1:
  stage: first
  script:
    - echo "..."

deploy:
  stage: third
  script:
    - deploy.sh
```

✅ **GOOD: Descriptive names**
```yaml
stages:
  - build
  - test
  - security
  - deploy

build:frontend:
  stage: build
  script:
    - npm run build

test:unit:backend:
  stage: test
  script:
    - pytest

security:sast:
  stage: security
  script:
    - /analyzer/sast run

deploy:production:
  stage: deploy
  script:
    - deploy-to-prod.sh
```

## 3. Performance Considerations

Optimize your pipelines for speed to provide fast feedback loops.

### 3.1. Implement Strategic Caching

Cache dependencies (`node_modules`, `vendor/`, `pip` caches) and build artifacts that are expensive to regenerate. Use dynamic cache keys to prevent conflicts between branches.

❌ **BAD: No caching or static cache keys**
```yaml
build_job:
  image: node:20-alpine
  script:
    - npm install # Downloads dependencies every time
    - npm run build
```

✅ **GOOD: Dynamic caching**
```yaml
build_job:
  image: node:20-alpine
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node-modules # Cache per branch/tag
    paths:
      - node_modules/
    policy: pull-push # Pull existing cache, push new one
  script:
    - npm install
    - npm run build
```

### 3.2. Use Small, Specific Docker Images

Minimize image size to reduce download times for runners. Use specific versions to ensure reproducibility.

❌ **BAD: Large or generic images**
```yaml
build_job:
  image: ubuntu:latest # Very large, contains many unnecessary tools
  script:
    - apt-get update && apt-get install -y nodejs npm
    - npm install
```

✅ **GOOD: Small, specific images**
```yaml
build_job:
  image: node:20-alpine # Alpine is much smaller than Debian/Ubuntu
  script:
    - npm install
    - npm run build
```

### 3.3. Parallelize Independent Jobs

Identify jobs that don't depend on each other and run them in parallel using `needs` or the `parallel` keyword for matrix builds.

```yaml
# Using 'parallel' for matrix builds
test:
  stage: test
  image: python:3.11-slim-buster
  script:
    - pip install -r requirements.txt
    - pytest
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.9", "3.10", "3.11"]
        OS: ["debian", "alpine"]
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to prevent pipeline failures and security vulnerabilities.

### 4.1. Never Hardcode Secrets

Secrets (API keys, passwords, tokens) must never be committed to your `.gitlab-ci.yml` or repository. Use GitLab CI/CD variables.

❌ **BAD: Hardcoded secret**
```yaml
deploy_prod:
  stage: deploy
  script:
    - export AWS_ACCESS_KEY_ID="AKIAxxxxxxxxxxxxxx" # DANGER!
    - aws s3 sync build/ s3://my-bucket
```

✅ **GOOD: Use GitLab CI/CD variables**
```yaml
deploy_prod:
  stage: deploy
  script:
    - aws s3 sync build/ s3://my-bucket
  variables:
    AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID # Defined in GitLab UI as protected/masked
    AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY # Defined in GitLab UI as protected/masked
```
**Action**: Define `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as **Protected** and **Masked** variables in your project's `Settings > CI/CD > Variables`.

### 4.2. Scope Variables Appropriately

Define variables at the narrowest scope possible (job-level, then global, then project/group/instance). This improves clarity and reduces unintended side effects.

❌ **BAD: Over-scoped variables**
```yaml
variables:
  DEPLOY_ENV: production # This applies to ALL jobs, even build/test

build_job:
  stage: build
  script:
    - echo "Building for ${DEPLOY_ENV}" # Misleading
```

✅ **GOOD: Job-scoped variables**
```yaml
variables:
  DEFAULT_IMAGE: alpine/git:latest # Global default, sensible

build_job:
  stage: build
  image: node:20-alpine # Overrides default
  script:
    - echo "Building..."

deploy_staging:
  stage: deploy
  script:
    - deploy.sh $DEPLOY_ENV
  variables:
    DEPLOY_ENV: staging # Specific to this job
```

### 4.3. Validate Your YAML

Use GitLab's built-in CI Lint tool (available in the CI/CD editor or `https://gitlab.com/<your-group>/<your-project>/-/ci/lint`) to validate your `.gitlab-ci.yml` before committing.

## 5. Configuration Management

Manage your pipeline configuration systematically.

### 5.1. Use CI/CD Components

For highly reusable and shareable pipeline logic across multiple projects, publish CI/CD Components to the CI/CD Catalog. This is the ultimate form of reusability.

```yaml
# .gitlab-ci.yml
include:
  - component: gitlab.com/my-group/my-ci-components/docker-build@1.0.0
    inputs:
      image_name: my-app
      dockerfile_path: ./Dockerfile
```
**Action**: Explore the CI/CD Catalog for existing components or create your own for common tasks.

## 6. Environment Variables

Beyond secrets, environment variables are crucial for dynamic pipeline behavior.

### 6.1. Leverage Predefined CI/CD Variables

GitLab provides many predefined variables (e.g., `$CI_COMMIT_BRANCH`, `$CI_COMMIT_SHORT_SHA`, `$CI_JOB_ID`). Use these for dynamic logic and naming.

```yaml
build_docker_image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t my-app:$CI_COMMIT_SHORT_SHA .
    - docker push my-app:$CI_COMMIT_SHORT_SHA
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

## 7. Logging

Clear and secure logging is essential for debugging and auditing.

### 7.1. Keep Logs Concise and Masked

Ensure sensitive information is masked in job logs. Avoid excessive output that clutters logs.

❌ **BAD: Verbose, unmasked output**
```bash
# In your script.sh
echo "AWS_SECRET_ACCESS_KEY is: $AWS_SECRET_ACCESS_KEY" # Exposes secret!
npm install --verbose # Too much noise
```

✅ **GOOD: Masked and concise output**
```bash
# In your script.sh
# Ensure AWS_SECRET_ACCESS_KEY is configured as a masked variable in GitLab
echo "Starting deployment..."
npm install --loglevel error # Only show errors
```

## 8. Testing Approaches

Integrate comprehensive testing into your pipeline.

### 8.1. Implement Dedicated Test Stages

Always include stages for unit, integration, and end-to-end tests. Make these jobs mandatory for successful pipelines.

```yaml
stages:
  - build
  - test
  - security
  - deploy

test:unit:
  stage: test
  script:
    - npm run test:unit
  allow_failure: false # Must pass

test:integration:
  stage: test
  script:
    - npm run test:integration
  allow_failure: false # Must pass
```

### 8.2. Integrate Security Scans

Embed Static Application Security Testing (SAST), Dependency Scanning, and Container Scanning into your pipeline. These should run early to catch issues quickly.

```yaml
include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml

sast:
  stage: security
  rules:
    - if: $CI_COMMIT_BRANCH == "main" || $CI_MERGE_REQUEST_IID
  allow_failure: true # Initially allow failure, enforce later

dependency_scanning:
  stage: security
  rules:
    - if: $CI_COMMIT_BRANCH == "main" || $CI_MERGE_REQUEST_IID
  allow_failure: true # Initially allow failure, enforce later

container_scanning:
  stage: security
  rules:
    - if: $CI_COMMIT_BRANCH == "main" || $CI_MERGE_REQUEST_IID
  allow_failure: true # Initially allow failure, enforce later
```
**Action**: Start with `allow_failure: true` for security scans, then switch to `false` once your team is ready to address findings proactively.

### 8.3. Lint Shell Scripts

If your pipeline uses shell scripts, lint them with `shellcheck` and format with `shfmt`.

```yaml
shellcheck:
  image: koalaman/shellcheck-alpine:stable
  stage: test
  script:
    - shellcheck scripts/**/*.sh

shfmt:
  image: mvdan/shfmt:v3.2.0-alpine
  stage: test
  script:
    - shfmt -i 2 -ci -d scripts # Check formatting, -d for diff only
```