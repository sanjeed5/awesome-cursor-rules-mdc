---
description: This guide provides definitive, opinionated best practices for writing robust, secure, and performant GitHub Actions workflows. Follow these rules to build maintainable CI/CD pipelines.
globs: **/*
---
# github-actions Best Practices

GitHub Actions is the backbone of modern CI/CD. Adhering to these best practices ensures your workflows are efficient, secure, and maintainable. This guide is your definitive reference for building high-quality pipelines.

## 1. Workflow Design & Code Organization

Structure your workflows for clarity, reusability, and efficiency.

### 1.1 Use Reusable Workflows and Composite Actions

Avoid duplication. Abstract common sequences into reusable workflows or composite actions.

**❌ BAD: Duplicated Steps**
```yaml
# .github/workflows/build-frontend.yml
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npm run build

# .github/workflows/build-backend.yml
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4 # Duplicated setup
        with: { node-version: '20' }
      - run: npm ci # Duplicated install
      - run: npm run build:backend # Different build step
```

**✅ GOOD: Reusable Workflow**
```yaml
# .github/workflows/reusable-build.yml
on:
  workflow_call:
    inputs:
      build-script: { required: true, type: string }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: ${{ inputs.build-script }}

# .github/workflows/frontend.yml
jobs:
  build:
    uses: ./.github/workflows/reusable-build.yml
    with: { build-script: 'npm run build:frontend' }

# .github/workflows/backend.yml
jobs:
  build:
    uses: ./.github/workflows/reusable-build.yml
    with: { build-script: 'npm run build:backend' }
```

### 1.2 Name Jobs and Steps Consistently

Use clear, descriptive names for jobs and steps. This improves readability and debugging.

**❌ BAD: Vague Naming**
```yaml
jobs:
  job1:
    steps:
      - run: echo "hello"
      - run: npm test

# logs will show 'Run echo "hello"' and 'Run npm test'
```

**✅ GOOD: Descriptive Naming**
```yaml
jobs:
  lint-and-test:
    name: Lint & Test Code
    steps:
      - name: Run ESLint
        run: npm run lint
      - name: Run Unit Tests
        run: npm test
```

### 1.3 Employ Matrix Strategies for Broad Testing

Test across multiple OSes, Node.js versions, or other configurations efficiently.

**❌ BAD: Separate Jobs for Each Variant**
```yaml
jobs:
  test-node-18-ubuntu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with: { node-version: '18' }
      - run: npm test
  test-node-20-ubuntu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm test
```

**✅ GOOD: Single Job with Matrix**
```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        node-version: ['18', '20']
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/setup-node@v4
        with: { node-version: ${{ matrix.node-version }} }
      - run: npm test
```

### 1.4 Set Explicit Concurrency Groups

Prevent simultaneous runs on the same environment, especially for deployments.

**❌ BAD: Overlapping Deployments**
```yaml
jobs:
  deploy:
    environment: production
    # Multiple pushes could trigger concurrent deployments
```

**✅ GOOD: Enforced Concurrency**
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    concurrency: production_deployment # Only one 'production_deployment' can run at a time
    steps:
      - name: Deploy to Production
        run: deploy-script.sh
```

## 2. Performance Considerations

Optimize your workflows for speed and cost-efficiency.

### 2.1 Cache Dependencies

Significantly reduce build times by caching `node_modules`, `pip` packages, etc.

**❌ BAD: Reinstalling Dependencies Every Run**
```yaml
steps:
  - run: npm ci # Always downloads everything
```

**✅ GOOD: Smart Caching**
```yaml
steps:
  - uses: actions/cache@v4
    with:
      path: ~/.npm # Or node_modules, ~/.cache/pip, etc.
      key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      restore-keys: |
        ${{ runner.os }}-node-
  - run: npm ci
```

## 3. Code Quality & Maintainability

Enforce high standards and ensure supply chain integrity.

### 3.1 Run Linters, Formatters, and Static Analysis Early

Catch issues before they merge. Complement with local pre-commit hooks.

**❌ BAD: No Early Checks**
```yaml
jobs:
  build:
    steps:
      - run: npm run build # Build fails due to linting errors discovered late
```

**✅ GOOD: Shift-Left Quality Checks**
```yaml
jobs:
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npm run lint # Fail fast
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: lint # Only run tests if lint passes
    steps:
      # ...
```

### 3.2 Pin Third-Party Actions to a Specific SHA or Version Tag

Prevent unexpected changes from breaking your workflows or introducing vulnerabilities.

**❌ BAD: Floating Version Tag**
```yaml
uses: actions/checkout@v4 # v4 could update at any time
```

**✅ GOOD: Specific Version Tag or SHA**
```yaml
uses: actions/checkout@v4.1.1 # Pin to specific patch
# OR (most secure)
uses: actions/checkout@a81bbbf8298bb0ba8a753697672f0999c0179a61 # Pin to specific SHA
```

## 4. Security Hardening & Configuration Management

Protect your secrets and mitigate common attack vectors.

### 4.1 Store All Secrets in GitHub Secrets

Never hardcode sensitive values. Use GitHub's built-in secret management.

**❌ BAD: Hardcoded Secrets**
```yaml
env:
  API_KEY: "sk_test_12345" # Exposed in workflow file
```

**✅ GOOD: GitHub Secrets**
```yaml
env:
  API_KEY: ${{ secrets.MY_API_KEY }} # Securely referenced
```

### 4.2 Apply Principle of Least Privilege to `GITHUB_TOKEN`

Set default `GITHUB_TOKEN` permissions to read-only, then elevate only when necessary for specific jobs.

**❌ BAD: Overly Permissive Default**
```yaml
permissions: write-all # Grants write access to many scopes by default
jobs:
  build:
    # ...
  deploy:
    # ...
```

**✅ GOOD: Granular Permissions**
```yaml
permissions:
  contents: read # Default to read-only for all jobs
jobs:
  build:
    # No additional permissions needed
    steps:
      # ...
  deploy:
    permissions:
      contents: write # Elevate only for this job to push changes
    steps:
      # ...
```

### 4.3 Mask Sensitive Data in Logs

Use `::add-mask::` for any non-GitHub secret sensitive values that might appear in logs.

**❌ BAD: Printing Sensitive Output**
```yaml
- run: echo "Debug info: ${{ env.TEMP_TOKEN }}" # TEMP_TOKEN will be visible
```

**✅ GOOD: Masking Output**
```yaml
- run: echo "::add-mask::${{ env.TEMP_TOKEN }}"
- run: echo "Debug info: ${{ env.TEMP_TOKEN }}" # TEMP_TOKEN will be masked
```

### 4.4 Mitigate Script Injection Attacks

Always pass untrusted input (e.g., from `github.event.pull_request.title`) to scripts via environment variables, not direct interpolation.

**❌ BAD: Direct Interpolation**
```yaml
- run: echo "Title: ${{ github.event.pull_request.title }}" # Vulnerable to `"; rm -rf /"`
```

**✅ GOOD: Intermediate Environment Variable**
```yaml
- name: Check PR Title
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: |
    if [[ "$PR_TITLE" =~ ^feat ]]; then
      echo "Feature PR"
    fi
```

## 5. Common Pitfalls & Gotchas

Be aware of these common issues to avoid debugging headaches.

### 5.1 Debugging with `ACTIONS_STEP_DEBUG`

Enable verbose logging for a specific workflow run to diagnose issues.

**❌ BAD: Guessing at Failures**
```yaml
# Workflow fails, no extra info
```

**✅ GOOD: Enable Debugging**
1.  Go to your repository settings.
2.  Navigate to "Secrets and variables" -> "Actions" -> "Repository secrets".
3.  Add a new repository secret: `ACTIONS_STEP_DEBUG` with value `true`.
4.  Rerun the workflow.
5.  **Remember to delete the secret after debugging!**

### 5.2 Correct Use of Conditional Logic

Use `if` conditions for steps or jobs effectively.

**❌ BAD: Incorrect `if` syntax**
```yaml
- name: Deploy if main
  if: github.ref == 'main' # This is for steps, not jobs
  run: deploy.sh
```

**✅ GOOD: Proper `if` Placement**
```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' # Condition for the entire job
    steps:
      - name: Deploy to Production
        if: success() # Condition for a step
        run: deploy.sh
```

## 6. Testing Approaches

Integrate robust testing into your CI/CD pipeline.

### 6.1 Implement Code Scanning

Use GitHub's built-in code scanning to find security vulnerabilities early.

**❌ BAD: No Automated Security Scans**
```yaml
# Relying solely on manual review for security
```

**✅ GOOD: Integrated Code Scanning**
```yaml
jobs:
  code-scan:
    name: Code Scanning
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with: { languages: javascript }
      - uses: github/codeql-action/autobuild@v3
      - uses: github/codeql-action/analyze@v3
```