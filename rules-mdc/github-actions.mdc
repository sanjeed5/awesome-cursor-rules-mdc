---
description: This rule provides comprehensive guidelines for GitHub Actions development, covering best practices, coding standards, performance, security, and testing.  It aims to ensure efficient, reliable, secure, and maintainable workflows.
globs: .github/workflows/*.yml
---
# GitHub Actions Best Practices and Coding Standards

This guide provides comprehensive guidelines for developing efficient, reliable, secure, and maintainable GitHub Actions workflows. It covers various aspects of GitHub Actions development, including code organization, common patterns, performance considerations, security best practices, testing approaches, and tooling.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

-   **Workflows Directory:**  Store all workflow files in the `.github/workflows` directory. This is the standard location recognized by GitHub.
-   **Reusable Actions Directory (Optional):** If you create custom reusable actions, consider storing them in a dedicated directory like `actions/` within your repository.
-   **Scripts Directory (Optional):** For complex workflows, you might have supporting scripts (e.g., shell scripts, Python scripts). Store these in a `scripts/` directory.
-   **Example Directory Structure:**

    
    .github/
    └── workflows/
        ├── main.yml
        ├── deploy.yml
        └── release.yml
    actions/
        ├── my-custom-action/
        │   ├── action.yml
        │   └── index.js
    scripts/
        ├── cleanup.sh
        └── build.py
    

### 1.2 File Naming Conventions

-   **Workflow Files:** Use descriptive and consistent names for workflow files (e.g., `deploy-staging.yml`, `code-analysis.yml`). Avoid generic names like `main.yml` if possible, especially in repositories with multiple workflows.
-   **Action Files:**  Name action files `action.yml` or `action.yaml` to clearly indicate their purpose.
-   **Script Files:**  Use appropriate extensions for scripts (e.g., `.sh` for shell scripts, `.py` for Python scripts).

### 1.3 Module Organization

-   **Reusable Workflows:** Break down complex workflows into smaller, reusable workflows using the `uses:` syntax. This promotes modularity, reduces duplication, and improves maintainability.
-   **Composite Actions:**  For reusable steps within a workflow, consider creating composite actions.  These group multiple steps into a single action.
-   **Modular Scripts:** If you're using scripts, organize them into modules or functions for better readability and reusability.

### 1.4 Component Architecture

-   **Workflow as a Component:** Treat each workflow as a self-contained component responsible for a specific task (e.g., building, testing, deploying).
-   **Separation of Concerns:**  Separate concerns within a workflow. For example, use different jobs for building, testing, and deploying.
-   **Inputs and Outputs:**  Define clear inputs and outputs for reusable workflows and composite actions to improve their composability.

### 1.5 Code Splitting Strategies

-   **Job Splitting:**  Divide a workflow into multiple jobs that run in parallel to reduce overall execution time.
-   **Step Splitting:** Break down long-running steps into smaller, more manageable steps.
-   **Conditional Execution:** Use `if:` conditions to conditionally execute jobs or steps based on specific criteria (e.g., branch name, file changes).

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to GitHub Actions

-   **Fan-out/Fan-in:** Use matrix builds to parallelize testing across different environments, then aggregate the results in a subsequent job.
-   **Workflow Orchestration:** Use reusable workflows to orchestrate complex processes involving multiple steps and dependencies.
-   **Event-Driven Workflows:** Trigger workflows based on specific GitHub events (e.g., push, pull request, issue creation) to automate tasks.
-   **Policy Enforcement:** Implement workflows that enforce coding standards, security policies, or other organizational guidelines.

### 2.2 Recommended Approaches for Common Tasks

-   **Dependency Caching:** Use the `actions/cache` action to cache dependencies (e.g., npm packages, Maven artifacts) to speed up subsequent workflow runs.
-   **Secret Management:** Store sensitive information (e.g., API keys, passwords) as GitHub Secrets and access them in your workflows using the `${{ secrets.SECRET_NAME }}` syntax.  Never hardcode secrets in your workflow files.
-   **Artifact Storage:** Use the `actions/upload-artifact` and `actions/download-artifact` actions to store and retrieve build artifacts (e.g., compiled binaries, test reports).
-   **Environment Variables:** Use environment variables to configure workflows and steps.  Set environment variables at the workflow, job, or step level.
-   **Workflow Status Badges:** Add workflow status badges to your repository's README file to provide a visual indication of the workflow's health.

### 2.3 Anti-patterns and Code Smells to Avoid

-   **Hardcoding Secrets:**  Never hardcode secrets directly in your workflow files. Use GitHub Secrets instead.
-   **Ignoring Errors:**  Don't ignore errors or warnings in your workflows.  Implement proper error handling to ensure workflows fail gracefully.
-   **Overly Complex Workflows:** Avoid creating overly complex workflows that are difficult to understand and maintain.  Break them down into smaller, reusable workflows.
-   **Lack of Testing:**  Don't skip testing your workflows. Implement unit tests, integration tests, and end-to-end tests to ensure they function correctly.
-   **Unnecessary Dependencies:** Avoid including unnecessary dependencies in your workflows. This can increase build times and introduce security vulnerabilities.
-   **Directly Modifying `GITHUB_PATH` or `GITHUB_ENV`:** While these environment variables exist, using the recommended step outputs is preferred for cleaner, more robust interaction with other steps.

### 2.4 State Management Best Practices

-   **Artifacts:**  Use artifacts for persisting files between jobs. Upload at the end of one job, download at the start of another.
-   **Environment Variables:** Define environment variables at the workflow or job level to pass configuration settings between steps.
-   **Outputs:**  Use step outputs to pass data between steps within a job.
-   **GitHub API:** Use the GitHub API to store and retrieve data related to your workflows (e.g., workflow run status, deployment information).
-   **External Databases:** For more complex state management requirements, consider using an external database.

### 2.5 Error Handling Patterns

-   **`if: always()`:** Ensures a step runs even if a previous step failed, useful for cleanup or notification tasks. `if: always()` should be used with caution, as it can mask underlying issues.
-   **`continue-on-error: true`:** Allows a job to continue even if a step fails. This is useful for non-critical steps or when you want to collect information about multiple failures before failing the workflow.
-   **`try...catch...finally` (within Scripts):**  Use `try...catch...finally` blocks in your scripts to handle exceptions and ensure proper cleanup.
-   **Notifications:**  Send notifications (e.g., email, Slack) when workflows fail or succeed to keep stakeholders informed.
-   **Workflow Retries:**  Consider using the `retry:` keyword to automatically retry failed jobs.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Caching:**  Use the `actions/cache` action aggressively to cache dependencies and intermediate build artifacts.
-   **Concurrency:** Use concurrency to prevent multiple workflows from running at the same time. 
-   **Parallel Execution:**  Run jobs in parallel to reduce overall execution time.
-   **Optimized Images:** Optimize images before uploading them to your repository to reduce their size.
-   **Minify Code:** Minify JavaScript and CSS files to reduce their size.

### 3.2 Memory Management

-   **Resource Limits:**  Be aware of the resource limits imposed by GitHub Actions runners.  Monitor memory and CPU usage to prevent workflows from exceeding these limits.
-   **Garbage Collection:**  Ensure that your scripts and actions properly manage memory and avoid memory leaks.
-   **Large Datasets:** If you're processing large datasets, consider using streaming techniques or splitting the data into smaller chunks.

### 3.3 Rendering Optimization

- N/A - Not typically relevant for GitHub Actions workflows themselves, but may be applicable to applications built and deployed by workflows.

### 3.4 Bundle Size Optimization

- N/A - Not typically relevant for GitHub Actions workflows themselves, but may be applicable to applications built and deployed by workflows.

### 3.5 Lazy Loading Strategies

- N/A - Not typically relevant for GitHub Actions workflows themselves, but may be applicable to applications built and deployed by workflows.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

-   **Code Injection:** Prevent code injection by validating all inputs and sanitizing data before using it in scripts or commands.
-   **Secret Exposure:**  Avoid exposing secrets in logs or error messages.  Mask secrets using the `::add-mask::` command.
-   **Third-Party Actions:**  Carefully vet third-party actions before using them in your workflows.  Pin actions to specific versions or commits to prevent unexpected changes.
-   **Privilege Escalation:**  Run workflows with the least privileges necessary to perform their tasks.
-   **Workflow Command Injection:** Be cautious when dynamically constructing commands.  If possible, use parameters or environment variables instead of concatenating strings.

### 4.2 Input Validation

-   **Validate Inputs:** Validate all inputs to your workflows and actions to prevent malicious data from being processed.
-   **Data Sanitization:** Sanitize data before using it in scripts or commands to prevent code injection vulnerabilities.
-   **Regular Expressions:** Use regular expressions to validate the format of inputs.

### 4.3 Authentication and Authorization Patterns

-   **GitHub Tokens:** Use GitHub tokens to authenticate with the GitHub API.  Grant tokens the minimum necessary permissions.
-   **Service Accounts:** Use service accounts to authenticate with external services.  Store service account credentials as GitHub Secrets.
-   **Role-Based Access Control (RBAC):** Implement RBAC to control access to your workflows and actions.

### 4.4 Data Protection Strategies

-   **Encryption:** Encrypt sensitive data at rest and in transit.
-   **Data Masking:** Mask sensitive data in logs and error messages.
-   **Data Retention:**  Establish a data retention policy to ensure that sensitive data is not stored indefinitely.

### 4.5 Secure API Communication

-   **HTTPS:** Use HTTPS for all API communication.
-   **TLS:** Use TLS encryption to protect data in transit.
-   **API Keys:** Protect API keys and other credentials. Store them as GitHub Secrets and use them securely in your workflows.
-   **Rate Limiting:** Implement rate limiting to prevent abuse of your APIs.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

-   **Test Reusable Actions:** Unit test your custom reusable actions to ensure they function correctly.
-   **Test Scripts:** Unit test your scripts to ensure they handle different inputs and edge cases correctly.
-   **Mock Dependencies:** Use mocking to isolate units of code and test them in isolation.

### 5.2 Integration Testing

-   **Test Workflow Integration:** Integrate test your workflows to ensure that all components work together correctly.
-   **Test API Integrations:** Test your integrations with external APIs to ensure they function correctly.
-   **Test Database Integrations:** Test your integrations with databases to ensure data is read and written correctly.

### 5.3 End-to-end Testing

-   **Full Workflow Tests:** Run end-to-end tests to verify that your workflows function correctly from start to finish.
-   **Simulate Real-World Scenarios:** Simulate real-world scenarios to ensure that your workflows can handle different situations.

### 5.4 Test Organization

-   **Dedicated Test Directory:**  Create a dedicated `tests/` directory for your tests.
-   **Test Naming Conventions:**  Follow consistent naming conventions for your test files and functions.
-   **Test Suites:** Organize your tests into test suites based on functionality or component.

### 5.5 Mocking and Stubbing

-   **Mock External Services:** Mock external services to isolate your tests from external dependencies.
-   **Stub Functions:** Stub functions to control the behavior of dependencies during testing.
-   **Mock GitHub API:** Mock the GitHub API to test your workflows without making real API calls.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

-   **Incorrect Syntax:** YAML syntax can be tricky. Use a linter or validator to catch syntax errors.
-   **Incorrect Indentation:** Indentation is crucial in YAML. Use consistent indentation throughout your workflow files.
-   **Missing Permissions:**  Grant workflows the necessary permissions to access resources (e.g., repository contents, secrets).
-   **Typos in Secrets:** Double-check the names of your secrets to avoid typos.
-   **Not Pinning Action Versions:**  Always pin actions to specific versions or commits to prevent unexpected changes.

### 6.2 Edge Cases to Be Aware Of

-   **Rate Limits:** Be aware of GitHub API rate limits. Implement retry logic to handle rate limit errors.
-   **Concurrent Workflow Runs:** Handle concurrent workflow runs gracefully to avoid conflicts.
-   **Network Issues:** Implement error handling to handle network issues and transient errors.
-   **Large File Sizes:** Be aware of the maximum file sizes supported by GitHub Actions.

### 6.3 Version-Specific Issues

-   **Action Compatibility:** Ensure that your actions are compatible with the version of GitHub Actions you are using.
-   **Runner Images:** Be aware of the changes in runner images and update your workflows accordingly.

### 6.4 Compatibility Concerns

-   **Cross-Platform Compatibility:** Ensure that your workflows are compatible with different operating systems (e.g., Linux, Windows, macOS).
-   **Browser Compatibility:** If your workflows involve web applications, test them in different browsers.

### 6.5 Debugging Strategies

-   **Workflow Logs:** Examine workflow logs to identify errors and warnings.
-   **Debugging Actions:** Use debugging actions to inspect the state of your workflows.
-   **Step-by-Step Debugging:**  Insert `echo` statements or debugging actions to trace the execution of your workflows step by step.
-   **Local Testing:** Use tools like `act` to test your workflows locally before pushing them to GitHub.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

-   **VS Code with GitHub Actions Extension:**  Use VS Code with the GitHub Actions extension for syntax highlighting, code completion, and validation.
-   **GitHub CLI:** Use the GitHub CLI to interact with the GitHub API from your workflows.
-   **`act`:** Use `act` to test your workflows locally.
-   **YAML Linter:** Use a YAML linter to catch syntax errors in your workflow files.

### 7.2 Build Configuration

-   **`.github/workflows/`:** Place all workflow files in this directory.
-   **`action.yml`:** For reusable actions, define their metadata in this file.

### 7.3 Linting and Formatting

-   **YAML Lint:** Use a YAML linting tool to enforce consistent formatting and catch syntax errors.
-   **Shellcheck:** Use Shellcheck to lint your shell scripts.
-   **Prettier:** Use Prettier to format your JavaScript and CSS files.

### 7.4 Deployment Best Practices

-   **Environment Variables:** Use environment variables to configure your deployments.
-   **Deployment Strategies:** Use appropriate deployment strategies (e.g., blue/green deployment, canary deployment) to minimize downtime.
-   **Rollback Strategies:**  Implement rollback strategies to revert to a previous version if a deployment fails.

### 7.5 CI/CD Integration

-   **Continuous Integration (CI):**  Run automated tests on every commit to ensure code quality.
-   **Continuous Delivery (CD):** Automate the deployment process to deliver new features and bug fixes to users quickly.
-   **Automated Releases:**  Automate the release process to create and publish releases automatically.

## Conclusion

By following these best practices and coding standards, you can create efficient, reliable, secure, and maintainable GitHub Actions workflows. Remember to adapt these guidelines to your specific needs and context. Continuously review and improve your workflows to ensure they meet your evolving requirements.