---
description: Definitive guidelines for writing robust, secure, and maintainable scripts and workflows using the Google Cloud CLI (gcloud).
globs: **/*
---
# gcp-cli Best Practices

The `gcloud` CLI is your primary interface for Google Cloud. These guidelines ensure your `gcloud` scripts and commands are reliable, secure, and easy to maintain, adhering to Google's own Shell Style Guide and modern cloud security principles.

## 1. Code Organization and Structure

Always structure your shell scripts for readability and robustness. Use `bash` and adhere to a consistent style.

*   **Use Bash with Strict Mode**: Always start scripts with the correct shebang and enable strict mode for early error detection.
    *   `#!/usr/bin/env bash`
    *   `set -euo pipefail` ensures scripts exit on error, undefined variables, and pipe failures.
    *   `IFS=$'\n\t'` prevents unexpected word splitting.

    ❌ BAD:
    ```bash
    #!/bin/sh
    # No strict mode, prone to silent failures
    gcloud compute instances list --format="value(name)" | while read instance; do
      echo "Processing $instance"
      # ... potentially fails silently if gcloud command exits non-zero
    done
    ```

    ✅ GOOD:
    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    IFS=$'\n\t'

    # Function for consistent error logging
    err() {
      echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
    }

    main() {
      # Explicitly set project for clarity and safety
      local PROJECT_ID="your-gcp-project-id"

      # Example: Iterate and process instances
      gcloud compute instances list --project="${PROJECT_ID}" --format="value(name)" | while read -r instance; do
        echo "Processing instance: ${instance}"
        # Perform gcloud operation, check exit code if not using set -e for specific commands
        if ! gcloud compute instances stop "${instance}" --project="${PROJECT_ID}" --zone="us-central1-a" --quiet; then
          err "Failed to stop instance: ${instance}"
          exit 1 # Exit script on critical failure
        fi
      done
    }

    main "$@"
    ```

*   **Functions for Reusability**: Encapsulate logic in functions. Use `local` for function-scoped variables.
    *   Include clear function comments (description, globals, arguments, outputs, returns).

    ❌ BAD:
    ```bash
    # Global variable pollution, hard to debug
    PROJECT="my-project"
    gcloud config set project "${PROJECT}"
    # ... more script ...
    ```

    ✅ GOOD:
    ```bash
    # Function header comment as per Google Shell Style Guide
    ########################################
    # Sets the active gcloud project.
    # Globals:
    #   None
    # Arguments:
    #   $1: The project ID to set.
    # Returns:
    #   0 if successful, non-zero on error.
    #######################################
    set_gcp_project() {
      local project_id="$1"
      if ! gcloud config set project "${project_id}" --quiet; then
        err "Failed to set gcloud project to ${project_id}"
        return 1
      fi
      echo "Active gcloud project set to: ${project_id}"
      return 0
    }

    # Usage
    set_gcp_project "my-gcp-project-id"
    ```

## 2. Common Patterns and Anti-patterns

*   **Explicit Project and Zone/Region**: Always specify `--project`, `--zone`, and `--region` flags. Relying on `gcloud config` for critical operations can lead to errors in different environments.

    ❌ BAD:
    ```bash
    # Relies on configured project/zone, brittle in automation
    gcloud compute instances create my-vm --machine-type=e2-medium
    ```

    ✅ GOOD:
    ```bash
    gcloud compute instances create my-vm \
      --project="my-gcp-project-id" \
      --zone="us-central1-a" \
      --machine-type="e2-medium"
    ```

*   **Structured Output Parsing**: Use `--format=json` or `--format=yaml` with `jq` or `yq` for reliable output parsing. Avoid `grep`, `awk`, `sed` on human-readable output.

    ❌ BAD:
    ```bash
    # Fragile: depends on output format, breaks with gcloud updates
    gcloud compute instances list | grep "RUNNING" | awk '{print $1}'
    ```

    ✅ GOOD:
    ```bash
    # Robust: parses JSON output reliably
    gcloud compute instances list --project="my-gcp-project-id" --format="json" | \
      jq -r '.[] | select(.status == "RUNNING") | .name'
    ```

*   **Service Account Impersonation**: For automation, prefer `--impersonate-service-account` over activating service account keys directly. This uses short-lived tokens and improves security.

    ❌ BAD:
    ```bash
    # Activates a service account key, long-lived credential risk
    gcloud auth activate-service-account --key-file=/path/to/key.json
    gcloud compute instances list # ...
    ```

    ✅ GOOD:
    ```bash
    # Impersonates a service account using current user's permissions
    gcloud compute instances list \
      --project="my-gcp-project-id" \
      --impersonate-service-account="my-service-account@my-gcp-project-id.iam.gserviceaccount.com"
    ```

## 3. Performance Considerations

*   **Minimize API Calls**: Batch operations where possible. Avoid calling `gcloud` commands inside tight loops if a single call can fetch all necessary data.

    ❌ BAD:
    ```bash
    # N+1 problem: many API calls for each instance
    gcloud compute instances list --project="my-gcp-project-id" --format="value(name)" | while read -r instance; do
      gcloud compute instances describe "${instance}" --project="my-gcp-project-id" --format="value(status)"
    done
    ```

    ✅ GOOD:
    ```bash
    # Single API call, process data locally
    gcloud compute instances list --project="my-gcp-project-id" --format="json" | \
      jq -r '.[] | "\(.name) \(.status)"' | \
      while read -r name status; do
        echo "Instance ${name} has status ${status}"
      done
    ```

## 4. Common Pitfalls and Gotchas

*   **Unquoted Variables**: Always quote variable expansions to prevent word splitting and globbing.

    ❌ BAD:
    ```bash
    # If INSTANCE_NAME contains spaces or wildcards, this will fail
    INSTANCE_NAME="my instance"
    gcloud compute instances describe $INSTANCE_NAME
    ```

    ✅ GOOD:
    ```bash
    INSTANCE_NAME="my instance"
    gcloud compute instances describe "${INSTANCE_NAME}" --project="my-gcp-project-id"
    ```

*   **Interactive Prompts in Automation**: Use `--quiet` (`-q`) for non-interactive scripts.

    ❌ BAD:
    ```bash
    # Will prompt for confirmation, blocking automation
    gcloud compute instances delete my-vm --project="my-gcp-project-id"
    ```

    ✅ GOOD:
    ```bash
    gcloud compute instances delete my-vm --project="my-gcp-project-id" --quiet
    ```

*   **Keep `gcloud` Updated**: Regularly update your Cloud SDK to access the latest features and security patches.

    ```bash
    gcloud components update --quiet
    ```

## 5. Configuration Management

*   **Named Configurations**: Use `gcloud config configurations` to manage distinct environments (e.g., dev, staging, prod).

    ```bash
    # Create a new configuration
    gcloud config configurations create staging
    # Activate and initialize it
    gcloud config configurations activate staging
    gcloud init --console-only # Use --console-only for non-browser init
    # Switch back to default
    gcloud config configurations activate default
    ```

*   **Store Configs Outside Source Control**: Never commit `gcloud` configuration files or service account keys to source control.

## 6. Environment Variables

*   **Override Configuration**: Use `CLOUDSDK_ACTIVE_CONFIG_NAME` to temporarily switch configurations without modifying the default.

    ```bash
    # Run a command against the 'staging' config without activating it globally
    CLOUDSDK_ACTIVE_CONFIG_NAME=staging gcloud compute instances list --format="value(name)"
    ```

*   **Disable Prompts**: `CLOUDSDK_CORE_DISABLE_PROMPTS=1` is equivalent to `--quiet`.

    ```bash
    # Non-interactive deletion via environment variable
    CLOUDSDK_CORE_DISABLE_PROMPTS=1 gcloud compute instances delete old-vm --project="my-gcp-project-id"
    ```

## 7. Logging

*   **Separate STDOUT and STDERR**: Direct command output to `STDOUT` and informational/error messages to `STDERR`.

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    IFS=$'\n\t'

    err() {
      echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')][ERROR]: $*" >&2
    }

    log() {
      echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')][INFO]: $*"
    }

    log "Starting instance creation..."
    if ! gcloud compute instances create my-new-vm --project="my-gcp-project-id" --zone="us-central1-a" --machine-type="e2-small" --quiet; then
      err "Failed to create instance my-new-vm."
      exit 1
    fi
    log "Instance my-new-vm created successfully."
    ```

*   **Verbose Logging**: Use `--verbosity=debug` for detailed `gcloud` command output during debugging.

## 8. Testing Approaches

*   **ShellCheck**: Integrate ShellCheck into your CI/CD pipeline and local development workflow to catch common shell scripting errors.

    ```bash
    shellcheck your-script.sh
    ```

*   **Unit Testing**: For complex shell scripts, use frameworks like `shunit2` to write unit tests for individual functions.
*   **Integration Testing**: Create a dedicated, isolated GCP project for integration tests. Use `gcloud` commands to provision resources, run tests, and then tear down resources. This ensures your scripts work against real GCP services.