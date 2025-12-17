---
description: Enforce Datadog best practices for structured logging, consistent tagging, metric governance, and CI/CD integration to ensure reliable and actionable observability across all services.
globs: **/*
---
# Datadog Best Practices

This guide outlines our team's definitive standards for integrating Datadog into our applications and infrastructure. Adhering to these rules ensures consistent, high-quality observability data, enabling faster debugging, better monitoring, and unified insights across our stack.

---

## 1. Unified Service Tagging (UST) is Non-Negotiable

Every single piece of telemetry (metrics, logs, traces, events) **must** include the `service`, `env`, and `version` tags. This is fundamental for correlating data across Datadog products and achieving true end-to-end observability. Without these, your data is effectively siloed and useless for holistic analysis.

### Configuration Management

**Always** set these as environment variables in your deployment pipeline. This ensures consistency and reduces boilerplate.

❌ BAD: Ad-hoc tagging or missing core tags
```python
# Inconsistent or incomplete tagging
import datadog.dogstatsd as dogstatsd
import logging

# Metric without service/env/version
dogstatsd.DogStatsd(host='localhost', port=8125).increment('my_app.requests.total')

# Log without service/env/version context
logging.info("User logged in successfully", extra={"user_id": "123"})
```

✅ GOOD: Centralized environment variables and automatic tag injection
```bash
# Set these in your deployment environment (e.g., Kubernetes, Docker, CI/CD)
export DD_SERVICE="my-api-service"
export DD_ENV="production"
export DD_VERSION="1.0.0"
export DD_AGENT_HOST="datadog-agent.monitoring.svc.cluster.local" # Or your specific agent host
```
```python
# Python example using ddtrace and standard logging
import logging
from ddtrace import tracer, config
from ddtrace.contrib.logging.logging import DatadogLogHandler
import datadog.dogstatsd as dogstatsd

# Configure ddtrace to pick up env vars automatically.
# Ensure ddtrace is initialized early in your application lifecycle.
# For auto-instrumentation, run your app with `ddtrace-run python your_app.py`.
# If not using ddtrace-run, explicitly configure:
tracer.configure(
    hostname=config.agent.hostname,
    port=config.agent.port,
    service=config.service, # Picks up DD_SERVICE env var
    env=config.env,         # Picks up DD_ENV env var
    version=config.version  # Picks up DD_VERSION env var
)

# Configure logging to send to Datadog agent and inject trace IDs
# DatadogLogHandler automatically adds service, env, version, trace_id, span_id
handler = DatadogLogHandler(host=config.agent.hostname, port=10518) # Default log intake port
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Initialize DogStatsd client once, using agent host from ddtrace config
statsd = dogstatsd.DogStatsd(host=config.agent.hostname, port=config.agent.port)

@tracer.wrap()
def process_request(request_id):
    # Metrics automatically inherit service/env/version from ddtrace config when using ddtrace-run
    # Or ensure you pass them as tags if not using auto-instrumentation for metrics.
    tracer.current_span().set_tag('request.id', request_id)
    logger.info("Processing request", extra={"request_id": request_id, "user_agent": "Mozilla/5.0"})
    statsd.increment('my_api.requests.processed', tags=[f'request_id:{request_id}'])

# Example usage
process_request("req-abc-123")
```

## 2. Structured Logging is Mandatory

**Always** emit logs in a structured (JSON) format. This makes logs parseable, searchable, and correlatable in Datadog. Unstructured logs are a last resort for debugging and should never be used for production telemetry.

❌ BAD: Unstructured log messages
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"User {user_id} failed to authenticate because of invalid credentials.")
```

✅ GOOD: Structured logging with context and automatic trace correlation
```python
import logging
from ddtrace.contrib.logging.logging import DatadogLogHandler
from ddtrace import tracer, config

# Ensure ddtrace is configured as above
handler = DatadogLogHandler(host=config.agent.hostname, port=10518)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

@tracer.wrap()
def authenticate_user(user_id, password_hash):
    if password_hash != "expected_hash":
        # Log structured data; trace_id and span_id are injected automatically by DatadogLogHandler
        logger.error(
            "Authentication failed",
            extra={
                "event": "user.authentication_failed",
                "user.id": user_id,
                "reason": "invalid_credentials",
                "http.status_code": 401
            }
        )
        return False
    logger.info(
        "Authentication successful",
        extra={
            "event": "user.authentication_successful",
            "user.id": user_id,
            "http.status_code": 200
        }
    )
    return True

authenticate_user("test_user", "wrong_password")
```

## 3. Metric Governance: Avoid Sprawl

Custom metrics are powerful but can quickly lead to "metric sprawl" and increased costs if not managed. **Always** define a clear naming convention and lifecycle for custom metrics.

### Naming Convention

Use `service_name.component.metric_name.unit` (e.g., `my_api.database.query_duration.seconds`).

❌ BAD: Inconsistent or overly granular metric names
```python
# Too generic, hard to filter
dogstatsd.DogStatsd(host='localhost', port=8125).increment('requests.total')
dogstatsd.DogStatsd(host='localhost', port=8125).gauge('db_latency', 150)

# Overly specific, creates too many unique metrics
dogstatsd.DogStatsd(host='localhost', port=8125).increment(f'user.{user_id}.login_attempts')
```

✅ GOOD: Structured, consistent, and tagged metrics
```python
import datadog.dogstatsd as dogstatsd
from ddtrace import config

# Initialize DogStatsd client once, using agent host from ddtrace config
statsd = dogstatsd.DogStatsd(host=config.agent.hostname, port=config.agent.port)

def record_api_request(endpoint, status_code):
    # Tags are crucial for slicing and dicing data without creating new metrics
    statsd.increment('my_api.requests.total', tags=[f'endpoint:{endpoint}', f'status_code:{status_code}'])
    statsd.histogram('my_api.request_duration_ms', 120, tags=[f'endpoint:{endpoint}'])

# Record a user login attempt, using tags for user_id instead of embedding in metric name
def record_login_attempt(user_id, success):
    statsd.increment('my_api.auth.login_attempts', tags=[f'user_id:{user_id}', f'success:{success}'])

record_api_request("/users", 200)
record_login_attempt("user-456", True)
```

## 4. Tracing: Context Propagation

**Always** ensure trace context is propagated across service boundaries. This is critical for end-to-end distributed tracing. Datadog's APM libraries handle this automatically for most common frameworks, but be aware of custom integrations.

❌ BAD: Breaking trace context across service calls
```python
# Service A
import requests
from ddtrace import tracer

@tracer.wrap()
def call_service_b():
    # No trace headers propagated
    response = requests.get("http://service-b/data")
    return response.json()
```

✅ GOOD: Automatic trace context propagation (requires `ddtrace-run` or explicit instrumentation)
```python
# Service A (using requests with ddtrace auto-instrumentation)
import requests
from ddtrace import tracer

# Run your application with `ddtrace-run python your_app.py`
# Or explicitly patch requests:
# from ddtrace import patch; patch(requests=True)

@tracer.wrap()
def call_service_b():
    # ddtrace automatically injects trace headers into outgoing requests
    response = requests.get("http://service-b/data")
    return response.json()

# Service B (with ddtrace auto-instrumentation)
# ddtrace automatically extracts trace headers from incoming requests
# and continues the trace.
```

## 5. CI/CD Integration for Observability

**Always** integrate Datadog CI Visibility into your build and deployment pipelines. This links code changes directly to their operational impact, closing the feedback loop from code to production.

### Key Practices:

*   **Store API Keys Securely**: Use secrets management (e.g., Azure Key Vault, AWS Secrets Manager, HashiCorp Vault) for `DD_API_KEY` and `DD_APP_KEY`.
*   **Tag Releases**: Inject commit hashes, branch names, and ticket IDs as tags on deployments and CI Visibility data.
*   **Monitor Build Metrics**: Track build duration, test failures, and deployment success rates.

❌ BAD: Disconnected CI/CD and observability
```yaml
# azure-pipelines.yml (no Datadog integration)
- script: |
    python -m pytest
  displayName: 'Run Tests'
```

✅ GOOD: Integrated CI Visibility and deployment tagging
```yaml
# azure-pipelines.yml (example for Python, adapt for your language/framework)
variables:
  DD_API_KEY: $(DD_API_KEY) # Stored securely in Key Vault
  DD_APP_KEY: $(DD_APP_KEY) # Stored securely in Key Vault
  DD_SITE: "datadoghq.com" # Or your specific Datadog site
  DD_ENV: "staging" # Or production, based on environment
  DD_SERVICE: "my-api-service"
  DD_VERSION: "$(Build.BuildId)" # Use build ID or commit hash for version

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.x'
    addToPath: true

- script: |
    pip install ddtrace pytest
  displayName: 'Install Dependencies'

- script: |
    # Run tests with Datadog CI Visibility
    # DD_GIT_COMMIT_SHA, DD_GIT_REPOSITORY_URL, DD_GIT_BRANCH are often auto-detected by ddtrace-run.
    # For manual tagging, you can set them explicitly.
    ddtrace-run pytest --ddtrace
  displayName: 'Run Tests with Datadog CI Visibility'
  env:
    DD_API_KEY: $(DD_API_KEY)
    DD_APP_KEY: $(DD_APP_KEY)
    DD_SITE: $(DD_SITE)
    DD_ENV: $(DD_ENV)
    DD_SERVICE: $(DD_SERVICE)
    DD_VERSION: $(DD_VERSION)
    DD_CIVISIBILITY_ENABLED: "true"
    DD_TAGS: "team:backend,project:my-app,build_id:$(Build.BuildId),commit:$(Build.SourceVersion)"

- script: |
    # Example: Notifying Datadog of a deployment event
    # Using curl for simplicity; in real-world use a dedicated Datadog API client or integration.
    curl -X POST -H "Content-Type: application/json" \
         -H "DD-API-KEY: $(DD_API_KEY)" \
         -H "DD-APPLICATION-KEY: $(DD_APP_KEY)" \
         "https://api.$(DD_SITE)/api/v1/events" \
         -d '{
               "title": "Deployment to $(DD_ENV) for $(DD_SERVICE)",
               "text": "Version $(DD_VERSION) deployed by $(Build.RequestedFor)",
               "tags": ["deployment", "env:$(DD_ENV)", "service:$(DD_SERVICE)", "version:$(DD_VERSION)", "commit:$(Build