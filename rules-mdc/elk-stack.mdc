---
description: Definitive guidelines for emitting, structuring, and managing logs within the ELK stack, ensuring robust observability and efficient troubleshooting.
globs: **/*
---
# elk-stack Best Practices

The ELK (Elasticsearch-Logstash-Kibana) stack is our standard for centralized logging and observability. Adhering to these guidelines ensures our logs are consistent, actionable, and efficient, enabling rapid troubleshooting and deep insights.

## Code Organization and Structure

### 1. Standardize on Structured Logging

Always emit logs as structured JSON to `stdout`. This is the **only** acceptable method for application logging. Avoid writing to local files.

**Rationale**: `stdout` is the standard stream for containerized applications, easily captured by Elastic Agent or Filebeat. Structured JSON ensures logs are machine-readable and parsable without complex regex, making them immediately queryable in Elasticsearch.

**✅ GOOD: Python with `structlog`**

```python
# app/logging_config.py
import sys
import structlog
import os

def configure_logging():
    # Define canonical fields and processors
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.merge_extra_context,
        # Enforce canonical fields: service_name, trace_id, span_id
        lambda logger, method_name, event_dict: event_dict.update(
            service_name=os.getenv("SERVICE_NAME", "unknown-service"),
            trace_id=os.getenv("X_B3_TRACEID", "no-trace-id"), # Example for B3 propagation
            span_id=os.getenv("X_B3_SPANID", "no-span-id"),
        ),
    ]

    if os.getenv("APP_ENV", "development") == "production":
        # Production: JSON output for log aggregators
        processors = shared_processors + [
            structlog.processors.dict_tracebacks, # Structured tracebacks
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty printing for local readability
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Optionally, redirect standard Python logging to structlog
    # import logging
    # logging.basicConfig(handlers=[structlog.stdlib.ProcessorFormatter.wrap_for_formatter], level=os.getenv("LOG_LEVEL", "INFO").upper())
    # structlog.stdlib.ProcessorFormatter.remove_processors_from_logger(logging.getLogger())

# In your application entry point:
# from app.logging_config import configure_logging
# configure_logging()
# log = structlog.get_logger(__name__)
```

### 2. Enforce a Canonical Log Line Schema

Every log entry **must** conform to a shared schema. This enables consistent parsing, filtering, and dashboarding across all services.

**Required Fields**:
-   `@timestamp` (ISO 8601 format)
-   `log.level` (e.g., `info`, `warn`, `error`, `debug`)
-   `message` (human-readable string)
-   `service.name` (unique identifier for the service)
-   `trace.id` (correlation ID for distributed tracing)
-   `span.id` (specific operation within a trace)
-   Additional context-specific key-value pairs (e.g., `user_id`, `request.method`, `http.status_code`).

**Rationale**: A consistent schema is crucial for automated parsing and correlation in Kibana. It reduces noise and makes logs immediately useful.

## Common Patterns and Anti-patterns

### 1. Emit Structured JSON to `stdout`

**✅ GOOD: Logging structured data to `stdout`**

```python
# Using the configured structlog logger
from app.logging_config import log # Assuming log is configured as above

log.info("user_registered", user_id="abc-123", email="user@example.com", source="web_app")
log.error("database_connection_failed", db_host="db.prod.internal", port=5432, retry_count=3, exc_info=True)
```

**❌ BAD: Writing to local files or unstructured logs**

```python
# Avoid this at all costs
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.info(f"User abc-123 registered from web_app with email user@example.com")

# Also bad: unstructured print statements
print("ERROR: Database connection failed!")
```

### 2. Inject Correlation IDs for Distributed Tracing

Always inject `trace.id` and `span.id` into your logs, typically from incoming request headers (e.g., OpenTelemetry B3 headers). This links all log entries related to a single request across microservices.

**Rationale**: Essential for debugging distributed systems and understanding end-to-end request flows.

**✅ GOOD: Automatic context injection (e.g., using middleware)**

```python
# Example for a Python web framework (e.g., Flask/FastAPI middleware)
from contextvars import ContextVar
import structlog

request_context = ContextVar("request_context", default={})

def trace_middleware(request):
    trace_id = request.headers.get("X-B3-TraceId", "no-trace-id")
    span_id = request.headers.get("X-B3-SpanId", "no-span-id")
    request_context.set({"trace_id": trace_id, "span_id": span_id})
    # Bind context to structlog for this request
    structlog.contextvars.bind_contextvars(trace_id=trace_id, span_id=span_id)
    # ... process request ...
    structlog.contextvars.clear_contextvars() # Clean up after request

# In your application code, simply log:
log = structlog.get_logger(__name__)
log.info("request_received", path=request.path, method=request.method)
# trace_id and span_id will be automatically added by structlog's contextvars processor
```

**❌ BAD: Manually passing IDs or omitting them**

```python
# This is error-prone and leads to missing correlation
def process_order(order_id):
    # No trace_id here, difficult to correlate
    log.info("processing_order", order_id=order_id)
```

### 3. Use Consistent Log Levels

Adhere to standard log levels and their intended use.

-   **`DEBUG`**: Detailed diagnostic information, useful only during development or deep troubleshooting. **Never enable in production.**
-   **`INFO`**: General operational messages, indicating normal application flow.
-   **`WARN`**: Potentially harmful situations, unexpected but recoverable events. Requires attention.
-   **`ERROR`**: Runtime errors or unexpected conditions that prevent normal operation. Requires immediate attention.
-   **`CRITICAL`**: Severe errors leading to application shutdown or data loss.

**Rationale**: Proper log levels enable effective filtering and alerting.

**✅ GOOD: Appropriate log level usage**

```python
log.info("user_authenticated", user_id="user-456")
if cache_miss:
    log.warn("cache_miss", key="product_data", reason="expired")
try:
    # ... risky operation ...
except Exception as e:
    log.error("failed_to_process_payment", order_id="ord-789", error=str(e), exc_info=True)
```

**❌ BAD: Inconsistent or arbitrary level usage**

```python
# Using INFO for an error condition
log.info("Failed to connect to external API, retrying...") # Should be WARN or ERROR
```

## Performance Considerations

### 1. Offload Heavy Processing to Ingest Pipelines

Your application should only emit raw, structured logs. All complex parsing, enrichment (e.g., adding host/container info), and routing **must** happen in Logstash or Elasticsearch Ingest Pipelines.

**Rationale**: Keeps application lightweight, focused on business logic, and prevents logging from becoming a performance bottleneck.

### 2. Avoid Excessive Logging in Production

Set log levels appropriately for production environments (`INFO` or `WARN`). `DEBUG` logs are for development only.

**Rationale**: High volume logging increases I/O, network traffic, storage costs, and Elasticsearch indexing load.

## Common Pitfalls and Gotchas

### 1. Schema Drift

Inconsistent field names or data types across services will break Kibana dashboards and search queries.

**Solution**: Define a strict logging schema and enforce it via code reviews and CI/CD checks. Use shared logging libraries.

### 2. Logging Sensitive Data

Never log Personally Identifiable Information (PII), credentials, API keys, or other sensitive data.

**Solution**: Implement data masking or redaction at the application level **before** logs are emitted.

### 3. Neglecting Index Lifecycle Management (ILM)

Without ILM, Elasticsearch indices will grow indefinitely, leading to performance degradation and high storage costs.

**Solution**: Configure ILM policies in Elasticsearch to automatically roll over, shrink, and delete old indices based on age or size.

### 4. Relying on Ad-hoc Log Files

Writing logs to files on disk in containerized environments is a critical anti-pattern. These files are often ephemeral, hard to access, and not centrally managed.

**Solution**: Always log to `stdout` (and `stderr` for errors). Let the container runtime and Elastic Agent/Filebeat handle collection.

## Testing Approaches

### 1. Unit Test Log Output

Verify that your application emits logs with the correct structure, content, and required fields.

**✅ GOOD: Unit test example (Python `pytest`)**

```python
# test_app.py
import pytest
import json
from unittest.mock import patch
from io import StringIO
import sys

# Assume configure_logging() is called in test setup
from app.logging_config import configure_logging
configure_logging() # Ensure structlog is configured for JSON output in tests
log = structlog.get_logger("test_logger")

def test_user_registration_log():
    with patch('sys.stdout', new=StringIO()) as fake_stdout:
        log.info("user_registered", user_id="test-user", email="test@example.com")
        log_output = fake_stdout.getvalue().strip()
        
        assert log_output
        parsed_log = json.loads(log_output)
        
        assert parsed_log['message'] == "user_registered"
        assert parsed_log['log.level'] == "info"
        assert parsed_log['user_id'] == "test-user"
        assert parsed_log['email'] == "test@example.com"
        assert 'service.name' in parsed_log
        assert '@timestamp' in parsed_log
```

### 2. Integration Test Log Flow

Deploy your application and verify that logs are correctly ingested, parsed, and appear in Kibana with the expected fields.

**Solution**: Use a dedicated test environment with a minimal ELK setup. Send synthetic traffic and query Kibana or Elasticsearch directly to confirm log presence and structure.

### 3. Validate Logstash/Ingest Pipelines

Test your Logstash configurations or Elasticsearch Ingest Pipelines with synthetic log data to ensure they correctly parse, enrich, and transform logs before indexing.

**Solution**: Use tools like `logstash -f config.conf --config.test_and_exit` or Elasticsearch's `_simulate` API for ingest pipelines.

### 4. CI/CD Gates for Logging Standards

Integrate automated checks into your CI/CD pipeline to enforce logging best practices, such as:
-   Linting for common anti-patterns (e.g., `print()` statements).
-   Schema validation of generated logs.
-   Ensuring `DEBUG` level is not enabled in production builds.

**Rationale**: Catch logging regressions early, before they impact production observability.