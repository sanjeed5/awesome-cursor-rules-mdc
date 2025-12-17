---
description: Definitive guidelines for integrating and using Sentry for error tracking and performance monitoring, ensuring optimal setup, performance, and debuggability across all applications.
globs: **/*
---
# Sentry Best Practices

Sentry is our definitive platform for error tracking and performance monitoring. Proper integration ensures we debug faster, ship more reliably, and maintain high application health. This guide outlines the mandatory best practices for all projects.

## 1. Code Organization and Initialization

Always initialize the Sentry SDK early in your application's bootstrap process. This ensures maximum coverage for errors and performance tracing.

### 1.1. Early and Centralized Initialization

Initialize Sentry once, as close to your application's entry point as possible. Use environment variables for sensitive data like DSN and for configuration that varies by environment.

❌ **BAD: Late or Scattered Initialization**
```python
# app/views.py (Django) or a random module
import sentry_sdk

def my_function():
    if not sentry_sdk.is_initialized(): # Don't do this, it's too late and prone to race conditions
        sentry_sdk.init(dsn="YOUR_DSN")
    # ...
```

✅ **GOOD: Application Bootstrap (Python Example)**
```python
# app.py or wsgi.py/asgi.py for web apps
import os
import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration # Example for Lambda

SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "development")
SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "unknown")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")) # Default to 0.0 in prod

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=SENTRY_ENVIRONMENT,
        release=SENTRY_RELEASE,
        enable_tracing=True, # Always enable tracing, sampling controls overhead
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        # Add integrations as needed, e.g., for specific frameworks or platforms
        # integrations=[AwsLambdaIntegration()] if SENTRY_ENVIRONMENT == "aws-lambda" else [],
    )
```

✅ **GOOD: Application Bootstrap (JavaScript/TypeScript Example)**
```typescript
// src/index.ts or app.ts
import * as Sentry from "@sentry/node"; // or @sentry/browser, @sentry/react, etc.

const SENTRY_DSN = process.env.SENTRY_DSN;
const SENTRY_ENVIRONMENT = process.env.SENTRY_ENVIRONMENT || "development";
const SENTRY_RELEASE = process.env.SENTRY_RELEASE || "unknown";
const SENTRY_TRACES_SAMPLE_RATE = parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || "0.0");

if (SENTRY_DSN) {
    Sentry.init({
        dsn: SENTRY_DSN,
        environment: SENTRY_ENVIRONMENT,
        release: SENTRY_RELEASE,
        integrations: [
            // Add framework-specific integrations here, e.g., new Sentry.Integrations.Http({ tracing: true })
            // new Sentry.Integrations.Express(),
        ],
        tracesSampleRate: SENTRY_TRACES_SAMPLE_RATE,
        enableTracing: true, // Redundant with tracesSampleRate > 0, but good for clarity
    });
}
```

### 1.2. Environment, Release, and User Context

Always set `environment` and `release` during initialization. Enrich events with user and other relevant context for faster debugging.

✅ **GOOD: Setting Context**
```python
import sentry_sdk

# ... Sentry init as above ...

def process_order(user_id: str, order_details: dict):
    with sentry_sdk.push_scope() as scope: # Use push_scope for temporary context
        scope.set_user({"id": user_id, "email": "user@example.com"})
        scope.set_tag("order_type", "premium")
        scope.set_context("order_details", order_details)
        try:
            # ... order processing logic ...
            raise ValueError("Failed to process payment")
        except Exception as e:
            sentry_sdk.capture_exception(e)
```

## 2. Common Patterns and Anti-patterns

### 2.1. Performance Tracing with OpenTelemetry

Leverage OpenTelemetry for distributed tracing. Sentry SDKs integrate seamlessly, allowing end-to-end visibility without rewriting existing instrumentation.

✅ **GOOD: OpenTelemetry Integration (Python Example)**
```python
import sentry_sdk
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor

# Configure OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter())) # For local debugging
provider.add_span_processor(SentrySpanProcessor()) # Send spans to Sentry
trace.set_tracer_provider(provider)

# Initialize Sentry (ensure enable_tracing and traces_sample_rate are set)
sentry_sdk.init(
    dsn="YOUR_DSN",
    enable_tracing=True,
    traces_sample_rate=1.0, # Adjust for production
    # ... other config ...
)

# Use OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

def my_traced_function():
    with tracer.start_as_current_span("my-custom-operation"):
        print("Executing traced operation...")
        # ... your code ...
```

### 2.2. Isolated Clients for Shared Environments (JavaScript/TypeScript)

**NEVER** use `Sentry.init()` in shared environments (e.g., browser extensions, VS Code extensions, third-party widgets). This pollutes global state and leads to data leakage. Create a dedicated client instance manually.

❌ **BAD: Global Init in Shared Environment**
```typescript
// browser_extension/background.ts
import * as Sentry from "@sentry/browser";

Sentry.init({ // ❌ DANGER: Do NOT use Sentry.init() in shared environments!
    dsn: "YOUR_DSN",
    // ...
});
```

✅ **GOOD: Manual Client Setup for Shared Environments**
```typescript
// browser_extension/background.ts
import {
    BrowserClient,
    defaultStackParser,
    getDefaultIntegrations,
    makeFetchTransport,
    Scope,
} from "@sentry/browser";

// Filter out integrations that rely on global state
const integrations = getDefaultIntegrations({}).filter(
    (defaultIntegration) =>
        !["BrowserApiErrors", "Breadcrumbs", "GlobalHandlers"].includes(defaultIntegration.name)
);

const client = new BrowserClient({
    dsn: "YOUR_DSN",
    transport: makeFetchTransport,
    stackParser: defaultStackParser,
    integrations: integrations,
    tracesSampleRate: 1.0, // Adjust for production
    enableTracing: true,
});

const scope = new Scope();
scope.setClient(client);
client.init(); // Initialize the client after setting it on the scope

// Manually capture exceptions using the client's scope
try {
    // Your extension code here
    throw new Error("Example error in extension");
} catch (error) {
    scope.captureException(error);
}
```

### 2.3. AWS Lambda Integration

For serverless functions, use the platform-specific integration.

✅ **GOOD: AWS Lambda (Python Example)**
```python
import os
import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration

SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[AwsLambdaIntegration()],
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        release=os.getenv("SENTRY_RELEASE", "unknown"),
    )

def lambda_handler(event, context):
    # Your Lambda function logic
    print("Processing event...")
    raise ValueError("Something went wrong in Lambda!")
    return {"statusCode": 200, "body": "Success"}
```

## 3. Performance Considerations

### 3.1. Configure `traces_sample_rate` Judiciously

`traces_sample_rate` directly impacts performance overhead and Sentry quota usage.

*   **Development:** Set to `1.0` to capture all transactions for full visibility.
*   **Production:** Set to a lower value (e.g., `0.01` to `0.1`) to sample a representative subset of transactions. Adjust based on traffic volume and monitoring needs.

❌ **BAD: `traces_sample_rate=1.0` in Production**
```python
sentry_sdk.init(
    dsn="YOUR_PROD_DSN",
    traces_sample_rate=1.0, # ❌ This will send ALL traces, potentially incurring high costs and overhead.
    enable_tracing=True,
)
```

✅ **GOOD: Controlled Sampling in Production**
```python
sentry_sdk.init(
    dsn="YOUR_PROD_DSN",
    traces_sample_rate=0.05, # ✅ Sample 5% of transactions in production
    enable_tracing=True,
)
```

## 4. Common Pitfalls and Gotchas

### 4.1. Missing or Invalid DSN

Without a valid DSN, Sentry cannot send events. Always verify your DSN is correctly configured, ideally via environment variables.

❌ **BAD: Hardcoded or Missing DSN**
```python
sentry_sdk.init(
    dsn="", # ❌ No DSN means no events
    # ...
)
```

✅ **GOOD: DSN from Environment Variables**
```python
import os
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"), # ✅ Always load DSN from environment
    # ...
)
```

### 4.2. Unset `environment` and `release`

Events without `environment` or `release` are significantly harder to triage and correlate. Always set these values.

❌ **BAD: Omitting Environment/Release**
```python
sentry_sdk.init(
    dsn="YOUR_DSN",
    # No environment or release set ❌
)
```

✅ **GOOD: Explicit Environment/Release**
```python
sentry_sdk.init(
    dsn="YOUR_DSN",
    environment=os.getenv("SENTRY_ENVIRONMENT", "development"), # ✅
    release=os.getenv("SENTRY_RELEASE", "my-app@1.0.0"), # ✅
)
```

### 4.3. Ignoring Inbound Filters

If Sentry events aren't appearing, check your Sentry project's **Project Settings > Inbound Filters**. Ensure you haven't accidentally enabled filters that block events (e.g., "Filter out errors known to be caused by browser extensions" if you're in a shared environment).

## 5. Testing Approaches

### 5.1. Disabling Sentry in Test Environments

Prevent test noise and unnecessary Sentry events by disabling the SDK during automated tests.

✅ **GOOD: Conditional Sentry Initialization**
```python
import os
import sentry_sdk

# Only initialize Sentry if DSN is present AND not in a test environment
if os.getenv("SENTRY_DSN") and os.getenv("APP_ENV") != "test":
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE", "unknown"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        enable_tracing=True,
    )
```

### 5.2. Manual Error Capture

For specific, non-critical errors that you want to report without crashing the application, use `sentry_sdk.capture_message()` or `sentry_sdk.capture_exception()`.

✅ **GOOD: Manual Capture**
```python
import sentry_sdk

def process_data(data: list):
    if not data:
        sentry_sdk.capture_message("Received empty data for processing.", level="warning") # ✅ Report non-exception issues
        return False
    try:
        # ... process data ...
        raise ConnectionError("Database unreachable")
    except ConnectionError as e:
        sentry_sdk.capture_exception(e) # ✅ Explicitly capture exceptions
        return False
```