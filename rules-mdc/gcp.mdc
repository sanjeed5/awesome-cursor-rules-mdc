---
description: This guide provides definitive best practices for developing, deploying, and operating applications on Google Cloud Platform (GCP), emphasizing security, performance, and maintainability.
globs: **/*
---
# gcp Best Practices

Adhering to these guidelines ensures your GCP applications are robust, secure, and performant. Treat these as non-negotiable standards.

## Code Organization and Structure

### 1. Adhere to Google's Language Style Guides
Consistency is paramount. Always follow the official Google Style Guides for your chosen language. This improves readability and maintainability across the team.

**Guideline:** Integrate linting and formatting tools (e.g., Black for Python, Prettier for JS) configured with Google's style.

### 2. Infrastructure as Code (IaC) is Mandatory
Provision and manage all GCP resources using IaC. This ensures declarative, version-controlled, and auditable infrastructure. Terraform is the default choice.

**Guideline:** Treat your infrastructure code with the same rigor as application code.

❌ **BAD:** Manual console configuration, `gcloud` commands for provisioning.
✅ **GOOD:** Terraform for all resource definitions.
```terraform
# main.tf
resource "google_project_service" "compute_api" {
  project = var.project_id
  service = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_compute_instance" "default" {
  project      = var.project_id
  zone         = "us-central1-a"
  name         = "my-app-instance"
  machine_type = "e2-medium"
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }
  network_interface {
    network = "default"
  }
}
```

## Common Patterns and Anti-patterns

### 1. Write Idempotent Functions and Services
Your functions and services must produce the same result regardless of how many times they are called with the same input. This is critical for retries and distributed systems.

❌ **BAD:** Non-idempotent operation.
```python
# Function that decrements a counter without checking state
def process_order(order_id):
    # This will decrement the counter every time it's called
    db.update_counter(order_id, -1)
```
✅ **GOOD:** Idempotent operation using a transaction or state check.
```python
# Function that processes an order idempotently
def process_order_idempotent(order_id):
    # Use a transaction or check for existing processed state
    if not db.order_processed(order_id):
        db.process_order(order_id)
        db.mark_order_processed(order_id)
    else:
        print(f"Order {order_id} already processed, skipping.")
```

### 2. Ensure HTTP Functions Send a Response
HTTP-triggered Cloud Functions and Cloud Run services *must* send an HTTP response. Failing to do so results in timeouts and unnecessary billing.

```python
# main.py
import functions_framework

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function: Always sends a response."""
    name = request.args.get("name", "World")
    # ✅ GOOD: Explicitly send an HTTP response
    return f"Hello {name}!"
```

### 3. Least Privilege for Service Accounts
Service accounts must have the absolute minimum permissions required. Never embed service account keys in code. Use Workload Identity Federation or attach service accounts to resources.

❌ **BAD:** Hardcoding service account keys or granting `roles/owner`.
```python
# Insecure: Embedding keys directly
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/key.json"
```
✅ **GOOD:** Attaching service account to Cloud Run/Functions/GKE or using Workload Identity.
```python
# Code running on GCP resource (e.g., Cloud Run) with attached SA
from google.cloud import storage
client = storage.Client() # Automatically uses attached SA credentials
```

### 4. Firestore Document IDs: Avoid Sequential Values
Monotonically increasing or decreasing document IDs (e.g., `item1`, `item2`) can lead to "hotspotting" and contention, severely impacting write performance.

❌ **BAD:** Sequential document IDs.
```python
db.collection('products').document('product_1').set({'name': 'Widget A'})
db.collection('products').document('product_2').set({'name': 'Widget B'})
```
✅ **GOOD:** Use automatic IDs or UUIDs.
```python
# Automatic ID generation
db.collection('products').add({'name': 'Widget A'})

# UUID generation
import uuid
product_id = str(uuid.uuid4())
db.collection('products').document(product_id).set({'name': 'Widget B'})
```

## Performance Considerations

### 1. Minimize Cold Start Latency for Serverless
For Cloud Functions and Cloud Run, optimize container startup time by minimizing dependencies and performing heavy computations in global scope.

❌ **BAD:** Heavy computation on every request.
```python
# main.py
import time

def expensive_init():
    time.sleep(5) # Simulates heavy setup
    return "Initialized"

def handler(request):
    data = expensive_init() # Runs on every cold start AND every request
    return f"Hello, {data}!"
```
✅ **GOOD:** Cache expensive operations in global scope.
```python
# main.py
import time

# Global scope: runs once per instance cold start
GLOBAL_DATA = "Initialized"
time.sleep(5) # Only runs once per cold start

def handler(request):
    # Reuses GLOBAL_DATA for subsequent requests on the same instance
    return f"Hello, {GLOBAL_DATA}!"
```

### 2. Configure Cloud Run for Background Activities
If your Cloud Run service performs background tasks *after* responding to an HTTP request, you *must* use instance-based billing. Otherwise, CPU access will be severely limited.

**Guideline:** For request-based billing, ensure all asynchronous operations complete before sending a response.

### 3. Optimize Firestore Indexing
Reduce write latency by setting collection-level index exemptions for fields not used in queries (e.g., large strings, sequential values, TTL fields, large arrays/maps).

**Guideline:** Only index what you query.

```python
# Example of a Firestore index exemption (conceptual, managed via console/gcloud/Terraform)
# For 'my_collection', exempt 'large_text_field' from indexing.
# This reduces storage costs and write latency.
```

## Common Pitfalls and Gotchas

### 1. Neglecting Temporary Files in Serverless
Files written to `/tmp` in Cloud Functions/Run consume memory and can persist between invocations. Always delete temporary files to prevent out-of-memory errors.

❌ **BAD:** Not cleaning up temporary files.
```python
import os
with open('/tmp/data.txt', 'w') as f:
    f.write('some data')
# File persists, consumes memory
```
✅ **GOOD:** Explicitly delete temporary files.
```python
import os
temp_file_path = '/tmp/data.txt'
with open(temp_file_path, 'w') as f:
    f.write('some data')
# ... use file ...
os.remove(temp_file_path) # Clean up immediately
```

### 2. Relying on Offsets for Firestore Pagination
Using `offset` in Firestore queries retrieves all skipped documents internally, billing you for reads and increasing latency.

❌ **BAD:** Using `offset`.
```python
query = db.collection('items').order_by('timestamp').offset(10).limit(10)
```
✅ **GOOD:** Use cursors for efficient pagination.
```python
# Get the last document from the previous page
last_doc_on_previous_page = ...

# Use start_after with a cursor
query = db.collection('items').order_by('timestamp').start_after(last_doc_on_previous_page).limit(10)
```

## Testing Approaches

### 1. Implement a Comprehensive CI/CD Pipeline
Adopt Google's "change safety" lifecycle: design, development, qualification, and rollout. This mandates automated testing (unit, integration, E2E), canary deployments, and post-deployment monitoring.

**Guideline:** Every code change must pass automated tests and be deployed through a controlled, staged release process.

### 2. Test Infrastructure as Code
Validate your Terraform or Config Connector configurations. Use tools like `terraform validate` and `terraform plan` in CI, and consider policy enforcement tools like OPA Gatekeeper.

```bash
# CI/CD pipeline step for Terraform validation
- name: Terraform Validate
  run: terraform validate

- name: Terraform Plan
  run: terraform plan -out=tfplan
```