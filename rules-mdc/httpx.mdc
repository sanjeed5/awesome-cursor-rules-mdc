---
description: This guide outlines definitive best practices for using httpx, Python's modern HTTP client, ensuring robust, performant, and maintainable network interactions in both synchronous and asynchronous applications.
globs: **/*.py
---
# httpx Best Practices

`httpx` is the de-facto modern HTTP client for Python, offering both synchronous and asynchronous APIs, native HTTP/2 support, and advanced features essential for high-throughput services. Adhere to these guidelines for optimal performance, reliability, and code clarity.

## Code Organization and Structure

### Always use a `Client` or `AsyncClient` instance

Avoid top-level helper functions (`httpx.get`, `httpx.post`) for anything beyond simple, one-off scripts. Client instances enable connection pooling, HTTP/2, and persistent configuration.

❌ BAD:
```python
import httpx

# In a loop or repeated calls, this creates new connections every time
for _ in range(5):
    response = httpx.get("https://api.example.com/data")
    # ... process response
```

✅ GOOD:
```python
import httpx

# Client is instantiated once and reused
with httpx.Client() as client:
    for _ in range(5):
        response = client.get("https://api.example.com/data")
        # ... process response
```

### Instantiate a single client per process or request scope

Reuse `Client`/`AsyncClient` instances. Creating new clients in a hot loop defeats connection pooling and can exhaust resources. Pass a scoped client or use a global instance (with caution in async apps).

❌ BAD:
```python
# In an async web handler, this creates a new client per request
async def get_user_data(user_id: str):
    async with httpx.AsyncClient() as client: # Client created per request
        response = await client.get(f"https://api.example.com/users/{user_id}")
        return response.json()
```

✅ GOOD:
```python
# Instantiate client once at application startup
# For web frameworks, use dependency injection or app-level state
import httpx

# Global client (careful with stateful clients in multi-tenant apps)
# For web frameworks like FastAPI, prefer dependency injection.
_async_client: httpx.AsyncClient | None = None

async def get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(base_url="https://api.example.com")
    return _async_client

async def get_user_data(user_id: str):
    client = await get_async_client() # Reuse the same client
    response = await client.get(f"/users/{user_id}")
    return response.json()
```

### Use context managers for client lifecycle

Always use `with httpx.Client() as client:` or `async with httpx.AsyncClient() as client:` to guarantee proper shutdown of background workers and sockets.

❌ BAD:
```python
client = httpx.Client()
response = client.get("https://api.example.com/data")
# client.close() might be forgotten, leading to resource leaks
```

✅ GOOD:
```python
with httpx.Client() as client:
    response = client.get("https://api.example.com/data")
    # Client is automatically closed on exit
```

## Common Patterns and Anti-patterns

### Prioritize `AsyncClient` in async contexts

In async web frameworks (FastAPI, Starlette, Quart) or any `asyncio`/`trio` application, use `AsyncClient` and `await client.get(...)` to keep the event loop free.

❌ BAD:
```python
# In an async function, using a synchronous client blocks the event loop
import httpx
import asyncio

async def fetch_sync_in_async():
    client = httpx.Client() # This is a synchronous client
    response = client.get("https://slow.example.com") # BLOCKS
    client.close()
    return response.text
```

✅ GOOD:
```python
import httpx
import asyncio

async def fetch_async():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://fast.example.com") # Non-blocking
        return response.text
```

### Stream large payloads to reduce memory pressure

For large files or responses, use `client.stream(...)` and iterate over `response.aiter_bytes()` (or `response.iter_bytes()` in sync mode) to process data as it arrives.

❌ BAD:
```python
# Loads entire response into memory
response = client.get("https://large-file.example.com/data.zip")
file_content = response.content
# Process file_content
```

✅ GOOD:
```python
async with client.stream("GET", "https://large-file.example.com/data.zip") as response:
    response.raise_for_status()
    async for chunk in response.aiter_bytes():
        # Process chunk as it arrives
        pass
```

### Configure client-level defaults for DRY code

Pass `auth`, `headers`, `params`, and `base_url` to the client constructor to apply them to all requests made by that client instance.

❌ BAD:
```python
# Repeating headers and auth for every request
client.get("https://api.example.com/users", headers={"Authorization": "Bearer token"})
client.post("https://api.example.com/items", headers={"Authorization": "Bearer token"}, json={...})
```

✅ GOOD:
```python
client = httpx.Client(
    base_url="https://api.example.com",
    headers={"Authorization": "Bearer token"},
    auth=httpx.BasicAuth("user", "pass") # Can also set auth here
)
client.get("/users")
client.post("/items", json={...})
```

## Performance Considerations

### Enable HTTP/2 explicitly

For modern services, enable HTTP/2 support for multiplexing and improved performance. Install `httpx[http2]`.

❌ BAD:
```python
client = httpx.Client() # Defaults to HTTP/1.1
```

✅ GOOD:
```python
client = httpx.Client(http2=True) # Enables HTTP/2
```

### Set explicit, granular timeouts

Always configure timeouts. The default 5-second timeout is a safeguard, but customize it per-endpoint or client for specific needs (e.g., longer read timeouts for large responses).

❌ BAD:
```python
response = client.get("https://api.example.com/slow-endpoint") # Uses default 5s timeout
```

✅ GOOD:
```python
# Client-level timeout
client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0, read=30.0))
response = client.get("https://api.example.com/slow-endpoint")

# Request-level override
response = client.get("https://api.example.com/another-endpoint", timeout=5.0)
```

### Control redirects explicitly

`httpx` disables redirects by default. Only enable `follow_redirects=True` when necessary to avoid unexpected behavior or performance overhead.

❌ BAD:
```python
# Assuming redirects are handled automatically (they are not by default)
response = client.get("https://example.com/old-path")
print(response.url) # Might still be the old URL if not redirected
```

✅ GOOD:
```python
response = client.get("https://example.com/old-path", follow_redirects=True)
print(response.url) # Will be the new URL after redirect
```

## Common Pitfalls and Gotchas

### Handle HTTP errors gracefully

Always check for non-2xx responses. Use `response.raise_for_status()` to automatically raise `httpx.HTTPStatusError` for bad responses, or catch specific `httpx.HTTPError` subclasses.

❌ BAD:
```python
response = client.get("https://api.example.com/non-existent")
# Code proceeds even if status_code is 404 or 500
if response.status_code == 200:
    data = response.json()
else:
    print("Request failed") # Generic error handling
```

✅ GOOD:
```python
import httpx

try:
    response = client.get("https://api.example.com/non-existent")
    response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
    data = response.json()
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
except httpx.RequestError as e:
    print(f"An error occurred while requesting {e.request.url!r}: {e}")
```

### Understand `response.url` is a URL object

`response.url` is an `httpx.URL` object, not a string. Access its components or convert to string explicitly.

❌ BAD:
```python
response = client.get("https://example.com/path?query=1")
if response.url == "https://example.com/path?query=1": # String comparison fails
    pass
```

✅ GOOD:
```python
from httpx import URL

response = client.get("https://example.com/path?query=1")
if response.url == URL("https://example.com/path?query=1"): # Compare with URL object
    pass
if str(response.url) == "https://example.com/path?query=1": # Explicit string conversion
    pass
print(response.url.host) # Access components
```

## Type Hints

### Annotate `httpx` types for clarity and static analysis

Use `httpx.Client`, `httpx.AsyncClient`, `httpx.Response`, and `httpx.Request` for type hinting. This improves IDE support and aligns with modern Python practices.

❌ BAD:
```python
def fetch_data(client, url):
    response = client.get(url)
    return response.json()
```

✅ GOOD:
```python
import httpx
from typing import Dict, Any

def fetch_data(client: httpx.Client, url: str) -> Dict[str, Any]:
    response: httpx.Response = client.get(url)
    response.raise_for_status()
    return response.json()
```

## Virtual Environments, Packaging, and Testing

### Always use virtual environments

This is standard Python practice. Ensure `httpx` and its optional dependencies (e.g., `httpx[http2,brotli]`) are installed in a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "httpx[http2,brotli]"
```

### Declare `httpx` and extras in `pyproject.toml` or `requirements.txt`

For reproducible builds, explicitly list `httpx` and any required extras in your project's dependency management file.

`pyproject.toml`:
```toml
[project]
dependencies = [
    "httpx[http2,brotli]",
]
```

`requirements.txt`:
```
httpx[http2,brotli]
```

### Use `pytest-httpx` for robust testing

For unit and integration tests, `pytest-httpx` provides excellent fixtures for mocking `httpx` requests, allowing you to simulate API responses without making actual network calls. For testing internal ASGI applications, `httpx.ASGITransport` is the correct tool.

✅ GOOD (using `pytest-httpx`):
```python
import httpx
import pytest

# Install: pip install pytest-httpx

def test_fetch_user_data(httpx_mock):
    httpx_mock.add_response(url="https://api.example.com/users/1", json={"id": 1, "name": "Test User"})

    with httpx.Client() as client:
        response = client.get("https://api.example.com/users/1")
        assert response.status_code == 200
        assert response.json() == {"id": 1, "name": "Test User"}
```

✅ GOOD (using `ASGITransport` for internal ASGI app):
```python
import httpx
from httpx import ASGITransport
from my_app import app # Assuming 'app' is your ASGI application

def test_my_asgi_app():
    transport = ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://test") as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
```