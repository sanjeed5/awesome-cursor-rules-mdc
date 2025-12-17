---
description: This guide defines definitive best practices for using aiohttp, focusing on efficient, robust, and maintainable asynchronous HTTP client code.
globs: **/*.py
---
# `aiohttp` Best Practices

`aiohttp` is our go-to for high-performance asynchronous HTTP client operations. These rules ensure our `aiohttp` code is consistent, efficient, and resilient.

## 1. Manage `ClientSession` Lifecycle Properly

Always use a single, long-lived `ClientSession` per application or logical component. This leverages connection pooling and keep-alive, drastically reducing overhead. Always use it as an `async` context manager to ensure connections are closed cleanly.

❌ **BAD: Creating a new session for every request**
```python
import aiohttp
import asyncio

async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session: # Session created and closed per call
        async with session.get(url) as response:
            return await response.text()

async def main():
    await fetch_data("http://example.com/api/v1/data")
    await fetch_data("http://example.com/api/v1/status") # New session, new connection
```

✅ **GOOD: Single, long-lived `ClientSession`**
```python
import aiohttp
import asyncio

# Create the session once, typically at application startup
# or pass it down through dependencies.
_GLOBAL_HTTP_SESSION: aiohttp.ClientSession | None = None

async def get_http_session() -> aiohttp.ClientSession:
    global _GLOBAL_HTTP_SESSION
    if _GLOBAL_HTTP_SESSION is None or _GLOBAL_HTTP_SESSION.closed:
        _GLOBAL_HTTP_SESSION = aiohttp.ClientSession()
    return _GLOBAL_HTTP_SESSION

async def close_http_session():
    global _GLOBAL_HTTP_SESSION
    if _GLOBAL_HTTP_SESSION and not _GLOBAL_HTTP_SESSION.closed:
        await _GLOBAL_HTTP_SESSION.close()
        _GLOBAL_HTTP_SESSION = None

async def fetch_data(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        response.raise_for_status() # Always raise for non-2xx statuses
        return await response.text()

async def main():
    session = await get_http_session()
    try:
        data = await fetch_data(session, "http://example.com/api/v1/data")
        status = await fetch_data(session, "http://example.com/api/v1/status")
        print(f"Data: {data[:20]}..., Status: {status[:20]}...")
    finally:
        await close_http_session() # Ensure session is closed on app shutdown
```

## 2. Install All Speedups

Always install `aiohttp` with its `[speedups]` extra. This includes `aiodns` for faster DNS resolution and `backports.zstd` (for Python < 3.14) for modern compression, significantly boosting performance.

```bash
pip install aiohttp[speedups]
```

## 3. Implement Robust Timeouts and Error Handling

Explicitly define timeouts using `aiohttp.ClientTimeout` and enable `raise_for_status=True` for automatic HTTP error detection. Wrap requests in `try/except aiohttp.ClientError` to handle network and HTTP-level issues gracefully.

❌ **BAD: No timeouts, generic exception handling**
```python
async def fetch_unreliable(session: aiohttp.ClientSession, url: str) -> str:
    # This request could hang indefinitely
    async with session.get(url) as response:
        # 4xx/5xx responses won't raise an error, requiring manual check
        return await response.text()
```

✅ **GOOD: Explicit timeouts, `raise_for_status`, and specific error handling**
```python
from aiohttp import ClientSession, ClientTimeout, ClientError, ClientResponseError
import asyncio

async def fetch_reliable(session: ClientSession, url: str) -> str:
    timeout = ClientTimeout(total=5, connect=1, sock_read=3) # 5s total, 1s connect, 3s read
    try:
        async with session.get(url, timeout=timeout, raise_for_status=True) as response:
            return await response.text()
    except ClientResponseError as e:
        print(f"HTTP error for {url}: {e.status} - {e.message}")
        raise
    except ClientError as e:
        print(f"Network or client error for {url}: {e}")
        raise
    except asyncio.TimeoutError:
        print(f"Request to {url} timed out.")
        raise
```

## 4. Use Typed, Reusable Request Helpers

Encapsulate common request patterns in small, type-hinted `async` functions. This improves readability, maintainability, and enables better IDE support.

❌ **BAD: Repeated request logic and untyped functions**
```python
async def get_user_data(session, user_id):
    resp = await session.get(f"http://api.example.com/users/{user_id}")
    return await resp.json()

async def create_user(session, user_data):
    resp = await session.post("http://api.example.com/users", json=user_data)
    return await resp.json()
```

✅ **GOOD: Type-hinted, reusable functions with `json=` shortcut**
```python
from typing import Any, Dict
from aiohttp import ClientSession

async def fetch_json(session: ClientSession, url: str) -> Dict[str, Any]:
    """Fetches JSON from a URL, raises for status, and returns parsed data."""
    async with session.get(url, raise_for_status=True) as response:
        return await response.json()

async def post_json(session: ClientSession, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Posts JSON payload to a URL, raises for status, and returns parsed response."""
    async with session.post(url, json=payload, raise_for_status=True) as response:
        return await response.json()

async def get_user_data(session: ClientSession, user_id: str) -> Dict[str, Any]:
    return await fetch_json(session, f"http://api.example.com/users/{user_id}")

async def create_user(session: ClientSession, user_data: Dict[str, Any]) -> Dict[str, Any]:
    return await post_json(session, "http://api.example.com/users", user_data)
```

## 5. Prefer Built-in Authentication and Session Headers

Use the `auth` parameter for basic authentication or `DigestAuthMiddleware` for digest auth. For other authentication schemes or default headers, set them when initializing `ClientSession` or update `session.headers`. Avoid manual `Authorization` header manipulation where built-in options exist.

❌ **BAD: Manually constructing `Authorization` headers**
```python
import base64
async def fetch_with_basic_auth(session: ClientSession, url: str, user: str, password: str):
    auth_str = base64.b64encode(f"{user}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_str}"}
    async with session.get(url, headers=headers) as response:
        return await response.text()
```

✅ **GOOD: Using `auth` parameter for basic auth**
```python
from aiohttp import ClientSession, BasicAuth

async def fetch_with_basic_auth(session: ClientSession, url: str, user: str, password: str):
    auth = BasicAuth(login=user, password=password)
    async with session.get(url, auth=auth) as response:
        response.raise_for_status()
        return await response.text()

# For session-wide default headers (e.g., Bearer tokens)
async def create_auth_session(token: str) -> ClientSession:
    headers = {"Authorization": f"Bearer {token}"}
    return ClientSession(headers=headers)
```

## 6. Leverage Client Middleware for Cross-Cutting Concerns

For concerns like tracing, retries, or rate limiting that apply across many requests, implement client-side middleware. This keeps your request logic clean and separates concerns effectively.

```python
from aiohttp import ClientSession, ClientRequest, ClientResponse
from aiohttp.typedefs import ClientHandler
import logging

logging.basicConfig(level=logging.INFO)

async def logging_middleware(req: ClientRequest, handler: ClientHandler) -> ClientResponse:
    """Logs request and response details."""
    logging.info(f"Requesting: {req.method} {req.url}")
    response = await handler(req)
    logging.info(f"Received: {response.status} for {req.method} {req.url}")
    return response

async def main_with_middleware():
    async with ClientSession(middlewares=(logging_middleware,)) as session:
        await fetch_json(session, "http://httpbin.org/get")
        await post_json(session, "http://httpbin.org/post", {"key": "value"})

# Example usage with the `fetch_json` and `post_json` helpers from above.
```