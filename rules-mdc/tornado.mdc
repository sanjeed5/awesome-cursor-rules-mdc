---
description: Definitive guidelines for writing high-performance, secure, and maintainable Tornado applications using modern Python 3.9+ async/await patterns and best practices.
globs: **/*.py
---
# tornado Best Practices

Tornado is a powerful, non-blocking web framework ideal for high-concurrency applications, WebSockets, and long-polling. These rules enforce modern (Python 3.9+, 2025 standards) best practices, ensuring your code is performant, secure, and maintainable.

## 1. Embrace `async/await` for All I/O

Always use `async def` and `await` for any I/O-bound operations. Avoid older callback-based APIs. Tornado integrates seamlessly with `asyncio`.

❌ BAD: Blocking I/O in a handler
```python
import time
import tornado.web

class BadHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        time.sleep(1) # Blocks the IOLoop
        self.write("Hello, blocked world!")
```

✅ GOOD: Asynchronous I/O
```python
import asyncio
import tornado.web

class GoodHandler(tornado.web.RequestHandler):
    async def get(self) -> None:
        await asyncio.sleep(1) # Non-blocking
        self.write("Hello, async world!")
```

## 2. Leverage Type Hints

Utilize full type annotations (`from __future__ import annotations`) for clarity, static analysis, and improved IDE support. Tornado's stubs are well-maintained.

❌ BAD: Untyped handler
```python
class UntypedHandler(tornado.web.RequestHandler):
    def post(self):
        name = self.get_argument("name")
        self.write(f"Hello, {name}")
```

✅ GOOD: Fully typed handler
```python
from __future__ import annotations
import tornado.web

class TypedHandler(tornado.web.RequestHandler):
    async def post(self) -> None:
        name: str = self.get_argument("name")
        self.write(f"Hello, {name}")
```

## 3. Centralized Error Handling

Let Tornado's built-in error handling for malformed requests (e.g., `multipart-form-data`, invalid headers) propagate. Implement a custom `ErrorHandler` for consistent responses.

❌ BAD: Ignoring or manually handling low-level header errors
```python
# This is an anti-pattern; Tornado handles these automatically.
# Do not try to catch exceptions for malformed headers or multipart data
# directly in every handler. Let it propagate.
```

✅ GOOD: Custom `HTTPError` handler
```python
from __future__ import annotations
import tornado.web
import json

class BaseHandler(tornado.web.RequestHandler):