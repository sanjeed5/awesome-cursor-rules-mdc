---
description: Definitive guidelines for building high-performance, maintainable, and secure Sanic applications using modern Python async best practices.
globs: **/*.py
---
# sanic Best Practices

Sanic is built for speed and scalability. To harness its full power and maintain a robust codebase, adhere strictly to these guidelines.

## 1. Code Organization and Structure

Always structure your Sanic application using the **factory pattern** and **Blueprints** for modularity. Keep configuration separate.

❌ BAD: Global app instance, monolithic file
```python
# app.py
from sanic import Sanic, response

app = Sanic("my_app") # Global instance is hard to test and configure

@app.route("/")
async def hello_world(request):
    return response.json({"message": "Hello!"})

# Configuration directly in code
app.config.DB_URL = "sqlite:///db.sqlite"
```

✅ GOOD: Factory pattern, Blueprints, external config
```python
# config.py
import os

class Config:
    DB_URL = os.getenv("DB_URL", "sqlite:///db.sqlite")
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")

# app.py
from sanic import Sanic
from .config import Config
from .blueprints.users import bp as users_bp
from .blueprints.auth import bp as auth_bp

def create_app() -> Sanic:
    app = Sanic("my_app")
    app.config.update(Config.__dict__) # Load config from a dedicated object

    app.blueprint(users_bp)
    app.blueprint(auth_bp)

    # Add listeners, middleware, etc. here
    @app.listener("before_server_start")
    async def setup_db(app, loop):
        # Example: Initialize a database connection pool
        app.ctx.db = "AsyncDatabaseConnection" # Placeholder for actual DB setup
        print(f"Connecting to DB: {app.config.DB_URL}")

    return app

# blueprints/users.py
from sanic import Blueprint, Request, response
from typing import Dict, Any

bp = Blueprint("users", url_prefix="/users")

@bp.route("/", methods=["GET"])
async def get_all_users(request: Request) -> response.HTTPResponse:
    # Simulate async DB call using app.ctx.db
    # await request.app.ctx.db.fetch_all("SELECT * FROM users")
    return response.json([{"id": 1, "name": "Alice"}])

@bp.route("/<user_id:int>", methods=["GET"])
async def get_user(request: Request, user_id: int) -> response.HTTPResponse:
    # Simulate async DB call
    # user = await request.app.ctx.db.fetch_one(f"SELECT * FROM users WHERE id={user_id}")
    user = {"id": user_id, "name": "Alice"} # Placeholder
    if user:
        return response.json(user)
    return response.json({"message": "User not found"}, status=404)
```

## 2. Common Patterns and Anti-patterns

**Never block the event loop.** Sanic's performance hinges on non-blocking I/O. Offload heavy computations or blocking I/O to background tasks or external services.

❌ BAD: Blocking I/O in a handler
```python
import time
from sanic import Request, response

@app.route("/blocking")
async def blocking_endpoint(request: Request) -> response.HTTPResponse:
    time.sleep(5) # Blocks the entire event loop!
    return response.json({"message": "Done after 5s"})
```

✅ GOOD: Use `asyncio.sleep` or offload blocking work
```python
import asyncio
from sanic import Request, response
from concurrent.futures import ThreadPoolExecutor # For I/O-bound blocking tasks
from typing import Dict, Any

# Use a global executor or pass it via app.ctx
executor = ThreadPoolExecutor()

def heavy_blocking_io_task(data: int) -> Dict[str, int]:
    # Simulate I/O-bound blocking task
    time.sleep(2)
    return {"result": data * 2}

@app.route("/non_blocking")
async def non_blocking_endpoint(request: Request) -> response.HTTPResponse:
    await asyncio.sleep(0.1) # Yields control, doesn't block
    return response.json({"message": "Done quickly"})

@app.route("/heavy_io_task")
async def heavy_io_task_endpoint(request: Request) -> response.HTTPResponse:
    data = request.json.get("value", 10) if request.json else 10
    # Run blocking I/O in a thread pool
    result = await request.app.loop.run_in_executor(executor, heavy_blocking_io_task, data)
    return response.json(result)
```

## 3. Performance Considerations

Leverage Sanic's async nature. Use `app.add_task` for fire-and-forget background operations.

✅ GOOD: Background tasks for non-critical work
```python
from sanic import Sanic, Request, response
import asyncio

app = Sanic("perf_app")

async def send_email_async(recipient: str, subject: str, body: str):
    await asyncio.sleep(2) # Simulate I/O for sending email
    print(f"Email sent to {recipient}: {subject}")

@app.route("/signup", methods=["POST"])
async def signup(request: Request) -> response.HTTPResponse:
    user_data = request.json
    # ... save user to DB ...

    # Add email sending to background task, doesn't block response
    request.app.add_task(send_email_async(user_data["email"], "Welcome!", "Thanks for signing up!"))

    return response.json({"message": "User signed up successfully"}, status=201)
```

## 4. Common Pitfalls and Gotchas

Forgetting `await` is a common mistake that leads to unhandled coroutines and unexpected behavior. Always `await` coroutine calls.