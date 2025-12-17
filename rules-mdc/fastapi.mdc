---
description: Definitive guidelines for building high-performance, maintainable, and secure FastAPI applications using modern Python best practices.
globs: **/*.py
---
# FastAPI Best Practices

FastAPI is the go-to for high-performance Python APIs. This guide ensures your projects are scalable, secure, and maintainable from day one.

## 1. Code Organization: Domain-Driven Modularity

For any project beyond a few endpoints, organize by domain functionality, not file type. This improves scalability and team collaboration.

❌ BAD: Single `main.py` or `routers/`, `schemas/` directories with all domains mixed.
```python
# app/routers/users.py
# app/routers/items.py
# app/schemas/user.py
# app/schemas/item.py
```

✅ GOOD: Group related components by domain.
```
src/
├── auth/
│   ├── router.py
│   ├── schemas.py
│   ├── service.py # Business logic
│   └── dependencies.py
├── users/
│   ├── router.py
│   ├── schemas.py
│   ├── service.py
│   └── dependencies.py
├── core/
│   ├── config.py # Pydantic BaseSettings
│   └── security.py
├── db/
│   ├── session.py # SQLAlchemy engine/session
│   └── base.py    # Base for models
├── main.py        # Entry point
└── __init__.py
```

## 2. Type Hints: Mandatory Everywhere

Leverage Python's type hints and Pydantic for robust data validation, auto-documentation, and IDE support.

❌ BAD: Missing or inconsistent type hints.
```python
@app.post("/items/")
def create_item(item: dict): # No Pydantic model
    return item
```

✅ GOOD: Explicit Pydantic models and type hints for all function signatures.
```python
from pydantic import BaseModel
from fastapi import FastAPI

class ItemCreate(BaseModel):
    name: str
    description: str | None = None
    price: float

app = FastAPI()

@app.post("/items/", response_model=ItemCreate)
async def create_item(item: ItemCreate) -> ItemCreate:
    # Logic to save item
    return item
```

## 3. Dependency Injection: Decouple Components

Use `fastapi.Depends` for managing database sessions, authentication, and other shared resources. This makes code testable and modular.

❌ BAD: Global database session or direct instantiation.
```python
# In router.py
from app.db.session import SessionLocal
db = SessionLocal() # Global or directly called
```

✅ GOOD: Inject dependencies using `Depends`.
```python
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
# Assume get_session and User are defined elsewhere
# from app.db.session import get_session
# from app.models.user import User

async def get_current_user(token: str) -> 'User': # 'User' for forward reference
    # ... auth logic
    return 'User'(id=1, username="test") # Placeholder

@app.get("/me/")
async def read_current_user(
    db: Annotated[AsyncSession, Depends(lambda: None)], # Placeholder for get_session
    current_user: Annotated['User', Depends(get_current_user)]
):
    return current_user
```

## 4. API Design: Versioning & Thin Endpoints

Version your API from day one. Keep router endpoints focused, delegating business logic to service layers.

❌ BAD: Unversioned API, fat endpoints with business logic.
```python
# app/main.py
@app.get("/users/{user_id}")
def get_user_details(user_id: int, db: 'Session'): # Placeholder
    user = None # db.query(User).filter(User.id == user_id).first()
    # Complex business logic here
    return user
```

✅ GOOD: Use `APIRouter` with prefixes and tags. Delegate logic to `service.py`.
```python
# src/users/router.py
from fastapi import APIRouter, Depends
# Assume get_session, User, UserOut, service are defined elsewhere
# from src.users import service, schemas
# from app.db.session import get_session

router = APIRouter(prefix="/v1/users", tags=["Users"])

@router.get("/{user_id}", response_model=None) # Placeholder for schemas.UserOut
async def read_user(user_id: int, db: 'AsyncSession' = Depends(lambda: None)): # Placeholder
    user = None # await service.get_user_by_id(db, user_id)
    return user

# src/users/service.py
async def get_user_by_id(db: 'AsyncSession', user_id: int) -> 'User': # Placeholder
    # Database query logic
    return None # await db.get(User, user_id)
```

## 5. Error Handling: Use `HTTPException`

Raise `HTTPException` for API-specific errors. Implement custom handlers for global error types.

❌ BAD: Raising generic Python exceptions.
```python
items_db = {1: {"name": "item1"}}
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items_db:
        raise ValueError("Item not found") # Returns 500
    return items_db[item_id]
```

✅ GOOD: Raise `HTTPException` with appropriate status codes.
```python
from fastapi import HTTPException, status

items_db = {1: {"name": "item1"}}
@app.get("/items/{item_id}")
async def get_item_good(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )
    return items_db[item_id]
```

## 6. Performance: Async-First & Production Deployment

Embrace `async`/`await` for I/O-bound operations. For CPU-bound tasks, use `run_in_threadpool`. Deploy with Gunicorn + Uvicorn.

❌ BAD: Blocking I/O in async endpoints.
```python
# In an async endpoint
import time
@app.get("/blocking")
async def blocking_endpoint():
    time.sleep(1) # Blocks the event loop
    return {"message": "Done blocking work"}
```

✅ GOOD: Use async libraries (e.g., `asyncpg`, `httpx[async]`) or `run_in_threadpool`.
```python
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
import time

app = FastAPI()

def cpu_bound_task():
    time.sleep(0.1) # Simulate CPU work
    return "Done CPU work"

@app.get("/cpu-work")
async def handle_cpu_work():
    result = await run_in_threadpool(cpu_bound_task)
    return {"message": result}

# Production deployment:
# gunicorn -k uvicorn.workers.UvicornWorker src.main:app --workers 4 --bind 0.0.0.0:8000
```

## 7. Security: Environment Variables & Auth

Store sensitive configuration in environment variables using `pydantic-settings`. Implement authentication via `Depends`.

❌ BAD: Hardcoded secrets or config.
```python
# app/core/config.py
DATABASE_URL = "postgresql://user:pass@host:port/db" # Bad
```

✅ GOOD: Use `pydantic-settings` for environment-based configuration.
```python
# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
```
```python
# .env (local development)
DATABASE_URL="postgresql+asyncpg://user:pass@db:5432/app"
SECRET_KEY="your-super-secret-key"
```
```python
# src/core/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_active_user(token: str = Depends(oauth2_scheme)):
    # Validate token and return user
    return {"username": "current_user"} # Placeholder
```

## 8. Logging: Structured & Centralized

Log to `stdout`/`stderr` in a structured format (e.g., JSON). Let your deployment environment handle aggregation.

❌ BAD: Writing logs to local files or unstructured print statements.
```python
print("User accessed /health endpoint")
```

✅ GOOD: Use Python's `logging` module with a structured formatter.
```python
import logging
# from pythonjsonlogger.jsonlogger import JsonFormatter # Install python-json-logger

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "jsonlogger.JsonFormatter", # Use if python-json-logger is installed
            "format": "%(levelname)s %(asctime)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "json", # Change to "standard" if jsonlogger not installed
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "app": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["default"], "level": "INFO"},
}

logger = logging.getLogger("app")

@app.get("/health")
async def health():
    logger.info("Health check called", extra={"endpoint": "/health"})
    return {"status": "ok"}
```