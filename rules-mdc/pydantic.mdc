---
description: Definitive guidelines for writing robust, maintainable, and performant pydantic models in Python, focusing on v2.12.5 best practices.
globs: **/*.py
---
# pydantic Best Practices

Pydantic v2 is the standard for data validation in modern Python. Follow these rules to build type-safe, efficient, and maintainable data models.

## 1. Model Naming and Organization

**Always use clear, singular nouns for models and group them in a dedicated `models/` package.** This improves discoverability and maintains a consistent project structure.

❌ **BAD:**
```python
# In user_operations.py
class UserData(BaseModel):
    name: str
    email: EmailStr

# In users_api.py
class UserRequest(BaseModel):
    name: str
    email: EmailStr
```

✅ **GOOD:**
```python
# In models/user.py
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
```

## 2. Strict Typing and Immutability

**Prefer concrete types over `Any`. Use `Strict*` types when no coercion is acceptable. Enable `validate_assignment=True` for mutable models, or `frozen=True` for immutable models.** Immutability is generally preferred for data models to prevent unexpected state changes.

❌ **BAD:**
```python
from pydantic import BaseModel
from typing import Any

class Item(BaseModel):
    quantity: Any # Allows "5" or 5
    price: float # Allows "10.5" or 10.5
```

✅ **GOOD:**
```python
from pydantic import BaseModel, StrictInt, StrictFloat, ConfigDict

class ImmutableItem(BaseModel):
    model_config = ConfigDict(frozen=True) # Makes instances immutable
    id: int
    name: str
    quantity: StrictInt # Only accepts int, "5" will fail
    price: StrictFloat # Only accepts float, "10.5" will fail

class MutableUser(BaseModel):
    model_config = ConfigDict(validate_assignment=True) # Validates on update
    name: str
    age: int

user = MutableUser(name="Alice", age=30)
user.age = "31" # This will raise a ValidationError
```

## 3. Safe Default Values

**Never use mutable objects (lists, dicts, sets) as direct default values.** This leads to shared state across instances. Always use `default_factory` or `Field(default_factory=...)`.

❌ **BAD:**
```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    tags: list = [] # Shared list across all Product instances
```

✅ **GOOD:**
```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list) # Each instance gets a new list
```

## 4. Custom Validation Logic

**Use `@field_validator` for single-field validation and `@model_validator` for cross-field validation.** Keep validation logic concise and focused within the model definition. Extract complex business logic to separate service layers.

❌ **BAD:**
```python
from pydantic import BaseModel, ValidationError

class Event(BaseModel):
    start_time: int
    end_time: int

    def __post_init__(self): # Not a Pydantic v2 pattern
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
```

✅ **GOOD:**
```python
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from typing import Self

class Event(BaseModel):
    start_time: int
    end_time: int

    @field_validator('start_time')
    @classmethod
    def check_start_time_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Start time must be positive")
        return v

    @model_validator(mode='after')
    def check_time_order(self) -> Self:
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        return self

try:
    Event(start_time=10, end_time=5)
except ValidationError as e:
    print(e) # Shows 'Start time must be before end time'
```

## 5. Settings Management

**Store application configuration in `pydantic_settings.BaseSettings` subclasses.** This automatically validates environment variables and ensures secrets are handled securely (e.g., with `SecretStr`).

❌ **BAD:**
```python
# config.py
API_KEY = os.getenv("MY_API_KEY", "default_secret")
DATABASE_URL = "sqlite:///./test.db"
```

✅ **GOOD:**
```python
# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore') # Load from .env, ignore unknown env vars

    api_key: SecretStr
    database_url: str = "sqlite:///./test.db"

# Usage:
settings = Settings()
print(settings.api_key.get_secret_value()) # Access secret value safely
```

## 6. Editor Integration (VS Code / Pylance)

**Configure VS Code with Pylance (`Type Checking Mode: strict`) to get real-time autocompletion and type error diagnostics.** For intentional Pydantic coercions that Pylance flags, use specific ignore comments or `cast`.

❌ **BAD:**
```python
# No Pylance, or Pylance in 'off' mode. Misses errors early.
# Or, using general # type: ignore for all lines with Pydantic coercion.
from pydantic import BaseModel
class User(BaseModel):
    age: int
user = User(age='23') # No error shown, but Pylance could warn
```

✅ **GOOD:**
```python
# VS Code settings: "python.analysis.typeCheckingMode": "strict"
from pydantic import BaseModel
from typing import cast

class User(BaseModel):
    age: int

user_str_age = User(age='23') # Pylance will flag this as str -> int, which Pydantic handles.
# If you want to silence Pylance for this specific, known coercion:
user_str_age_ignored = User(age='23') # pyright: ignore[reportGeneralTypeIssues]
# Or explicitly cast if you prefer:
user_str_age_casted = User(age=cast(int, '23'))
```

## 7. Common Pitfalls

**Avoid field names that collide with Python keywords or Pydantic internal attributes.** This can lead to unexpected behavior or validation errors.

❌ **BAD:**
```python
from pydantic import BaseModel
from typing import Optional

class BadModel(BaseModel):
    int: Optional[int] = None # 'int' is a built-in type, causes collision
    model_config: str = "some_config" # Collides with Pydantic's model_config
```

✅ **GOOD:**
```python
from pydantic import BaseModel
from typing import Optional

class GoodModel(BaseModel):
    integer_value: Optional[int] = None
    custom_config_name: str = "some_config"
```