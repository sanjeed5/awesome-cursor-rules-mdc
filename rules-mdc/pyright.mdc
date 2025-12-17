---
description: This guide provides definitive, actionable best practices for configuring and using pyright (and basedpyright) to ensure robust, type-safe Python codebases in 2025.
globs: **/*.py
---
# pyright Best Practices

`pyright` is the definitive static type-checker for modern Python, essential for "type-first" development. We standardize on `basedpyright` for its stricter defaults and enhanced diagnostics, ensuring our codebase remains consistently type-clean.

## 1. Project-Wide Configuration

Always define a `pyrightconfig.json` (or `[tool.pyright]` in `pyproject.toml`) at your project root. Commit this file to version control to guarantee consistent type-checking across all developer environments and CI pipelines. Start with a baseline, fix errors, then progressively tighten rules.

❌ **BAD**: No config file, relying on default `pyright` behavior.
✅ **GOOD**: Explicitly define your type-checking rules.

```json
// pyrightconfig.json
{
  "include": ["src"],
  "exclude": ["**/node_modules", "**/__pycache__"],
  "reportMissingTypeStubs": "error",
  "reportAny": "error", // basedpyright specific, essential
  "reportExplicitAny": "error", // basedpyright specific, essential
  "reportImplicitRelativeImport": "error", // basedpyright specific
  "reportInvalidCast": "error", // basedpyright specific
  "reportUnsafeMultipleInheritance": "error", // basedpyright specific
  "strict": [ // Apply strict mode to specific directories
    "src/core",
    "src/models"
  ],
  "stubPath": "stubs" // Where custom type stubs live
}
```

## 2. Strict Diagnostics & `Any` Usage

Embrace strict diagnostics to eliminate implicit `Any` types and enforce explicit type declarations. `basedpyright`'s `reportAny` and `reportExplicitAny` are non-negotiable for maintaining a truly type-safe codebase.

### Prefer `object` over `Any`

Use `object` when a function truly accepts *any* Python object, but its type isn't relevant to its operation (e.g., for `str()`, `repr()`, or passing to a callback that ignores its argument). Reserve `Any` only when the type system *cannot* express the intent.

❌ **BAD**: Overusing `Any`, disabling type checks.

```python
from typing import Any

def process_data(data: Any) -> Any: # Type checking effectively disabled
    print(str(data))
    return data
```

✅ **GOOD**: Using `object` for truly generic inputs, `Any` only when necessary.

```python
from typing import Any, Callable

def process_data(data: object) -> object: # Accepts any object, but type is known
    print(str(data))
    return data

def call_callback(cb: Callable[[int], object]) -> None: # Callback return value is ignored
    cb(42)

# Use Any only when absolutely no specific type can be expressed
def dynamic_factory(name: str) -> Any: # Return type is truly unknown at static analysis time
    if name == "int": return 1
    if name == "str": return "hello"
    return None
```

### Explicit Error Codes for Ignores

Always specify the exact `pyright` error code when suppressing diagnostics. This prevents accidentally ignoring other, unrelated errors on the same line and improves maintainability. `basedpyright`'s `reportIgnoreCommentWithoutRule` enforces this.

❌ **BAD**: Generic type ignores.

```python
import os

# type: ignore # What error is this ignoring?
path = os.getenv("MY_PATH")
```

✅ **GOOD**: Specific `pyright` ignore comments.

```python
import os
from typing import cast

# pyright: ignore[reportGeneralTypeIssues] # path might be None, but we know it's not
path = cast(str, os.getenv("MY_PATH"))
```

## 3. Type Aliases

Declare type aliases using `TypeAlias` for clarity and consistency. This makes the intent explicit and helps `pyright` understand your type definitions.

❌ **BAD**: Implicit type aliases.

```python
MyDict = dict[str, int] # Looks like an alias, but isn't explicitly declared
```

✅ **GOOD**: Explicit `TypeAlias` declaration.

```python
from typing import TypeAlias

MyDict: TypeAlias = dict[str, int]
```

## 4. Arguments and Return Types

### Function Arguments: Prefer Protocols and Abstract Types

For function arguments, use abstract base classes (ABCs) or protocols (`Iterable`, `Sequence`, `Mapping`) to maximize flexibility and reusability. This allows your functions to accept a wider range of compatible types.

❌ **BAD**: Overly specific argument types.

```python
def process_items(items: list[str]) -> None: # Only accepts lists
    for item in items:
        print(item)
```

✅ **GOOD**: Using abstract types for arguments.

```python
from collections.abc import Iterable

def process_items(items: Iterable[str]) -> None: # Accepts any iterable
    for item in items:
        print(item)
```

### Function Return Types: Prefer Concrete Types

For return values, prefer concrete types (`list`, `dict`) unless you are returning an instance of a protocol or an abstract base class where the specific implementation is not relevant to the caller. Avoid union return types that force `isinstance()` checks.

❌ **BAD**: Abstract return types for concrete implementations, or complex unions.

```python
from collections.abc import MutableMapping
from typing import Union

def create_mapping() -> MutableMapping[str, int]: # Returns a dict, but declared as MutableMapping
    return {"a": 1, "b": 2}

def get_value(key: str) -> Union[str, int, None]: # Forces caller to check types
    if key == "name": return "Alice"
    if key == "age": return 30
    return None
```

✅ **GOOD**: Concrete return types, simpler unions when necessary.

```python
def create_mapping() -> dict[str, int]: # Clearly returns a dict
    return {"a": 1, "b": 2}

def get_value(key: str) -> str | int | None: # Shorthand union, still explicit
    if key == "name": return "Alice"
    if key == "age": return 30
    return None
```

## 5. Modern Type Hint Syntax

Always use the modern shorthand syntax for unions (`X | Y`) and built-in generics (`list[int]`). Place `None` as the last element in a union.

❌ **BAD**: Legacy `typing` module syntax.

```python
from typing import Union, Optional, List, Dict, Type

def foo(x: Union[str, int]) -> None: ...
def bar(x: Optional[str]) -> Optional[int]: ...
def baz(items: List[str]) -> Dict[str, Type[object]]: ...
```

✅ **GOOD**: Modern, concise type hint syntax.

```python
def foo(x: str | int) -> None: ...
def bar(x: str | None) -> int | None: ...
def baz(items: list[str]) -> dict[str, type[object]]: ...
```

## 6. Type Stubs Management

Maintain a `stubs/` directory for custom type stubs for libraries that lack them. Configure `pyright` to use this path and enable `reportMissingTypeStubs`. Keep third-party libraries updated to leverage inline types.

```json
// pyrightconfig.json
{
  // ... other settings ...
  "stubPath": "stubs",
  "reportMissingTypeStubs": "error"
}
```

## 7. Virtual Environments

Always use virtual environments. Configure `pyright` to correctly locate your environment to ensure it uses the installed package types.

```json
// pyrightconfig.json
{
  // ... other settings ...
  "venvPath": ".venv", // Path to directory containing virtual environments
  "venv": "my_project_env" // Name of the virtual environment
}
```

## 8. Packaging for Type Safety

For libraries, include a `py.typed` marker file in your package to signal that it provides type information. This allows downstream users to benefit from your type annotations.

```
# my_library/py.typed
# This file can be empty. Its presence indicates the package is type-annotated.
```

## 9. Testing Approaches & CI Integration

Integrate `pyright` into your CI pipeline and pre-commit hooks. This ensures that type errors are caught early, preventing them from reaching the main branch.

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/DetachHead/basedpyright
  rev: v1.1.407 # Pin to a specific basedpyright version
  hooks:
    - id: basedpyright
      args: ["--ignoreexternal"] # Only check our code, not third-party libs
```

## 10. Common Pitfalls & `basedpyright` Gotchas

`basedpyright` introduces critical rules to prevent common typing mistakes:

### `reportImplicitRelativeImport`

Bans ambiguous relative imports that work as scripts but fail as modules.

❌ **BAD**: Implicit relative import.

```python
# my_package/module_a.py
def func_a(): pass

# my_package/module_b.py
import module_a # Fails when my_package is imported
```

✅ **GOOD**: Explicit relative import.

```python
# my_package/module_b.py
from . import module_a # Correct relative import
```

### `reportInvalidCast`

Prevents casting to types that have no overlap with the original, indicating a logical error.

❌ **BAD**: Invalid cast.

```python
from typing import cast

value: int = 10
str_value = cast(str, value) # int can never be str
```

✅ **GOOD**: Valid cast (e.g., narrowing a union).

```python
from typing import cast

value: int | None = None
if value is not None:
    int_value = cast(int, value) # Valid: narrowing from int | None to int
```

### `reportUnsafeMultipleInheritance`

Discourages multiple inheritance when base classes have `__init__` or `__new__` methods, as it's often unsafe and unpredictable.

❌ **BAD**: Unsafe multiple inheritance.

```python
class BaseA:
    def __init__(self): super().__init__()
class BaseB:
    def __init__(self): pass

class MyClass(BaseA, BaseB): # Unsafe: __init__ chain is ambiguous
    pass
```

✅ **GOOD**: Avoid unsafe multiple inheritance or use mixins carefully.

```python
class BaseA:
    def __init__(self): pass

class MyMixin:
    def do_something(self): pass

class MyClass(BaseA, MyMixin): # Safe if mixin doesn't have __init__
    def __init__(self):
        super().__init__()
```