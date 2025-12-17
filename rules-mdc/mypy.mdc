---
description: This guide provides definitive, actionable best practices for using mypy to ensure robust static type checking in Python projects, focusing on modern patterns and common pitfalls.
globs: **/*.py
---
# mypy Best Practices

mypy is the definitive static type checker for Python. Adhering to these guidelines ensures your codebase is robust, maintainable, and catches type errors before runtime.

## 1. Configuration & Integration

Always use a shared `pyproject.toml` for mypy configuration. This ensures consistency across the team and CI/CD. Pin your mypy version.

### 1.1. Centralized Configuration

**Always** configure mypy via `pyproject.toml`. This is the modern standard.

❌ BAD: `mypy.ini` or command-line flags only
✅ GOOD: `pyproject.toml`
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
strict_optional = true
warn_redundant_casts = true
warn_return_any = true
warn_unreachable = true
# Enable strict mode for new code, or specific modules
# strict = true
```

### 1.2. CI/CD & Pre-commit Hooks

Integrate mypy into your CI/CD pipeline and as a pre-commit hook. Catch errors early.

```bash
# .pre-commit-config.yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.5.0 # Use the latest stable version
  hooks:
    - id: check-added-large-files
    - id: check-yaml
    - id: end-of-file-fixer
    - id: trailing-whitespace
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.8.0 # Pin to the same version as your pyproject.toml
  hooks:
    - id: mypy
      args: [--no-strict-optional, --ignore-missing-imports] # Override global strict for pre-commit if needed
```

### 1.3. Incremental Adoption

For existing codebases, start with a minimal configuration and gradually enable stricter checks. Prioritize annotating widely imported modules.

```toml
# pyproject.toml (initial, less strict)
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true # Temporarily ignore missing stubs
allow_untyped_defs = true     # Allow unannotated functions
```
Then, enable stricter checks module by module or globally.

## 2. Type Hinting Best Practices

### 2.1. Prefer `object` over `Any`

Use `object` when a function truly accepts any value, especially for arguments. `Any` disables type checking.

❌ BAD:
```python
def process_data(data: Any) -> None:
    print(str(data))
```
✅ GOOD:
```python
def process_data(data: object) -> None:
    # mypy knows 'data' is an object, but not its specific type
    print(str(data))
```

### 2.2. Use `TypeAlias` for Type Aliases

Explicitly declare type aliases with `TypeAlias` for clarity.

❌ BAD:
```python
Vector = list[float]
```
✅ GOOD:
```python
from typing import TypeAlias

Vector: TypeAlias = list[float]
```

### 2.3. Concrete vs. Abstract Types

For arguments, prefer abstract types (protocols, ABCs) for flexibility. For return values, prefer concrete types for clear implementation.

❌ BAD (arguments too specific, returns too abstract):
```python
from typing import MutableMapping

def map_data(input: list[str]) -> MutableMapping[str, int]:
    return {s: len(s) for s in input}
```
✅ GOOD:
```python
from collections.abc import Iterable

def map_data(input: Iterable[str]) -> dict[str, int]:
    """Maps strings to their lengths."""
    return {s: len(s) for s in input}
```

### 2.4. Shorthand Union Syntax

Use `|` for unions and `None` as the last element.

❌ BAD:
```python
from typing import Union, Optional

def get_item(key: Union[str, int]) -> Optional[str]:
    ...
```
✅ GOOD:
```python
def get_item(key: str | int) -> str | None:
    ...
```

### 2.5. Built-in Generics

Use built-in generic types (`list`, `dict`, `set`) instead of `typing` aliases.

❌ BAD:
```python
from typing import List, Dict

def process_items(items: List[str]) -> Dict[str, int]:
    ...
```
✅ GOOD:
```python
def process_items(items: list[str]) -> dict[str, int]:
    ...
```

### 2.6. Protocols for Structural Typing

Define interfaces with `Protocol` for robust duck typing.

```python
from typing import Protocol

class SupportsQuack(Protocol):
    def quack(self) -> str:
        ...

def make_it_quack(duck: SupportsQuack) -> None:
    print(duck.quack())

class MyDuck:
    def quack(self) -> str:
        return "Quack!"

make_it_quack(MyDuck()) # ✅ Works, MyDuck implicitly satisfies SupportsQuack
```

## 3. Managing Type Errors

### 3.1. Specific `# type: ignore`

Only use `# type: ignore` as a last resort, always with a specific error code, and review regularly.

❌ BAD:
```python
result = some_untyped_function() # type: ignore
```
✅ GOOD:
```python
result = some_untyped_function() # type: ignore[no-untyped-call]
```

### 3.2. Strict Mode for New Code

Enable `strict = true` in `pyproject.toml` for all new modules or the entire codebase once mature.

```toml
# pyproject.toml
[tool.mypy]
strict = true
```

## 4. Performance Considerations

Use the mypy daemon (`dmypy`) for faster incremental checks in large codebases.

```bash
# Start the daemon once
dmypy start

# Run checks repeatedly
dmypy run -- foo.py bar.py

# Stop the daemon when done
dmypy stop
```

## 5. Third-Party Libraries

For libraries without type stubs, install community stubs or generate your own.

```bash
# Install community stubs
mypy --install-types
```