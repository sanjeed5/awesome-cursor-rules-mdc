---
description: Enforce consistent Python code formatting using Black, automate style checks, and integrate seamlessly into development workflows to eliminate style debates and speed up code reviews.
globs: **/*.py
---
# black Best Practices

`black` is the uncompromising Python code formatter. Our team adopts `black` as the definitive style guide for all Python projects. By using `black`, you cede control over formatting minutiae, gaining speed, determinism, and freedom from style debates.

**🚨 IMPORTANT: For new projects, or if migrating, strongly consider [Ruff](https://pydevtools.com/handbook/reference/ruff/) instead of `black`. Ruff offers superior performance and combines formatting, linting, and import sorting into a single tool. If your project is already committed to `black`, these guidelines apply.**

## 1. Core Principle: Cede Control

`black` is opinionated by design. Do not fight its formatting. Embrace its defaults to achieve ultimate consistency across the codebase.

## 2. Installation and Configuration

Always install the latest stable version of `black` (v25.12.0 as of Dec 2025) and configure it via `pyproject.toml`.

### 2.1. Project Dependency

Declare `black` as a development dependency.

❌ **BAD: Global installation or missing dependency**
```toml
# pyproject.toml
# black not listed, relies on global install or manual management
```

✅ **GOOD: `black` as a development dependency**
```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = ">=3.10,<3.13"

[tool.poetry.group.dev.dependencies]
black = "^25.12.0" # Always pin to latest major/minor
```

### 2.2. `pyproject.toml` Configuration

Stick to `black`'s defaults. Only configure line length if absolutely necessary for legacy reasons (e.g., integrating with an older codebase that has a different standard).

❌ **BAD: Over-configuring `black`**
```toml
# pyproject.toml
[tool.black]
line-length = 100 # Deviating from 88 without strong reason
skip-string-normalization = true # Disabling core black features
```

✅ **GOOD: Minimal `black` configuration (default 88 chars)**
```toml
# pyproject.toml
[tool.black]
# No configuration needed; defaults are best.
# If you must change line-length, do so sparingly:
# line-length = 88
```

## 3. Automated Formatting Workflow

Integrate `black` into your development pipeline to ensure continuous consistency.

### 3.1. Pre-commit Hooks

Use `pre-commit` to automatically format files before every commit. This catches formatting issues early.

❌ **BAD: Manual formatting or relying on CI only**
```bash
# Developer forgets to run black before committing
git commit -m "feat: new feature" # Unformatted code committed
```

✅ **GOOD: `black` via `pre-commit`**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.12.0 # Use the latest stable version
    hooks:
      - id: black
        language_version: python3.11 # Match your project's Python version
```
**Action**: Install `pre-commit` (`pip install pre-commit`) and set up hooks (`pre-commit install`).

### 3.2. CI/CD Integration

Ensure your CI pipeline checks for `black` compliance and fails the build on any formatting mismatch.

❌ **BAD: CI passes even with unformatted code**
```bash
# In CI script
# No black check, or black --diff only
```

✅ **GOOD: CI enforces `black` formatting**
```bash
# In CI script (e.g., .github/workflows/main.yml)
- name: Check code style with Black
  run: pip install black==25.12.0 && black --check .
```

## 4. Code Organization and Structure

`black` ensures consistent formatting, which indirectly improves code organization and readability.

### 4.1. Import Sorting

`black` does *not* sort imports. Pair it with `isort` (or Ruff's built-in import sorting) for a complete solution.

❌ **BAD: Unsorted imports**
```python
import os
from my_module import ClassA
import sys
from third_party import lib
```

✅ **GOOD: Sorted imports (with `isort` or Ruff)**
```python
import os
import sys

from third_party import lib

from my_module import ClassA
```
**Action**: Add `isort` to your `pre-commit-config.yaml` or use Ruff.

### 4.2. Consistent String Quotes and Trailing Commas

`black` enforces double quotes for strings and trailing commas in multi-line constructs.

❌ **BAD: Inconsistent quotes, missing trailing commas**
```python
my_string = 'hello world'
my_list = [
    1,
    2
]
```

✅ **GOOD: `black`-formatted strings and lists**
```python
my_string = "hello world"
my_list = [
    1,
    2, # Trailing comma for multi-line
]
```

## 5. Type Hints

`black` consistently formats type hints, making them more readable. Always use explicit type hints.

❌ **BAD: Inconsistent spacing or missing type hints**
```python
def process_data(data : list):
    return len(data)

def get_name(user):
    return user.name
```

✅ **GOOD: `black`-formatted and explicit type hints**
```python
from typing import Any

def process_data(data: list[Any]) -> int:
    return len(data)

def get_name(user: Any) -> str:
    return user.name
```

## 6. Testing Approaches

`black` ensures your test files are as consistently formatted and readable as your application code. This aids in test maintenance and debugging.

❌ **BAD: Unformatted or inconsistently formatted test files**
```python
def test_addition():
    assert 1+1 == 2
```

✅ **GOOD: `black`-formatted test files**
```python
def test_addition():
    assert 1 + 1 == 2
```
**Action**: Ensure `black` runs on your `tests/` directory.

## 7. Common Pitfalls and Gotchas

### 7.1. Version Mismatches

Inconsistent `black` versions across the team or CI can lead to re-formatting churn.

❌ **BAD: Different `black` versions in use**
```bash
# Dev A uses black 24.1.0, Dev B uses black 25.12.0
# Code formatted by A gets re-formatted by B, leading to noisy diffs.
```

✅ **GOOD: Standardized `black` version**
```toml
# pyproject.toml
[tool.poetry.group.dev.dependencies]
black = "==25.12.0" # Pin to exact version for consistency
```
**Action**: Pin `black` to an exact version in `pyproject.toml` and `pre-commit-config.yaml`.

### 7.2. Ignoring `black`

Disabling `black` for specific files or sections without a strong, documented reason undermines consistency.

❌ **BAD: Disabling `black` for convenience**
```python
# my_module.py
# fmt: off
def ugly_function():
    # ... unformatted code ...
# fmt: on
```

✅ **GOOD: Trust `black`'s formatting**
```python
# my_module.py
def ugly_function():
    # black will format this consistently
    result = (
        "a_very_long_string_that_black_will_wrap_nicely"
        + "another_part_of_the_string"
    )
    return result
```
**Action**: Only use `fmt: off`/`fmt: on` directives as a last resort for genuinely un-black-able code, and document why.