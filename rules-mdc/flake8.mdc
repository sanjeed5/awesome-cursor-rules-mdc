---
description: This guide provides definitive, actionable best practices for configuring and using flake8 to enforce Python code quality, style, and catch common errors.
globs: **/*.py
---
# flake8 Best Practices

`flake8` is your essential Python linter, combining `pycodestyle`, `pyflakes`, and `mccabe` complexity checks with a powerful plugin ecosystem. It ensures your codebase adheres to PEP 8, catches common bugs, and maintains high quality. This guide outlines the definitive approach to using `flake8` effectively.

## 1. Configuration: The Single Source of Truth

Always configure `flake8` at the repository root using `pyproject.toml`. This centralizes tooling configuration and ensures consistent behavior across all environments. Avoid `.flake8` or `setup.cfg` for new projects.

**✅ GOOD: `pyproject.toml`**

```toml
# pyproject.toml
[tool.flake8]
max-line-length = 88 # Black's default, ensures compatibility
extend-ignore = [
    "E203", # Black compatibility: whitespace before ':'
    "W503", # Black compatibility: line break before binary operator
    "E501", # Ignore line length for specific files/legacy (use sparingly)
]
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    "venv",
    ".venv",
    "*.egg-info",
    "docs", # Exclude documentation build artifacts
    "migrations", # Django migrations often have long lines
]
max-complexity = 10 # Enforce reasonable function complexity
per-file-ignores = [
    "src/legacy_module.py:E501,F401", # Ignore specific errors in legacy files
    "tests/*:S101", # Allow assert for tests (flake8-bandit)
]
# Enable plugins (install them first: pip install flake8-bandit flake8-bugbear flake8-mypy)
enable-extensions = ["B", "B9", "S", "M"] # Bugbear, Bandit, Mypy
```

**❌ BAD: `.flake8` (Outdated)**

```ini
# .flake8 (Avoid this for new projects)
[flake8]
max-line-length = 88
ignore = E203,W503
exclude = .git,__pycache__,build,dist,venv,.venv,*.egg-info,docs,migrations
max-complexity = 10
```

## 2. Core Checks: What `flake8` Enforces

`flake8` combines `pycodestyle` (PEP 8 style), `pyflakes` (common bugs), and `mccabe` (cyclomatic complexity).

### a. `pycodestyle` (E/W codes) - Style Enforcement

**E501: Line Too Long**
Keep lines concise for readability. Pair `flake8` with `black` or `ruff format` for automatic formatting.

**❌ BAD**

```python
def calculate_complex_value(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10):
    return (param1 + param2) * (param3 - param4) / (param5 + param6) - (param7 * param8) + param9 - param10 # E501
```

**✅ GOOD**

```python
def calculate_complex_value(
    param1, param2, param3, param4, param5,
    param6, param7, param8, param9, param10
):
    intermediate_sum = (param1 + param2) * (param3 - param4)
    intermediate_diff = (param5 + param6) - (param7 * param8)
    return intermediate_sum / intermediate_diff + param9 - param10
```

### b. `pyflakes` (F codes) - Bug Detection

**F401: Unused Import**
Remove unused imports to keep your code clean and prevent circular dependencies.

**❌ BAD**

```python
import os # F401
import sys

def greet(name):
    print(f"Hello, {name}!")
```

**✅ GOOD**

```python
import sys

def greet(name):
    print(f"Hello, {name}!")
```

### c. `mccabe` (C901) - Cyclomatic Complexity

High cyclomatic complexity indicates a function is doing too much. Refactor complex functions into smaller, focused units. Aim for `max-complexity = 10`.

**❌ BAD**

```python
def process_data(data): # C901
    if data is None:
        return None
    if not isinstance(data, list):
        return []
    processed = []
    for item in data:
        if isinstance(item, dict):
            if "value" in item and item["value"] > 0:
                processed.append(item["value"] * 2)
            elif "default" in item:
                processed.append(item["default"])
        elif isinstance(item, int) and item % 2 == 0:
            processed.append(item / 2)
        else:
            processed.append(item)
    return processed
```

**✅ GOOD**

```python
def _process_dict_item(item):
    if "value" in item and item["value"] > 0:
        return item["value"] * 2
    if "default" in item:
        return item["default"]
    return None # Or raise an error, or return original item

def _process_int_item(item):
    if item % 2 == 0:
        return item / 2
    return item

def process_data(data):
    if data is None:
        return None
    if not isinstance(data, list):
        return []

    processed = []
    for item in data:
        if isinstance(item, dict):
            result = _process_dict_item(item)
            if result is not None:
                processed.append(result)
        elif isinstance(item, int):
            processed.append(_process_int_item(item))
        else:
            processed.append(item)
    return processed
```

## 3. Essential Plugins: Extend `flake8`'s Power

Always install and enable these plugins for a robust linting setup.

*   **`flake8-bandit` (S codes)**: Security linter. Catches common security issues.
*   **`flake8-bugbear` (B/B9 codes)**: Catches common "bear" bugs and design problems.
*   **`flake8-mypy` (M codes)**: Integrates `mypy` type checking into `flake8` output.

**Installation:**

```bash
pip install flake8 flake8-bandit flake8-bugbear flake8-mypy
```

**Configuration (already in `pyproject.toml` example):**

```toml
[tool.flake8]
enable-extensions = ["B", "B9", "S", "M"]
```

**Example: `flake8-bandit` (S101 - `assert` statement)**

**❌ BAD**

```python
def check_password(password):
    assert password is not None, "Password cannot be None" # S101
    # ... security checks ...
```

**✅ GOOD**

```python
def check_password(password):
    if password is None:
        raise ValueError("Password cannot be None")
    # ... security checks ...
```

## 4. Workflow Integration: Lint Early, Lint Often

Integrate `flake8` into your development workflow to catch issues as early as possible.

### a. Local Development (IDE Integration)

Configure your IDE (e.g., Cursor, VS Code) to run `flake8` on file save. This provides immediate feedback.

### b. Pre-commit Hooks

Use `pre-commit` to automatically run `flake8` (and formatters like `black` or `ruff format`) before every commit. This ensures no unlinted code ever reaches your repository.

**`pyproject.toml` (or `.pre-commit-config.yaml`)**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0 # Use the latest stable version
    hooks:
      - id: flake8
        args: ["--config=pyproject.toml"] # Ensure flake8 uses your pyproject.toml
```

### c. CI/CD Pipelines

Include `flake8` as a mandatory step in your CI pipeline. If `flake8` reports any errors, the build *must* fail. This acts as a quality gate.

## 5. Ignoring Errors: Be Specific, Not Lazy

Only ignore errors when absolutely necessary, and always be specific.

**❌ BAD: Blanket Ignore**

```python
import os, sys # noqa
```

**✅ GOOD: Specific Ignore**

```python
import os # noqa: F401,E402
import sys
```

**✅ GOOD: File-level Ignore (in `pyproject.toml`)**

```toml
[tool.flake8]
per-file-ignores = [
    "src/legacy_module.py:E501,F401",
]
```

## 6. Type Hints: Leverage `flake8-mypy`

While `mypy` is the primary type checker, `flake8-mypy` integrates type checking results into your `flake8` output, providing a unified view of code quality.

**❌ BAD**

```python
def add(a, b): # No type hints
    return a + b
```

**✅ GOOD**

```python
def add(a: int, b: int) -> int: # M501 (flake8-mypy will check these)
    return a + b
```

## 7. Virtual Environments: Isolate Dependencies

Always install `flake8` and its plugins within a project-specific virtual environment. This prevents dependency conflicts and ensures consistent behavior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install flake8 flake8-bandit flake8-bugbear flake8-mypy
```

## 8. Code Organization & Structure

`flake8` indirectly encourages good structure by enforcing PEP 8 and catching issues.

*   **Separate Concerns**: Run `flake8` on `src/` and `tests/` directories. Exclude `docs/` or `migrations/` as needed.
*   **Import Order**: While `flake8` itself doesn't enforce strict import order, `isort` (or `ruff format`) does. Pair `flake8` with `isort` for this.
    *   Standard library
    *   Third-party
    *   Local application
*   **Avoid Circular Dependencies**: `pyflakes` (part of `flake8`) will warn about these.

## 9. Common Pitfalls & Gotchas

*   **Outdated Configuration**: Always use `pyproject.toml`.
*   **Ignoring too much**: A long `ignore` list hides real issues. Fix the code, don't ignore the linter.
*   **Not running `flake8` everywhere**: Ensure `flake8` runs on all relevant Python files (`src`, `tests`).
*   **Mixing `flake8` with `black` without `E203, W503` ignores**: This will lead to conflicts. Always configure `flake8` for `black` compatibility.

By following these guidelines, your team will maintain a consistent, high-quality Python codebase, reducing technical debt and improving collaboration.