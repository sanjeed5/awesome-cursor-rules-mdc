---
description: Definitive guidelines for configuring and using pylint for deep semantic analysis and code quality in Python projects.
globs: **/*.py
---
# pylint Best Practices

Pylint is our go-to for **deep semantic analysis** and identifying complex code smells in Python. While Ruff handles fast style checks and MyPy ensures type correctness, Pylint focuses on broader architectural and logical issues. This guide ensures Pylint is a powerful, not noisy, part of our workflow.

## 1. Configuration is King: `pyproject.toml`

Always use `pyproject.toml` for Pylint configuration. This ensures consistency across all environments and developers.

**Action**: Generate a baseline config and commit it.
```bash
pylint --generate-toml-config > pyproject.toml
```

### 1.1. Silence the Noise, Enable What Matters

Pylint is notoriously noisy by default. Start by disabling everything and selectively enabling relevant categories. Focus on `convention`, `refactor`, `warning`, and `error` for semantic checks.

❌ BAD: Default Pylint output (overwhelming)
```bash
pylint your_module.py # Flooded with style and minor issues
```

✅ GOOD: Targeted Pylint checks in `pyproject.toml`
```toml
# pyproject.toml
[tool.pylint.main]
disable = "all"
enable = [
    "convention",
    "refactor",
    "warning",
    "error",
    # Add specific messages if needed, e.g., "W0611", "R0913"
]
```

### 1.2. Filter by Confidence

Reduce false positives by only showing warnings with high confidence.

✅ GOOD: Filter low-confidence warnings in `pyproject.toml`
```toml
# pyproject.toml
[tool.pylint.main]
confidence = ["HIGH", "CONTROL_FLOW"] # Focus on reliable detections
```

## 2. Code Organization & Readability

Pylint helps enforce structural best practices beyond basic style.

### 2.1. Docstrings for Everything

Every module, class, and function **must** have a docstring. Pylint enforces this.

❌ BAD: Missing docstrings
```python
def calculate_sum(a, b):
    return a + b
```

✅ GOOD: Clear and concise docstrings
```python
"""This module provides basic arithmetic operations."""

def calculate_sum(a: int, b: int) -> int:
    """
    Calculates the sum of two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of a and b.
    """
    return a + b
```
*Pylint messages: `C0114` (module), `C0115` (class), `C0116` (function/method)*

### 2.2. Naming Conventions

Adhere to PEP 8 naming. Pylint helps catch deviations, especially for constants.

❌ BAD: Inconsistent naming
```python
shift = 3 # Pylint will flag as C0103: invalid-name
def my_function(): pass
```

✅ GOOD: PEP 8 compliant naming
```python
SHIFT_AMOUNT = 3 # Constants are UPPER_CASE
def my_function(): pass # Functions are snake_case
```
*Pylint message: `C0103` (invalid-name)*

### 2.3. Manage Complexity

Pylint flags overly complex functions or classes, indicating a need for refactoring.

❌ BAD: Monolithic function (Pylint `R0915` too-many-statements, `R0912` too-many-branches)
```python
def process_data_and_save(data, config):
    # ... 50 lines of data validation ...
    # ... 30 lines of data transformation ...
    # ... 20 lines of database interaction ...
    # ... 10 lines of logging ...
    pass
```

✅ GOOD: Break down into smaller, focused units
```python
def _validate_data(data):
    # ... validation logic ...
    pass

def _transform_data(data):
    # ... transformation logic ...
    pass

def _save_to_db(data, config):
    # ... database interaction ...
    pass

def process_data_and_save(data, config):
    _validate_data(data)
    transformed_data = _transform_data(data)
    _save_to_db(transformed_data, config)
    # ... logging ...
```
*Pylint messages: `R0913` (too-many-arguments), `R0914` (too-many-locals), `R0915` (too-many-statements), `R0902` (too-many-instance-attributes), `R0904` (too-many-public-methods)*

## 3. Robustness & Error Prevention

Pylint identifies patterns that lead to bugs or make debugging difficult.

### 3.1. Specific Exception Handling

Always catch specific exceptions. Broad `except Exception:` blocks hide issues.

❌ BAD: Catching all exceptions
```python
try:
    # risky operation
    pass
except Exception: # W0703: broad-exception-caught
    print("An error occurred")
```

✅ GOOD: Catching specific exceptions
```python
try:
    # risky operation
    pass
except (ValueError, TypeError) as e:
    print(f"Data error: {e}")
except IOError as e:
    print(f"File error: {e}")
```
*Pylint message: `W0703` (broad-exception-caught)*

### 3.2. Avoid Unused Code

Remove unused imports and variables to keep the codebase clean. Ruff handles import sorting, but Pylint still catches unused declarations.

❌ BAD: Unused import/variable
```python
import os # W0611: unused-import
def my_func():
    x = 10 # W0612: unused-variable
    return 5
```

✅ GOOD: Clean code
```python
def my_func():
    result = 5
    return result
```
*Pylint messages: `W0611` (unused-import), `W0612` (unused-variable)*

## 4. Integration into Workflow

Pylint is most effective when integrated into development and CI.

### 4.1. Pre-commit Hooks

Run Pylint automatically before commits. Ensure it runs *after* Ruff and Black.

✅ GOOD: `.pre-commit-config.yaml` snippet
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8 # Use latest stable version
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/PyCQA/pylint
    rev: v3.0.0 # Use latest stable version
    hooks:
      - id: pylint
        args: ["--rcfile=pyproject.toml"] # Ensure it uses our config
```
*Install with `pre-commit install`*

### 4.2. CI/CD Integration

Integrate Pylint into your CI pipeline to enforce quality on every push. Use JSON output for machine readability and `--fail-under` to set a quality gate.

✅ GOOD: GitHub Actions workflow snippet
```yaml
# .github/workflows/lint.yml
name: Lint
on: [push, pull_request]
jobs:
  pylint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11' # Or your project's version
      - name: Install dependencies
        run: pip install pylint
      - name: Run Pylint
        run: pylint --output-format=json --fail-under=8.0 --clear-cache-post-run . > pylint_report.json
      - name: Upload Pylint report (optional)
        uses: actions/upload-artifact@v4
        with:
          name: pylint-report
          path: pylint_report.json
```
*Set `--fail-under` to a score that makes sense for your project (e.g., 8.0 or 9.0). Use `--clear-cache-post-run` to prevent stale AST caches in long-running CI agents.*

## 5. Pylint in the Modern Python Stack

Pylint is one piece of a comprehensive linting strategy.

*   **Ruff**: Handles fast style checks, basic errors, and import sorting (replaces Flake8, isort, pyupgrade).
*   **Black**: Uncompromising code formatter.
*   **MyPy**: Static type checking.
*   **Pylint**: Deeper semantic analysis, code smells, and architectural issues.

This layered approach ensures maximum code quality with minimal developer friction.