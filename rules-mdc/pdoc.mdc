---
description: This guide provides definitive rules for writing Python docstrings and type hints to maximize clarity and utility with pdoc, ensuring accurate and maintainable API documentation.
globs: **/*.py
---
# pdoc Best Practices

`pdoc` is our definitive tool for generating API documentation from Python docstrings and type annotations. This guide outlines the mandatory practices to ensure our generated documentation is consistent, accurate, and immediately useful.

## 1. Docstrings: The Foundation (Google Style)

Always use **Google Style docstrings** for all public modules, classes, functions, and methods. This provides a clear, structured format that `pdoc` parses effectively.

### Mandatory Docstring Sections:
*   **Summary Line**: A concise, one-line summary of the object's purpose.
*   **Detailed Description**: Elaborate on behavior, context, and important considerations.
*   **`Args`**: List each parameter with its type and description.
*   **`Returns`**: Describe the return value and its type.
*   **`Raises`**: Document any exceptions that can be raised.
*   **`Example`**: Provide a short, copy-pasteable usage example.

❌ **BAD: Incomplete and inconsistent docstring**
```python
def calculate_discount(price, percentage):
    """Applies a discount to a price.
    Returns the discounted price.
    """
    return price * (1 - percentage / 100)
```

✅ **GOOD: Comprehensive Google Style docstring**
```python
def calculate_discount(price: float, percentage: float) -> float:
    """Applies a percentage-based discount to a given price.

    This function calculates the final price after applying a specified
    discount percentage. The percentage should be a value between 0 and 100.

    Args:
        price (float): The original price of the item.
        percentage (float): The discount percentage to apply (0-100).

    Returns:
        float: The discounted price.

    Raises:
        ValueError: If the percentage is not within the 0-100 range.

    Example:
        >>> calculate_discount(100.0, 10.0)
        90.0
        >>> calculate_discount(50.0, 0.0)
        50.0
    """
    if not (0 <= percentage <= 100):
        raise ValueError("Discount percentage must be between 0 and 100.")
    return price * (1 - percentage / 100)
```

## 2. Type Hints: pdoc's Superpower

**Always use explicit type hints** for all function arguments, return values, and class attributes in public APIs. `pdoc` leverages type hints for accurate cross-referencing and improved readability.

❌ **BAD: Missing type hints**
```python
def process_data(data, config):
    """Processes raw data using provided configuration."""
    # ... logic ...
    return processed_data
```

✅ **GOOD: Complete type hints**
```python
from typing import Any, Dict

def process_data(data: list[Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Processes raw data using provided configuration.

    Args:
        data (list[Any]): A list of raw data items to process.
        config (Dict[str, Any]): Configuration parameters for processing.

    Returns:
        Dict[str, Any]: The processed data as a dictionary.
    """
    # ... logic ...
    return {"result": "processed"}
```

## 3. Module and Package Docstrings

Every Python module (`.py` file) and package (`__init__.py`) must start with a docstring providing a high-level overview.

*   **Module Docstring**: Explain the module's primary purpose and key contents.
*   **Package Docstring (`__init__.py`)**: Describe the package's overall functionality and main sub-modules.

```python
# my_module.py
"""This module provides utility functions for data manipulation.

It includes functions for filtering, sorting, and transforming lists of data.
"""

def filter_list(data: list[Any], condition: Callable[[Any], bool]) -> list[Any]:
    # ...
    pass
```

## 4. `__all__` for Explicit API Exposure

Define `__all__` in `__init__.py` files and modules to explicitly declare the public API. This guides `pdoc` on what to include in the generated documentation and prevents accidental exposure of internal components.

```python
# my_package/__init__.py
"""A package for advanced data processing utilities."""

__all__ = ["data_processor", "config_manager"] # Only these modules are public
```

```python
# my_package/data_processor.py
"""Core data processing functions."""

__all__ = ["process_records", "clean_entries"] # Only these functions are public

def process_records(...): ...
def clean_entries(...): ...
def _internal_helper(...): ... # Not in __all__, not part of public API
```

## 5. Inline Comments vs. Docstrings

**Use inline comments for *why*, docstrings for *what***.
*   **Docstrings**: Describe the *public interface*—what a function/class does, its inputs, outputs, and how to use it.
*   **Inline Comments**: Explain *complex implementation details* or *non-obvious logic* within a function body.

❌ **BAD: Explaining obvious code with comments, or using comments for API details**
```python
def add(a: int, b: int) -> int:
    # This function adds two numbers
    # a: first number
    # b: second number
    # returns: sum of a and b
    return a + b # Return the sum
```

✅ **GOOD: Docstring for API, comments for internal logic**
```python
def process_financial_transaction(amount: float, user_id: str) -> bool:
    """Processes a financial transaction for a user.

    Applies specific business rules for fraud detection before committing.

    Args:
        amount (float): The transaction amount.
        user_id (str): The ID of the user initiating the transaction.

    Returns:
        bool: True if the transaction was successful, False otherwise.
    """
    # Check for known fraud patterns before proceeding.
    # This specific rule (amount > 1000 for new users) is mandated by compliance.
    if amount > 1000 and _is_new_user(user_id):
        _log_fraud_attempt(user_id, amount)
        return False
    
    # Proceed with transaction
    return _commit_transaction(amount, user_id)
```

## 6. Automate Documentation Generation (CI/CD)

Integrate `pdoc` into your CI/CD pipeline. Documentation must be generated and deployed automatically on every successful merge to `main`. This ensures docs are always up-to-date.

```yaml
# .github/workflows/docs.yml (GitHub Actions example)
name: Generate and Deploy Docs

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pdoc
      - name: Generate pdoc documentation
        run: |
          pdoc --html --output-dir docs my_package
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## 7. Test Docstring Examples with `doctest`

Ensure that `Example` sections in your docstrings remain accurate and executable by running `doctest` as part of your test suite. This prevents documentation drift.

```python
# tests/test_my_module.py
import doctest
import unittest
from my_package import my_module # Assuming my_module has doctests

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(my_module))
    return tests

if __name__ == '__main__':
    unittest.main()
```