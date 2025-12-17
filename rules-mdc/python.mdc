---
description: This guide defines the definitive Python best practices for our team, focusing on readability, maintainability, and modern development standards. Adhere to these rules for consistent, high-quality Python code.
globs: **/*.py
---
# python Best Practices

This document outlines the definitive Python best practices for our team. Adherence ensures consistent, readable, and maintainable code across all projects. We prioritize **PEP 8** as the foundation, augmented with modern tooling and patterns.

## 1. Code Layout & Formatting

Always adhere to PEP 8. Use an auto-formatter like `Black` or `Ruff` to enforce consistency.

*   **Indentation**: Use 4 spaces. Never tabs.
*   **Line Length**: Limit lines to 88 characters. Docstrings and comments should ideally wrap at 72 characters.
*   **Blank Lines**:
    *   Two blank lines between top-level functions and classes.
    *   One blank line between methods within a class.
    *   One blank line to separate logical sections within functions/methods.

❌ BAD:
```python
def my_func():
    x = 1
    y = 2
    return x + y
class MyClass:
    def __init__(self, name):
        self.name = name
    def greet(self):
        print(f"Hello, {self.name}!")
```

✅ GOOD:
```python
def my_function():
    x = 1
    y = 2

    # Separate logical steps
    result = x + y
    return result


class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")
```

## 2. Imports

Organize imports for clarity and to prevent circular dependencies. Use `isort` to automate this.

*   **Grouping**:
    1.  Standard library imports.
    2.  Third-party library imports.
    3.  Local application/project-specific imports.
*   **Alphabetical Order**: Sort imports alphabetically within each group.
*   **Absolute Imports**: Prefer absolute imports over relative imports.

❌ BAD:
```python
import os, sys
from my_package.sub_module import some_function
import requests
from .another_module import another_function
```

✅ GOOD:
```python
import os
import sys

import requests

from my_package.sub_module import some_function
from my_package.another_module import another_function
```

## 3. Naming Conventions

Follow PEP 8 naming conventions strictly.

*   **Modules**: `lowercase_with_underscores`
*   **Packages**: `lowercase_with_underscores`
*   **Classes**: `CamelCase`
*   **Functions/Methods**: `lowercase_with_underscores`
*   **Variables**: `lowercase_with_underscores`
*   **Constants**: `UPPERCASE_WITH_UNDERSCORES`
*   **Protected Members**: `_single_leading_underscore` (internal use)
*   **Private Members**: `__double_leading_underscore` (name mangling, avoid unless necessary for mixins)

❌ BAD:
```python
class myClass: # Class name not CamelCase
    def Get_Data(self): # Method name not lowercase_with_underscores
        MY_VAR = 10 # Variable name not lowercase_with_underscores
        return MY_VAR
```

✅ GOOD:
```python
class MyClass:
    def get_data(self):
        my_var = 10
        return my_var

GLOBAL_CONSTANT = 100
```

## 4. Docstrings & Comments

Document all public modules, classes, and functions using PEP 257 docstring conventions. Use reStructuredText format for Sphinx compatibility.

*   **Module Docstrings**: Top of the file, after `__future__` imports.
*   **Class Docstrings**: First line after the class definition.
*   **Function/Method Docstrings**: First line after the `def` statement.
*   **Comments**: Use sparingly for *why* code exists, not *what* it does.

❌ BAD:
```python
def calculate_sum(a, b): # This function adds two numbers
    return a + b
```

✅ GOOD:
```python
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of a and b.
    :raises TypeError: If a or b are not integers.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Inputs must be integers.")
    return a + b
```

## 5. Type Hints

**Always use type hints.** They improve readability, enable static analysis with `mypy`, and catch errors early.

*   **All Function Signatures**: Annotate parameters and return types.
*   **Variables**: Annotate complex or ambiguous variable types.
*   **`typing` module**: Use `List`, `Dict`, `Optional`, `Union`, `Callable`, `Any`, etc.
*   **`TypeAlias`**: For complex type signatures.

❌ BAD:
```python
def process_data(data):
    # ...
    return len(data)

def get_user(user_id):
    # ...
    return {"id": user_id, "name": "Test"}
```

✅ GOOD:
```python
from typing import Dict, Any, List, Optional, Union, TypeAlias

UserId: TypeAlias = Union[int, str]

def process_data(data: List[str]) -> int:
    """Processes a list of strings and returns its length."""
    return len(data)

def get_user(user_id: UserId) -> Optional[Dict[str, Any]]:
    """Retrieves user data by ID."""
    if user_id == 1:
        return {"id": 1, "name": "Alice"}
    return None
```

## 6. Virtual Environments

**Mandatory for all projects.** Use `Poetry` or `Pipenv` for dependency management and environment isolation.

*   **Poetry**: Recommended for new projects due to superior dependency resolution and packaging features.
*   **Pipenv**: Acceptable for existing projects already using it.
*   **Never commit `venv/` directories.**

❌ BAD:
```bash
# Installing directly into global Python environment
pip install requests black
```

✅ GOOD:
```bash
# Using Poetry
poetry new my_project
cd my_project
poetry add requests black --group dev
poetry run python my_script.py

# Using Pipenv
mkdir my_project && cd my_project
pipenv install requests
pipenv install black --dev
pipenv run python my_script.py
```

## 7. Packaging

Structure projects for easy distribution and installation.

*   **`src/` Layout**: Place all package code inside a `src/` directory.
*   **`pyproject.toml`**: Use this for project metadata and build configuration (PEP 621).
*   **`README.md`**: Comprehensive project description.
*   **`LICENSE`**: Clearly state the project's license.

❌ BAD:
```
my_project/
├── my_module.py
├── setup.py # Old style
└── requirements.txt
```

✅ GOOD:
```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── main.py
├── pyproject.toml
├── README.md
├── LICENSE
└── tests/
    └── test_main.py
```

## 8. Testing Approaches

**Automated testing is non-negotiable.** Use `pytest` for all tests.

*   **`pytest`**: The standard test runner.
*   **Coverage**: Integrate `pytest-cov` to ensure adequate test coverage. Aim for >90%.
*   **Fixtures**: Use `pytest` fixtures for setup and teardown.
*   **Parametrization**: Use `pytest.mark.parametrize` for testing multiple inputs.
*   **Mocks**: Use `unittest.mock` (or `pytest-mock`) for isolating units under test.
*   **Test-Driven Development (TDD)**: Strongly encouraged. Write tests before code.

❌ BAD:
```python
# No tests, or using unittest.TestCase directly without pytest
def add(a, b):
    return a + b

# Manual testing
print(add(1, 2))
```

✅ GOOD:
```python
# src/my_package/math.py
def add(a: int, b: int) -> int:
    return a + b

# tests/test_math.py
import pytest
from src.my_package.math import add

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(a: int, b: int, expected: int):
    assert add(a, b) == expected

def test_add_raises_type_error():
    with pytest.raises(TypeError):
        add("1", 2) # type: ignore
```

## 9. Common Patterns & Anti-patterns

*   **Context Managers**: Use `with` statements for resource management.

    ❌ BAD:
    ```python
    f = open("file.txt", "r")
    data = f.read()
    f.close() # Easy to forget or miss on error
    ```

    ✅ GOOD:
    ```python
    with open("file.txt", "r") as f:
        data = f.read()
    # File is automatically closed
    ```

*   **List Comprehensions/Generator Expressions**: For concise data transformations.

    ❌ BAD:
    ```python
    squares = []
    for i in range(10):
        squares.append(i * i)
    ```

    ✅ GOOD:
    ```python
    squares = [i * i for i in range(10)]
    ```

*   **F-strings**: Prefer `f-strings` for string formatting.

    ❌ BAD:
    ```python
    name = "Alice"
    age = 30
    print("Hello, %s. You are %d years old." % (name, age))
    print("Hello, {}. You are {} years old.".format(name, age))
    ```

    ✅ GOOD:
    ```python
    name = "Alice"
    age = 30
    print(f"Hello, {name}. You are {age} years old.")
    ```
    *   **Caveat**: Avoid complex expressions or function calls inside f-strings. Assign to a variable first.

*   **Enums**: Use `enum.Enum` for symbolic constants.

    ❌ BAD:
    ```python
    STATUS_PENDING = "pending"
    STATUS_COMPLETED = "completed"
    ```

    ✅ GOOD:
    ```python
    from enum import Enum

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"

    current_status = Status.PENDING
    ```

*   **Avoid Mutable Default Arguments**: This is a common pitfall.

    ❌ BAD:
    ```python
    def add_item(item, item_list=[]): # item_list is created once
        item_list.append(item)
        return item_list

    print(add_item(1)) # [1]
    print(add_item(2)) # [1, 2] - unexpected!
    ```

    ✅ GOOD:
    ```python
    from typing import List, Optional

    def add_item(item: Any, item_list: Optional[List[Any]] = None) -> List[Any]:
        if item_list is None:
            item_list = []
        item_list.append(item)
        return item_list

    print(add_item(1)) # [1]
    print(add_item(2)) # [2]
    ```

*   **Catch Specific Exceptions**: Never use bare `except:`.

    ❌ BAD:
    ```python
    try:
        # risky operation
    except: # Catches ALL exceptions, including SystemExit, KeyboardInterrupt
        print("An error occurred.")
    ```

    ✅ GOOD:
    ```python
    try:
        result = 1 / 0
    except ZeroDivisionError:
        print("Cannot divide by zero.")
    except Exception as e: # Catching a broader base for unexpected errors
        print(f"An unexpected error occurred: {e}")
    ```