---
description: This rule file provides opinionated best practices for writing, organizing, and configuring tests using nose2, emphasizing modern Python development workflows and common pitfalls.
globs: **/*.py
---
# nose2 Best Practices

`nose2` is the modern successor to `nose`, built upon Python's `unittest` module. It offers powerful test discovery, a plugin architecture, and helpers for writing clean, maintainable test suites. This guide outlines essential best practices for leveraging `nose2` effectively in your projects.

## 1. Project Layout & Naming

Always organize your tests in a dedicated `tests/` directory at the root of your project. Name test files consistently using the `test_*.py` pattern. This ensures `nose2`'s default discovery works reliably.

**❌ BAD: Scattered tests or inconsistent naming**
```
# Project root
my_module/
  feature_a.py
  feature_a_tests.py  # Inconsistent naming
  test_feature_b.py   # Mixed with source
```

**✅ GOOD: Dedicated `tests/` directory and `test_*.py` naming**
```
# Project root
my_module/
  __init__.py
  feature_a.py
  feature_b.py
tests/
  __init__.py
  test_feature_a.py
  test_feature_b.py
```

## 2. Configuration (`pyproject.toml`)

Centralize all `nose2` configuration within your `pyproject.toml` file under the `[tool.nose2]` table. This keeps your project configuration tidy, version-controlled, and compatible with modern Python tooling. Avoid `unittest.cfg` or `nose2.cfg` for new projects.

**❌ BAD: Using legacy `.cfg` files**
```ini
# unittest.cfg
[unittest]
start-dir = tests
```

**✅ GOOD: `pyproject.toml` for all configuration**
```toml
# pyproject.toml
[tool.nose2]
start-dir = "tests"
code-directories = ["src"] # Add your source code directories to sys.path if needed
plugins = [
    "nose2.plugins.coverage",
    "nose2.plugins.params",
    "nose2.plugins.junitxml",
]
exclude-plugins = [
    "nose2.plugins.loader.functions", # Avoid loading plain functions as tests
]

[tool.nose2.coverage]
always-on = true
coverage-combine = true # Essential for multiprocessing
coverage-report = ["term", "html"]
coverage-config = ".coveragerc"
```

## 3. Test Structure & Assertions

All tests must inherit from `unittest.TestCase`. Inside your test methods, use `self.assert*` methods provided by `unittest.TestCase` for detailed failure reports. Avoid raw `assert` statements, as they offer less diagnostic information on failure.

**❌ BAD: Raw `assert` statements**
```python
import unittest

class MyFeatureTest(unittest.TestCase):
    def test_addition(self):
        result = 1 + 1
        assert result == 3, "Addition failed" # Vague error message
```

**✅ GOOD: `unittest.TestCase` and `self.assert*` methods**
```python
import unittest

class MyFeatureTest(unittest.TestCase):
    def test_addition(self0):
        result = 1 + 1
        self.assertEqual(result, 2, "Addition should be 2") # Specific error message
        self.assertNotEqual(result, 3)
```

## 4. Setup/Teardown

Utilize `setUp` and `tearDown` methods within your `unittest.TestCase` classes for per-test setup and cleanup. For class-level setup/teardown, use `setUpClass` and `tearDownClass`. Avoid module-level `setup()`/`teardown()` functions unless maintaining legacy `nose` suites.

**❌ BAD: Module-level setup/teardown (legacy `nose` style)**
```python
# test_db_operations.py
DB_CONNECTION = None

def setup_module():
    global DB_CONNECTION
    DB_CONNECTION = connect_to_db()

def teardown_module():
    DB_CONNECTION.close()

class DatabaseTest(unittest.TestCase):
    def test_query(self):
        # ... use DB_CONNECTION ...
        pass
```

**✅ GOOD: Class-level `setUp`/`tearDown` for fixtures**
```python
import unittest
from my_module.db import connect_to_db, disconnect_from_db

class DatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Connect to the database once for all tests in this class."""
        cls.db_connection = connect_to_db()

    @classmethod
    def tearDownClass(cls) -> None:
        """Disconnect from the database after all tests in this class."""
        disconnect_from_db(cls.db_connection)

    def setUp(self) -> None:
        """Reset database state before each test."""
        self.db_connection.clear_data()

    def test_insert_user(self) -> None:
        self.db_connection.insert("users", {"name": "Alice"})
        user = self.db_connection.get("users", "Alice")
        self.assertIsNotNone(user)

    def test_delete_user(self) -> None:
        self.db_connection.insert("users", {"name": "Bob"})
        self.db_connection.delete("users", "Bob")
        user = self.db_connection.get("users", "Bob")
        self.assertIsNone(user)
```

## 5. Parameterized Tests

Leverage `nose2.tools.params` for writing data-driven tests. This significantly reduces boilerplate when you need to run the same test logic with different input data. Ensure the `nose2.plugins.params` plugin is enabled in `pyproject.toml`.

**❌ BAD: Duplicating test logic for different inputs**
```python
import unittest

class CalculatorTest(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(1, 2), 3)
    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)
    def test_add_zero(self):
        self.assertEqual(add(0, 5), 5)
```

**✅ GOOD: Using `nose2.tools.params`**
```python
import unittest
from nose2.tools import params

class CalculatorTest(unittest.TestCase):
    @params(
        (1, 2, 3),
        (-1, -2, -3),
        (0, 5, 5),
        (10, -3, 7)
    )
    def test_add(self, a: int, b: int, expected: int) -> None:
        self.assertEqual(add(a, b), expected)
```

## 6. Coverage & CI Integration

Enable the `coverage` plugin and set `coverage-combine = true` in `pyproject.toml` to ensure accurate coverage reporting, especially in environments using multiprocessing. Run `nose2 -v` in your CI pipeline and enforce a coverage threshold to prevent regressions.

**Actionable Steps:**
1. Install `coverage`: `pip install coverage`
2. Configure `pyproject.toml` (as shown in section 2).
3. Run tests with coverage in CI: `nose2 -v`
4. Use `coverage report` or `coverage html` to view results.
5. Enforce a minimum coverage percentage in your CI/CD pipeline.

## 7. Mocking Strategies

Use Python's built-in `unittest.mock` library for isolating units under test from their dependencies. Prefer `mock.patch` as a decorator or context manager for clear and scoped mocking.

**❌ BAD: Manually replacing dependencies or global state**
```python
# my_module.py
import external_service

def process_data(data):
    return external_service.send(data)

# test_my_module.py
import unittest
import my_module
import external_service # This will be the real service

class MyModuleTest(unittest.TestCase):
    def test_process_data(self):
        # This is bad, it affects the actual external_service module
        original_send = external_service.send
        external_service.send = lambda x: "mocked_response"
        result = my_module.process_data("test_data")
        self.assertEqual(result, "mocked_response")
        external_service.send = original_send # Must remember to clean up
```

**✅ GOOD: Using `unittest.mock.patch`**
```python
import unittest
from unittest import mock
from my_module import process_data # Assuming process_data is in my_module.py

class MyModuleTest(unittest.TestCase):
    @mock.patch('my_module.external_service.send') # Patch the dependency where it's looked up
    def test_process_data_decorator(self, mock_send):
        mock_send.return_value = "mocked_response_from_decorator"
        result = process_data("test_data")
        mock_send.assert_called_once_with("test_data")
        self.assertEqual(result, "mocked_response_from_decorator")

    def test_process_data_context_manager(self):
        with mock.patch('my_module.external_service.send') as mock_send:
            mock_send.return_value = "mocked_response_from_context"
            result = process_data("another_data")
            mock_send.assert_called_once_with("another_data")
            self.assertEqual(result, "mocked_response_from_context")
```

## 8. Type Hints

Apply type hints consistently to your test code, especially for method signatures (`setUp`, `test_*`, `tearDown`). This improves readability, enables static analysis, and makes your tests easier to understand and maintain.

**❌ BAD: Untyped test methods**
```python
class MyTest(unittest.TestCase):
    def setUp(self):
        self.client = create_test_client()

    def test_endpoint(self):
        response = self.client.get('/api/data')
        self.assertEqual(response.status_code, 200)
```

**✅ GOOD: Type-hinted test methods**
```python
from typing import Any
import unittest

class MyTest(unittest.TestCase):
    client: Any # Define attributes used across methods

    def setUp(self) -> None:
        self.client = create_test_client()

    def test_endpoint(self) -> None:
        response = self.client.get('/api/data')
        self.assertEqual(response.status_code, 200)
```

## 9. Common Pitfalls & Gotchas

*   **Not staying current:** `nose2` is actively maintained. Always use the latest stable release (0.15.x as of May 2024) to benefit from bug fixes, new features, and Python version compatibility.
*   **Overly complex `setUp`:** Keep `setUp` methods focused on preparing the immediate test context. If setup becomes too complex, consider breaking tests into smaller, more focused classes or using factories.
*   **Mixing `nose` and `nose2` idioms:** `nose2` is a distinct project. Avoid relying on `nose`-specific features that `nose2` explicitly does not support (e.g., certain module-level functions or specific test discovery quirks).
*   **When to consider `pytest`:** If `nose2`'s `unittest`-centric approach feels restrictive, or you find yourself fighting its conventions, consider migrating to `pytest`. `pytest` offers a more "Pythonic" test syntax (raw `assert`), powerful fixtures, and a vast plugin ecosystem. `pytest` can run `nose`-style tests, making migration easier.