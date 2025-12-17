---
description: Definitive guidelines for writing robust, maintainable, and idiomatic `unittest` tests in Python. Focuses on structure, naming, assertions, mocking, and performance.
globs: **/*.py
---
# unittest Best Practices

This guide establishes the definitive best practices for writing `unittest` tests in Python. Adhering to these rules ensures our test suite is fast, reliable, readable, and maintainable, integrating seamlessly into our CI/CD pipeline.

## 1. Code Organization and Structure

Organize tests logically within a dedicated `tests/` directory, mirroring your application's module structure.

### Test File Naming
Name test files clearly to enable automatic discovery and indicate their purpose.

❌ BAD:
```python
# my_app/tests/users.py
# my_app/tests/test_user_logic.py
```

✅ GOOD:
```python
# my_app/tests/test_users.py
# my_app/tests/test_auth_service.py
```

### Test Class Naming
All test classes *must* inherit from `unittest.TestCase` and start with `Test`.

❌ BAD:
```python
import unittest

class UserServiceTests(unittest.TestCase):
    # ...
```

✅ GOOD:
```python
import unittest

class TestUserService(unittest.TestCase):
    # ...
```

### Test Method Naming
Each test method *must* start with `test_` and describe the specific scenario and expected outcome.

❌ BAD:
```python
class TestUserService(unittest.TestCase):
    def create_user(self): # Not a test method, but might be mistaken for one
        pass
    def test_valid_user(self): # Vague
        pass
```

✅ GOOD:
```python
class TestUserService(unittest.TestCase):
    def test_create_user_with_valid_data_returns_user_object(self):
        pass
    def test_create_user_with_existing_email_raises_value_error(self):
        pass
```

### Running Tests
Always include the standard `if __name__ == '__main__':` block for direct execution and discovery.

```python
import unittest

class TestExample(unittest.TestCase):
    def test_something(self) -> None:
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
```

## 2. Assertions: Single Logical Assertion

Each test method should focus on verifying *one logical outcome*. This makes tests easier to understand and debug.

❌ BAD:
```python
class TestCalculator(unittest.TestCase):
    def test_add_and_subtract(self) -> None:
        calc = Calculator()
        self.assertEqual(calc.add(2, 3), 5)
        self.assertEqual(calc.subtract(5, 2), 3) # Tests two distinct things
```

✅ GOOD:
```python
class TestCalculator(unittest.TestCase):
    def test_add_returns_correct_sum(self) -> None:
        calc = Calculator()
        self.assertEqual(calc.add(2, 3), 5)

    def test_subtract_returns_correct_difference(self) -> None:
        calc = Calculator()
        self.assertEqual(calc.subtract(5, 2), 3)
```

### Use Specific Assert Methods
Leverage `unittest`'s rich set of assert methods for clarity and better error messages.

❌ BAD:
```python
class TestUser(unittest.TestCase):
    def test_user_is_active(self) -> None:
        user = User(active=True)
        self.assertTrue(user.active == True, "User should be active") # Redundant comparison
        self.assertTrue(user.name in ["Alice", "Bob"]) # Generic, less informative error
```

✅ GOOD:
```python
class TestUser(unittest.TestCase):
    def test_user_is_active(self) -> None:
        user = User(active=True)
        self.assertTrue(user.active) # Direct assertion

    def test_user_name_is_in_allowed_list(self) -> None:
        user = User(name="Alice")
        self.assertIn(user.name, ["Alice", "Bob"]) # Specific for membership
```

## 3. Fixtures (`setUp`/`tearDown`)

Use fixtures to establish a clean, isolated state for tests.

### `setUp` and `tearDown` (Per Test Method)
Use `setUp` to prepare resources *before each test method* and `tearDown` to clean up *after each test method*.

```python
import unittest
from unittest.mock import MagicMock

class TestDatabaseOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.db_connection = MagicMock() # Mock a DB connection
        self.db_connection.create_table("users")

    def tearDown(self) -> None:
        self.db_connection.drop_table("users")
        self.db_connection.close()

    def test_add_user_to_db(self) -> None:
        self.db_connection.insert("users", {"name": "Alice"})
        self.db_connection.insert.assert_called_once_with("users", {"name": "Alice"})
```

### `setUpClass` and `tearDownClass` (Per Test Class)
Use `setUpClass` to prepare resources *once for all test methods in the class* and `tearDownClass` to clean up *once after all test methods*. This is suitable for expensive, shared resources.

```python
import unittest
from unittest.mock import MagicMock

class TestExpensiveResource(unittest.TestCase):
    expensive_resource: MagicMock

    @classmethod
    def setUpClass(cls) -> None:
        cls.expensive_resource = MagicMock() # e.g., a test server
        cls.expensive_resource.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.expensive_resource.stop()

    def test_resource_is_available(self) -> None:
        self.expensive_resource.is_running.return_value = True
        self.assertTrue(self.expensive_resource.is_running())
```

## 4. Mocking Strategies (`unittest.mock`)

Isolate the unit under test by replacing external dependencies with `MagicMock` objects.

### Use `patch` Decorator
For simple patching, the `@patch` decorator is concise.

```python
from unittest.mock import patch
import unittest

# Assume 'my_module' has a function 'get_external_data'
# and 'my_service' uses 'my_module.get_external_data'
class MyService:
    def process_data(self) -> str:
        data = my_module.get_external_data()
        return f"processed_{data['key']}"

class TestMyService(unittest.TestCase):
    @patch('my_module.get_external_data')
    def test_service_processes_data_correctly(self, mock_get_data: MagicMock) -> None:
        mock_get_data.return_value = {"key": "mocked_value"}
        service = MyService()
        result = service.process_data()
        self.assertEqual(result, "processed_mocked_value")
        mock_get_data.assert_called_once()
```

### Use `patch` as a Context Manager
For patching within a specific block of code, use `with patch(...)`.

```python
from unittest.mock import patch, MagicMock
import unittest

class TestMyService(unittest.TestCase):
    def test_service_handles_no_data(self) -> None:
        with patch('my_module.get_external_data') as mock_get_data:
            mock_get_data.return_value = None
            service = MyService()
            result = service.process_data()
            self.assertIsNone(result)
```

### Avoid Over-Mocking
Only mock external boundaries or complex dependencies. Do not mock internal logic or objects that are part of the unit under test.

❌ BAD:
```python
# Testing a 'User' class where 'is_admin' is a simple property
class TestUser(unittest.TestCase):
    @patch.object(User, 'is_admin') # Mocking an internal property
    def test_admin_user_can_access_panel(self, mock_is_admin: MagicMock) -> None:
        user = User(role='admin')
        mock_is_admin.return_value = True
        self.assertTrue(user.can_access_admin_panel())
```

✅ GOOD:
```python
# Testing a 'User' class where 'is_admin' is a simple property
class TestUser(unittest.TestCase):
    def test_admin_user_can_access_panel(self) -> None:
        user = User(role='admin') # Direct instantiation
        self.assertTrue(user.can_access_admin_panel())
```

## 5. Performance Considerations

Fast tests encourage frequent execution.

### Isolate Tests
Ensure tests are independent and don't rely on the order of execution or shared mutable state. Use `setUp`/`tearDown` to reset state.

### Mock Slow I/O
Replace database calls, network requests, and file system operations with mocks to avoid performance bottlenecks.

### Separate Integration Tests
If a test *must* interact with a real external system (e.g., a database), mark it as an integration test and run it separately from fast unit tests. `unittest` doesn't have built-in marking like `pytest`, so use naming conventions (e.g., `test_integration_*.py` files).

## 6. Type Hints in Tests

Apply type hints to test methods, fixture methods, and local variables for improved readability, maintainability, and static analysis.

```python
import unittest
from typing import Any
from unittest.mock import MagicMock

class TestDatabaseOperations(unittest.TestCase):
    db_connection: MagicMock # Type hint for instance variable

    def setUp(self) -> None:
        self.db_connection = MagicMock()
        self.db_connection.create_table("users")

    def tearDown(self) -> None:
        self.db_connection.drop_table("users")
        self.db_connection.close()

    def test_add_user_to_db(self) -> None:
        user_name: str = "Alice" # Type hint for local variable
        self.db_connection.insert("users", {"name": user_name})
        self.db_connection.insert.assert_called_once_with("users", {"name": user_name})
```

## 7. Coverage Patterns

While `unittest` doesn't provide coverage tools, integrate `coverage.py` to ensure thorough testing.

### Enforce Coverage Thresholds
Configure `coverage.py` in `pyproject.toml` or `setup.cfg` to fail CI builds if coverage drops below a set percentage.

```toml
# pyproject.toml (example for coverage.py)
[tool.coverage.run]
source = ["my_app"] # Specify which code to measure

[tool.coverage.report]
fail_under = 90 # Fail if total coverage is below 90%
show_missing = true
```

### Focus on Branches and Statements
Prioritize covering critical paths, edge cases, and error handling over simply reaching 100% line coverage.