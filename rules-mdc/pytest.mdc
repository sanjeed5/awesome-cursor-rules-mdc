---
description: This guide defines our team's definitive best practices for writing fast, reliable, and maintainable tests using pytest, ensuring consistency and high code quality across projects.
globs: **/*.py
---
# pytest Best Practices

This document outlines our team's standard for writing `pytest` tests. Adhering to these guidelines ensures our test suite is fast, reliable, and easy to maintain, integrating seamlessly into our CI/CD pipelines.

## 1. Code Organization & Structure

### 1.1 Project Layout
Always use the `src/` layout for your project. This prevents common import pitfalls and ensures tests run against the installed package, not local source.

**✅ GOOD:**
```
.
├── pyproject.toml
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── module.py
└── tests/
    ├── __init__.py
    └── test_module.py
```

**❌ BAD:**
```
.
├── pyproject.toml
├── mypackage/ # Top-level package
│   ├── __init__.py
│   └── module.py
└── tests/
    └── test_module.py
```

### 1.2 Test File Naming
Name test files `test_*.py` or `*_test.py`. We prefer `test_*.py`.
Place `__init__.py` files in `tests/` and its subdirectories to treat them as Python packages, enabling structured imports and avoiding name collisions.

**✅ GOOD:**
```python
# tests/test_my_feature.py
from mypackage.module import my_function

def test_my_function_returns_expected_value():
    assert my_function() == "expected"
```

### 1.3 Test Naming Conventions
Prefix test functions and methods with `test_`. For classes, prefix with `Test`. Use descriptive names that read like sentences.

**✅ GOOD:**
```python
def test_user_can_be_created_with_valid_email():
    # ...
    pass

class TestUserService:
    def test_get_user_by_id_returns_correct_user():
        # ...
        pass
```

**❌ BAD:**
```python
def create_user(): # Not discovered by pytest
    pass

def test_create_user_and_email(): # Tests multiple things
    pass

class UserServiceTest: # Incorrect class prefix
    pass
```

## 2. Common Patterns & Anti-patterns

### 2.1 Single Assert Per Test
Each test should verify one specific behavior. This improves readability and simplifies debugging.

**✅ GOOD:**
```python
def test_user_creation_sets_email():
    user = create_user("test@example.com")
    assert user.email == "test@example.com"

def test_user_creation_sends_welcome_email():
    user = create_user("test@example.com")
    assert email_service.last_sent_email == "test@example.com"
```

**❌ BAD:**
```python
def test_user_creation_and_email_sending():
    user = create_user("test@example.com")
    assert user.email == "test@example.com" # First assert
    send_welcome_email(user)
    assert email_service.last_sent_email == "test@example.com" # Second assert
```

### 2.2 Fixtures for Setup/Teardown & Dependency Injection
Use fixtures to provide isolated, reusable resources and manage setup/teardown.

*   **Scope**:
    *   `function` (default): Run once per test function. Use for most tests.
    *   `class`: Run once per test class.
    *   `module`: Run once per test module. Use for expensive setups shared across module tests.
    *   `session`: Run once per test session. Use for global, very expensive resources (e.g., database connection pool).
*   **Teardown**: Use `yield` in fixtures for automatic teardown.
*   **Modularity**: Compose fixtures by injecting them into others.
*   **Avoid `autouse=True`**: Only use for truly global, non-interfering setups (e.g., patching a common library function for all tests).

**✅ GOOD:**
```python
# conftest.py
import pytest

@pytest.fixture(scope="function")
def db_session():
    # Setup: Create a fresh database session
    session = create_test_db_session()
    yield session  # Test runs here
    # Teardown: Close and clean up the session
    session.close()

@pytest.fixture(scope="module")
def api_client(db_session): # Fixtures can use other fixtures
    # Setup: Create an API client with a specific DB session
    client = TestClient(db_session)
    yield client
    # Teardown: Cleanup client resources
```

**❌ BAD:**
```python
# test_bad.py
class TestUser:
    def setup_method(self): # unittest-style setup, less flexible
        self.user = User("test@example.com")

    def test_user_email(self):
        assert self.user.email == "test@example.com"

@pytest.fixture(autouse=True) # Avoid unless absolutely necessary
def global_patch():
    # This fixture runs for *every* test, potentially causing unexpected side effects.
    pass
```

### 2.3 Parameterization
Use `@pytest.mark.parametrize` for data-driven tests.

**✅ GOOD:**
```python
import pytest

@pytest.mark.parametrize("input, expected_output", [
    (1, 2),
    (2, 3),
    (0, 1),
])
def test_increment_function(input, expected_output):
    assert increment(input) == expected_output
```

### 2.4 Markers
Use `@pytest.mark.skip` for tests that are not yet ready or temporarily broken. Use `@pytest.mark.xfail` for known bugs that are expected to fail but should still be run.

**✅ GOOD:**
```python
import pytest

@pytest.mark.skip(reason="Feature not implemented yet")
def test_new_feature_endpoint():
    pass

@pytest.mark.xfail(reason="Known bug: #1234")
def test_buggy_legacy_function():
    assert legacy_function() == "fixed_output"
```

## 3. Performance Considerations

### 3.1 Keep Tests Fast
Prioritize fast tests. Slow tests discourage local execution and lead to longer CI times.
*   **Mock external services**: Use mocks for APIs, databases, and file systems.
*   **Appropriate fixture scope**: Don't use `session` scope for resources that need to be fresh per test.
*   **Small, focused tests**: Each test should do minimal work.

## 4. Mocking Strategies

Use the `mocker` fixture (from `pytest-mock`) for robust mocking. It integrates `unittest.mock` functionality with `pytest`'s fixture system.

**✅ GOOD:**
```python
# test_service.py
from mypackage.service import MyService
from mypackage.external_api import get_data

def test_my_service_fetches_data(mocker):
    mock_get_data = mocker.patch("mypackage.service.get_data")
    mock_get_data.return_value = {"key": "mocked_value"}

    service = MyService()
    result = service.fetch_and_process()

    mock_get_data.assert_called_once()
    assert result == "processed_mocked_value"
```

**❌ BAD:**
```python
# test_service.py (using raw monkeypatch for complex mocks)
import pytest
from mypackage.service import MyService
from mypackage.external_api import get_data

def test_my_service_fetches_data_raw_monkeypatch(monkeypatch):
    # monkeypatch is fine for simple cases, but mocker is better for complex mocks
    # and provides more assert methods.
    def mock_get_data():
        return {"key": "mocked_value"}
    monkeypatch.setattr("mypackage.service.get_data", mock_get_data)

    service = MyService()
    result = service.fetch_and_process()
    assert result == "processed_mocked_value"
```

## 5. Coverage Patterns

Integrate `pytest-cov` to measure test coverage. Aim for high coverage, but understand that 100% coverage doesn't guarantee bug-free code. Focus on covering critical paths and edge cases.

**✅ GOOD:**
```bash
# Run tests with coverage and generate report
pytest --cov=mypackage --cov-report=term-missing --cov-report=html
```

## 6. Type Hints

Apply type hints to test code, especially for fixtures and complex test functions. This improves readability, maintainability, and enables static analysis.

**✅ GOOD:**
```python
from typing import Generator
import pytest

class User:
    def __init__(self, email: str):
        self.email = email

@pytest.fixture
def new_user() -> Generator[User, None, None]:
    user = User("test@example.com")
    yield user
    # Cleanup if necessary

def test_user_email_property(new_user: User) -> None:
    assert new_user.email == "test@example.com"
```