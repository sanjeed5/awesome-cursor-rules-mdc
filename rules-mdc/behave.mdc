---
description: Definitive guidelines for writing maintainable, readable, and effective BDD tests with behave in Python, emphasizing thin steps, clear Gherkin, and robust fixture management.
globs: **/*.py
---
# behave Best Practices

This guide outlines the definitive best practices for writing `behave` tests. Adhere to these standards for consistent, high-quality BDD automation.

## 1. Code Organization & Structure

Maintain a clear, predictable project structure.

**✅ GOOD: Standard Project Layout**

```
.
├── features/
│   ├── authentication.feature   # Gherkin feature files
│   ├── product_management.feature
│   ├── environment.py           # Global hooks and fixtures
│   └── steps/
│       ├── common_steps.py      # Reusable, generic steps
│       ├── auth_steps.py        # Authentication-specific steps
│       └── product_steps.py     # Product-specific steps
├── src/                         # Your application code
│   └── my_app/
│       ├── __init__.py
│       ├── models.py
│       └── services.py          # Business logic layer
└── pyproject.toml               # Project configuration
```

## 2. Feature Files: Clear, Business-Centric Gherkin

Feature files describe *what* the system does from a business perspective, not *how* it's implemented. Keep scenarios concise.

**❌ BAD: Implementation Details in Gherkin**

```gherkin
Feature: User Login
  Scenario: Successful login
    Given I am on the "/login" page
    When I fill in "username" with "testuser" and "password" with "password123"
    And I click the "Login" button with CSS selector ".login-form button"
    Then I should be redirected to "/dashboard"
```

**✅ GOOD: Abstract, User-Focused Gherkin**

```gherkin
Feature: User Login
  Scenario: Successful login with valid credentials
    Given a registered user "testuser" with password "password123"
    When "testuser" logs in
    Then "testuser" should be logged in successfully
```

## 3. Step Implementations: Thin & Pythonic

Steps should be thin wrappers that delegate complex logic to your application's service layer or dedicated helper modules. Follow PEP 8.

**❌ BAD: Fat Steps with Business Logic**

```python
# features/steps/auth_steps.py
from behave import given
from my_app.models import User # Direct ORM access

@given('a registered user "{username}" with password "{password}"')
def step_impl(context, username, password):
    # Direct database manipulation and complex logic in step
    user = User.objects.create_user(username=username, password=password)
    context.user = user
    # ... more setup logic
```

**✅ GOOD: Thin Steps Delegating to Service Layer**

```python
# features/steps/auth_steps.py
from behave import given, when, then
from my_app.services import user_service # Delegate to service layer

@given('a registered user "{username}" with password "{password}"')
def step_impl(context, username: str, password: str):
    context.user = user_service.create_user(username, password)

@when('"{username}" logs in')
def step_impl(context, username: str):
    # Use a test client or API helper from service
    context.response = user_service.login_user(username, context.user.password)

@then('"{username}" should be logged in successfully')
def step_impl(context, username: str):
    assert context.response.status_code == 200
    assert user_service.is_logged_in(username)
```

## 4. Context Object: State Management

Use the `context` object to share state between steps within a scenario. Avoid polluting it with unnecessary or complex data structures.

**✅ GOOD: Minimal & Typed Context Usage**

```python
# features/steps/product_steps.py
from behave import given, when, then

@given('a product with name "{name}" and price {price:f}')
def step_impl(context, name: str, price: float):
    context.product_name = name
    context.product_price = price

@when('I add the product to the cart')
def step_impl(context):
    # Access context.product_name and context.product_price
    # Delegate to a service: cart_service.add_item(context.user, context.product_name, context.product_price)
    pass
```

## 5. Fixtures & Hooks: `environment.py`

Manage setup and teardown declaratively in `features/environment.py`. Use `behave`'s built-in fixture system.

**✅ GOOD: Centralized Fixture Management**

```python
# features/environment.py
from behave.fixture import use_fixture_by_tag
from features.fixtures import browser_session, database_cleanup

def before_all(context):
    # Global setup, e.g., configure logging, start external services
    context.config.setup_logging()

def before_feature(context, feature):
    # Setup before each feature
    pass

def before_scenario(context, scenario):
    # Setup before each scenario, apply fixtures based on tags
    use_fixture_by_tag(context, scenario.tags, "browser.session", browser_session)
    use_fixture_by_tag(context, scenario.tags, "database.cleanup", database_cleanup)

def after_scenario(context, scenario):
    # Teardown after each scenario
    pass

def after_all(context):
    # Global teardown, e.g., stop external services
    pass

# features/fixtures.py (example fixture definitions)
from behave import fixture

@fixture
def browser_session(context):
    # Setup browser instance
    context.browser = "some_selenium_browser_instance"
    yield context.browser # Yield control to scenario
    # Teardown browser
    context.browser.quit()

@fixture
def database_cleanup(context):
    # Setup database state (e.g., clear tables, start transaction)
    context.db_transaction = "start_transaction"
    yield # Yield control to scenario
    # Teardown database state (e.g., rollback transaction)
    context.db_transaction.rollback()
```

**Gherkin with Tags to trigger fixtures:**

```gherkin
@browser.session @database.cleanup
Feature: User Interface Interaction
  Scenario: Verify homepage content
    Given I open the browser
    When I navigate to the homepage
    Then I should see "Welcome"
```

## 6. Tagging for Test Management

Use tags (`@tag_name`) to categorize scenarios for selective execution (e.g., `@smoke`, `@regression`, `@slow`).

**✅ GOOD: Granular Tagging**

```gherkin
@smoke @authentication
Feature: User Login
  Scenario: Successful login
    # ...
@slow @integration @database_heavy
Feature: Order Processing
  Scenario: Complete order workflow
    # ...
```

**Running Tests with Tags:**

```bash
# Run only smoke tests
behave -t @smoke

# Run integration tests but exclude slow ones
behave -t "@integration and not @slow"
```

## 7. Mocking Strategies

Mock external dependencies at the service layer, not directly within steps. Use `unittest.mock` for controlled isolation.

**✅ GOOD: Mocking at Service Layer**

```python
# my_app/services.py
import requests

def get_external_data(item_id: str) -> dict:
    response = requests.get(f"https://api.external.com/items/{item_id}")
    response.raise_for_status()
    return response.json()

# features/steps/data_steps.py
from behave import when, then
from unittest.mock import patch
from my_app.services import get_external_data

@when('I request external data for "{item_id}"')
def step_impl(context, item_id: str):
    # Patch the dependency within the service layer
    with patch('my_app.services.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"id": item_id, "data": "mocked"}
        context.external_data = get_external_data(item_id)

@then('the external data should be mocked')
def step_impl(context, item_id: str):
    assert context.external_data["data"] == "mocked"
```

## 8. Type Hints

Apply type hints to step function arguments and helper functions for improved readability, maintainability, and static analysis.

**✅ GOOD: Explicit Type Hints**

```python
from behave import given, when, then

@given('a number {num:d}')
def step_impl(context, num: int):
    context.number = num

@when('I add {value:d} to the number')
def step_impl(context, value: int):
    context.number += value

@then('the number should be {expected:d}')
def step_impl(context, expected: int):
    assert context.number == expected
```

## 9. Coverage Patterns

Focus code coverage on your application's business logic, not the `behave` step definitions themselves.

**✅ GOOD: Measure Application Code Coverage**

Configure `coverage.py` (or `pytest-cov` if integrating with `pytest`) to include your `src/` directory and explicitly exclude `features/` and `features/steps/`.

```ini
# .coveragerc
[run]
source = src/my_app/ # Target your application code
omit =
    features/*
    features/steps/*
```

## 10. Performance Considerations

Optimize for speed by isolating slow tests and using efficient setup/teardown.

**✅ GOOD: Isolate Slow Tests with Tags**

```gherkin
@slow @database_heavy
Feature: Complex Reporting
  Scenario: Generate large report
    # ... This scenario takes a significant amount of time
```

Run slow tests separately: `behave -t @slow`

**✅ GOOD: Efficient Fixtures**

Ensure `before_scenario` and `after_scenario` hooks are as fast as possible. Use `before_all` for truly global, one-time setup.

```python
# features/environment.py
import some_db_module

def before_all(context):
    # Only run expensive setup once for the entire test suite
    context.db_connection_pool = some_db_module.create_connection_pool()

def after_all(context):
    # Only run expensive teardown once
    context.db_connection_pool.close_all()

def before_scenario(context, scenario):
    # Fast, isolated setup per scenario
    context.db_session = context.db_connection_pool.get_session()

def after_scenario(context, scenario):
    # Fast, isolated teardown per scenario
    context.db_session.close()
```