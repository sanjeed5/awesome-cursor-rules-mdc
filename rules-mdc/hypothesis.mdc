---
description: Definitive guidelines for writing robust, maintainable, and performant property-based tests using Hypothesis with Python and pytest.
globs: **/*.py
---
# hypothesis Best Practices

Hypothesis is a powerful property-based testing library. These guidelines ensure your Hypothesis tests are effective, maintainable, and integrate seamlessly into modern Python development workflows.

## 1. Code Organization and Structure

Always integrate Hypothesis with `pytest`. Keep test files in a dedicated `tests/` directory, mirroring your `src/` structure. Adhere to PEP 8, use clear docstrings, and apply type hints.

❌ BAD: Generic test function names, no `st` alias, mixed with application code.
```python
# my_app/utils.py
from hypothesis import given, strategies
def add(a, b): return a + b

@given(strategies.integers(), strategies.integers())
def test_add(a, b):
    assert add(a, b) == a + b
```

✅ GOOD: Dedicated test files, `st` alias, clear naming, `pytest` integration.
```python
# src/my_app/utils.py
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

# tests/test_utils.py
from hypothesis import given, strategies as st
import pytest
from src.my_app.utils import add

@given(st.integers(), st.integers())
def test_add_is_commutative(a: int, b: int) -> None:
    """Verify addition is commutative."""
    assert add(a, b) == add(b, a)

@given(st.integers())
def test_add_identity_element(a: int) -> None:
    """Verify zero is the identity element for addition."""
    assert add(a, 0) == a
```

## 2. Strategy Selection and Constraints

Always use the most specific and constrained strategies possible. This improves performance and focuses generated examples on relevant edge cases. Use `st.composite` for dependent generation.

❌ BAD: Overly broad strategies or excessive filtering with `assume()` for basic constraints.
```python
@given(st.integers(), st.integers())
def test_division(numerator, denominator):
    assume(denominator != 0) # Discards many examples
    assume(numerator % denominator == 0) # Discards even more
    assert numerator / denominator == numerator // denominator
```

✅ GOOD: Constrain strategies directly. Use `st.composite` for dependent values.
```python
from hypothesis import given, strategies as st, assume
from hypothesis.extra.math import non_zero_floats
from typing import Tuple

@given(st.integers(), st.integers(min_value=1, max_value=100))
def test_division_positive_divisor(numerator: int, divisor: int) -> None:
    """Test integer division with a positive divisor."""
    assert (numerator // divisor) * divisor + (numerator % divisor) == numerator

@st.composite
def non_zero_pairs(draw) -> Tuple[int, int]:
    """Generates a pair of integers where the second is non-zero."""
    numerator = draw(st.integers())
    denominator = draw(st.integers().filter(lambda d: d != 0))
    return numerator, denominator

@given(non_zero_pairs())
def test_division_any_non_zero(pair: Tuple[int, int]) -> None:
    """Test division property with any non-zero denominator."""
    numerator, denominator = pair
    # Property: (q * d) + r = n, where 0 <= abs(r) < abs(d)
    quotient = numerator // denominator
    remainder = numerator % denominator
    assert (quotient * denominator) + remainder == numerator
    assert abs(remainder) < abs(denominator)
```

## 3. Performance Considerations

Configure `hypothesis.settings` for appropriate `max_examples` and `deadline`. Use `pytest` fixtures for expensive setup.

❌ BAD: Slow tests due to high `max_examples` or expensive setup per run.
```python
from hypothesis import given, strategies as st, settings
import time

@settings(max_examples=10000) # Too many for local dev
@given(st.lists(st.integers(), min_size=1000, max_size=1000))
def test_expensive_sort(large_list):
    # Simulates expensive setup inside the test
    time.sleep(0.01)
    assert sorted(large_list) == custom_sort(large_list)
```

✅ GOOD: Sensible `settings`, `pytest` fixtures for setup.
```python
from hypothesis import given, strategies as st, settings, HealthCheck
import pytest
import time

# Global settings for CI/local runs. Override locally if needed.
# Use a specific profile for CI, e.g., @settings(my_ci_profile)
@settings(max_examples=200, deadline=500, suppress_health_check=[HealthCheck.filter_too_much])
@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_sort_correctness(data: list[int]) -> None:
    """Verify sorting algorithm maintains elements and order."""
    original_set = set(data)
    sorted_data = sorted(data)
    assert len(sorted_data) == len(data)
    assert set(sorted_data) == original_set
    assert all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data) - 1))
```

## 4. Common Pitfalls and Gotchas

Avoid `assume()` for core logic validation. Test invalid inputs explicitly. Replay failing examples immediately.

❌ BAD: Using `assume()` to hide bugs or for conditions that should be tested.
```python
import math
from hypothesis import given, strategies as st

@given(st.integers(min_value=-10, max_value=10))
def test_sqrt_positive(n):
    assume(n >= 0) # Hides behavior for negative numbers, might be a bug
    assert math.isclose(math.sqrt(n)**2, n)
```

✅ GOOD: Test invalid inputs explicitly or constrain strategies.
```python
import math
from hypothesis import given, strategies as st
import pytest

@given(st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
def test_sqrt_positive_numbers(x: float) -> None:
    """Verify sqrt property for non-negative floats."""
    assert math.isclose(math.sqrt(x)**2, x)

@given(st.floats(max_value=0.0, exclude_max=True, allow_nan=False, allow_infinity=False))
def test_sqrt_negative_numbers_raises_value_error(x: float) -> None:
    """Verify sqrt raises ValueError for negative floats."""
    with pytest.raises(ValueError):
        math.sqrt(x)
```

## 5. Test Organization and Replaying Failures

Run tests with `pytest` and use `--hypothesis-show-statistics` for insights. Hypothesis automatically saves and replays failing examples from the `.hypothesis/examples` database.

```bash
# Run all tests
pytest

# Run with Hypothesis statistics
pytest --hypothesis-show-statistics

# Replay a specific failing example (Hypothesis saves them)
# If a test fails, pytest will automatically replay the minimal falsifying example
# from the .hypothesis/examples directory on subsequent runs.
```

## 6. Mocking Strategies

Use `pytest-mock` (the `mocker` fixture) for mocking external dependencies. Ensure mocks are scoped correctly (function scope is usually best).

❌ BAD: Manual mocking or global mocks that persist across tests.
```python
import os
from hypothesis import given, strategies as st

@given(st.text())
def test_read_from_env(key):
    # This modifies the actual environment!
    os.environ['TEST_KEY'] = key
    assert get_env_var('TEST_KEY') == key
    del os.environ['TEST_KEY'] # Cleanup is error-prone
```

✅ GOOD: Use `mocker` fixture for safe, isolated mocks.
```python
import os
from hypothesis import given, strategies as st
import pytest
from pytest_mock import MockerFixture

def get_env_var(key: str) -> str | None:
    return os.getenv(key)

@given(st.text(min_size=1), st.text())
def test_get_env_var_with_mocker(mocker: MockerFixture, key: str, value: str) -> None:
    """Verify environment variable retrieval using mocker."""
    mocker.patch.dict(os.environ, {key: value})
    assert get_env_var(key) == value

@given(st.text(min_size=1))
def test_get_env_var_missing(mocker: MockerFixture, key: str) -> None:
    """Verify missing environment variable returns None."""
    mocker.patch.dict(os.environ, {}, clear=True) # Ensure env is clean
    assert get_env_var(key) is None
```

## 7. Type Hints

Always type hint your test function arguments and custom strategies. This improves readability, maintainability, and enables static analysis.

❌ BAD: Untyped arguments, unclear strategy return types.
```python
from hypothesis import given, strategies as st

@given(st.integers(), st.text())
def test_process(num, text):
    # What are num and text?
    assert isinstance(num, int)
    assert isinstance(text, str)
```

✅ GOOD: Explicit type hints for clarity and static analysis.
```python
from hypothesis import given, strategies as st
from typing import Any, TypeVar, Tuple

T = TypeVar('T')

@st.composite
def custom_data_strategy(draw) -> Tuple[int, str]:
    """Generates a tuple of an integer and a string."""
    num = draw(st.integers(min_value=0, max_value=100))
    text = draw(st.text(min_size=1, max_size=10))
    return num, text

@given(custom_data_strategy())
def test_process_data(data: Tuple[int, str]) -> None:
    """Process custom generated data."""
    num, text = data
    assert isinstance(num, int)
    assert isinstance(text, str)
    assert 0 <= num <= 100
    assert 1 <= len(text) <= 10
```