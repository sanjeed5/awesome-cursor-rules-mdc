---
description: This guide provides opinionated, actionable best practices for writing robust, performant, and maintainable Python code using SciPy, aligning with modern scientific computing standards.
globs: **/*.py
---
# SciPy Best Practices

This document outlines the definitive best practices for developing with SciPy. Adhering to these guidelines ensures your code is clean, efficient, and compatible with the broader scientific Python ecosystem.

## 1. Code Organization and Style

Strict adherence to style guides is non-negotiable for maintainability.

### 1.1. Line Length and Docstrings

Follow PEP 8 with a strict 88-character line limit. Docstrings must conform to PEP 257 and the NumPy docstring standard, using ASCII characters primarily, with a small whitelist of Unicode symbols.

❌ **BAD**
```python
def calculate_complex_metric(data_array, method='default', tolerance=1e-6, max_iterations=1000, verbose_output=False):
    """Calculates a very complex metric using various parameters and returns a tuple of results."""
    # ... implementation ...
    return result1, result2, result3
```

✅ **GOOD**
```python
def calculate_complex_metric(
    data_array: np.ndarray,
    *,
    method: str = 'default',
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    verbose_output: bool = False,
) -> MyResultObject:
    """Calculate a complex metric from `data_array`.

    Parameters
    ----------
    data_array : np.ndarray
        Input data array.
    method : {'default', 'optimized'}, optional
        Algorithm method to use. Default is 'default'.
    tolerance : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iterations : int, optional
        Maximum number of iterations. Default is 1000.
    verbose_output : bool, optional
        If True, print detailed progress. Default is False.

    Returns
    -------
    res : MyResultObject
        An object with attributes:
        statistic : float
            The primary calculated statistic.
        pvalue : float
            The associated p-value.
        converged : bool
            True if the algorithm converged.
    """
    # ... implementation ...
    return MyResultObject(statistic=..., pvalue=..., converged=...)

@dataclass
class MyResultObject:
    statistic: float
    pvalue: float
    converged: bool
```

### 1.2. Keyword-Only Arguments

For functions with more than a few arguments, enforce keyword-only arguments after the initial "obvious" ones using `*`. This improves readability and allows for backward-compatible additions of new parameters.

❌ **BAD**
```python
def process_data(data, 0.1, 'linear', True): # What do these mean?
    pass
```

✅ **GOOD**
```python
def process_data(data: np.ndarray, *, threshold: float, method: str, normalize: bool):
    """Process data with explicit keyword arguments."""
    # ...
    pass

process_data(my_array, threshold=0.1, method='linear', normalize=True)
```

### 1.3. Return Non-Iterable Objects

When returning two or more conceptually distinct elements, always use a custom class or an existing SciPy return class (e.g., `OptimizeResult`). Avoid `tuple` or `namedtuple` to force explicit attribute access and enable future extensions.

❌ **BAD**
```python
def analyze_signal(signal: np.ndarray) -> tuple[float, float]:
    # ...
    return peak_freq, amplitude
```

✅ **GOOD**
```python
from dataclasses import dataclass

@dataclass
class SignalAnalysisResult:
    peak_frequency: float
    amplitude: float

def analyze_signal(signal: np.ndarray) -> SignalAnalysisResult:
    # ...
    return SignalAnalysisResult(peak_frequency=peak_freq, amplitude=amplitude)

result = analyze_signal(my_signal)
print(result.peak_frequency)
```

## 2. Common Patterns and Anti-patterns

Leverage modern SciPy features and avoid deprecated interfaces.

### 2.1. Array API Standard Support

Enable `SCIPY_ARRAY_API=1` for seamless interoperability with array libraries like PyTorch or JAX. This enforces stricter input validation and ensures `array type in equals array type out`.

❌ **BAD**
```python
# Relying on implicit NumPy conversion, potentially slow or incompatible
import numpy as np
from scipy.cluster.vq import vq

features = np.asmatrix([[1.9, 2.3, 1.7], [1.5, 2.5, 2.2]])
code_book = np.array([[1., 1., 1.], [2., 2., 2.]])
code, dist = vq(features, code_book) # Works without SCIPY_ARRAY_API=1, but matrix is deprecated
```

✅ **GOOD**
```python
# Set environment variable before import for Array API support
# export SCIPY_ARRAY_API=1
import numpy as np
import torch
from scipy.cluster.vq import vq

# Use standard arrays (NumPy or compatible tensors)
features_np = np.array([[1.9, 2.3, 1.7], [1.5, 2.5, 2.2]])
code_book_np = np.array([[1., 1., 1.], [2., 2., 2.]])
code_np, dist_np = vq(features_np, code_book_np)

features_torch = torch.tensor([[1.9, 2.3, 1.7], [1.5, 2.5, 2.2]])
code_book_torch = torch.tensor([[1., 1., 1.], [2., 2., 2.]])
code_torch, dist_torch = vq(features_torch, code_book_torch)

# Avoid np.matrix, np.ma.MaskedArray, or object dtypes when Array API is enabled
# This will raise TypeError:
# vq(np.asmatrix(features_np), code_book_np)
```

## 3. Performance Considerations

Optimize strategically: "Make it work → make it reliable → make it fast."

### 3.1. Profiling

Never optimize without measuring. Use `timeit` for micro-benchmarks and `cProfile` (or `line_profiler`) for identifying bottlenecks in larger applications.

```python
import numpy as np
import timeit

# Micro-benchmark with timeit
setup_code = "import numpy as np; a = np.arange(1000)"
stmt1 = "a ** 2"
stmt2 = "a * a"

print(f"a ** 2: {timeit.timeit(stmt1, setup=setup_code, number=100000):.3f}s")
print(f"a * a: {timeit.timeit(stmt2, setup=setup_code, number=100000):.3f}s")

# For larger scripts, use cProfile:
# python -m cProfile -o my_profile.prof your_script.py
# Then analyze with `snakeviz my_profile.prof`
```

### 3.2. Algorithmic Optimization

Prioritize choosing the right SciPy algorithm over low-level code tweaks. SciPy's routines are highly optimized (often in C/Fortran).

❌ **BAD**
```python
# Manually implementing a simple optimization loop in Python
import numpy as np
def my_custom_optimizer(func, x0, lr=0.01, max_iter=100):
    x = x0
    for _ in range(max_iter):
        grad = (func(x + 1e-6) - func(x - 1e-6)) / (2 * 1e-6) # Numerical gradient
        x -= lr * grad
    return x
```

✅ **GOOD**
```python
# Leverage SciPy's robust and optimized optimization routines
from scipy.optimize import minimize
import numpy as np

def objective_function(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

x0 = np.array([0.0, 0.0])
result = minimize(objective_function, x0, method='BFGS')
print(f"Optimized x: {result.x}")
```

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that lead to subtle bugs or incorrect results.

### 4.1. Floating Point Comparisons

Always use `np.testing.assert_allclose` for floating-point array comparisons in tests. Avoid `assert_almost_equal` or `assert_approx_equal`.

❌ **BAD**
```python
import numpy as np
from numpy.testing import assert_almost_equal

arr1 = np.array([0.1 + 0.2, 0.3])
arr2 = np.array([0.3, 0.3])
assert_almost_equal(arr1, arr2) # Can fail due to precision issues
```

✅ **GOOD**
```python
import numpy as np
from numpy.testing import assert_allclose

arr1 = np.array([0.1 + 0.2, 0.3])
arr2 = np.array([0.3, 0.3])
assert_allclose(arr1, arr2, rtol=1e-7, atol=0) # Robust floating point comparison
```

### 4.2. Testing Exceptions and Warnings

Use `pytest.raises` and `pytest.warns` as context managers with the `match` argument to precisely test expected exceptions or warnings.

❌ **BAD**
```python
import numpy as np
from scipy import stats
from numpy.testing import assert_raises

def test_zmap_nan_policy_raise_bad():
    scores = np.array([1, 2, 3])
    compare = np.array([-8, -3, 2, 7, 12, np.nan])
    assert_raises(ValueError, stats.zmap, scores, compare, nan_policy='raise')
```

✅ **GOOD**
```python
import numpy as np
from scipy import stats
import pytest

def test_zmap_nan_policy_raise_good():
    scores = np.array([1, 2, 3])
    compare = np.array([-8, -3, 2, 7, 12, np.nan])
    with pytest.raises(ValueError, match='input contains nan'):
        stats.zmap(scores, compare, nan_policy='raise')
```

## 5. Type Hints

Embrace PEP 484-style type hints for improved code clarity, static analysis, and maintainability.

❌ **BAD**
```python
def process(data, config):
    # What are data and config?
    return data * config['scale']
```

✅ **GOOD**
```python
import numpy as np
from typing import Dict, Any

def process(data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Process numerical data using a configuration dictionary."""
    return data * config['scale']
```

## 6. Virtual Environments

Always use a virtual environment (`venv` or `conda`) to manage dependencies. This isolates your project's dependencies and prevents conflicts.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install SciPy and other dependencies
pip install scipy numpy pandas matplotlib
```

## 7. Packaging

For any project beyond a simple script, structure it as a proper Python package using `pyproject.toml`.

❌ **BAD**
```
my_script.py
data.csv
```

✅ **GOOD**
```
my_project/
├── pyproject.toml
├── README.md
├── my_project/
│   ├── __init__.py
│   ├── analysis.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_analysis.py
```

## 8. Testing Approaches

Comprehensive testing is crucial for reliable scientific code.

### 8.1. Pytest and Pre-commit Hooks

Use `pytest` as your testing framework. Integrate `pre-commit` hooks (e.g., `ruff`, `isort`, `black`) to automate style and linting checks before every commit.

```bash
# Install pre-commit
pip install pre-commit

# In your project root, create a .pre-commit-config.yaml
# Example:
# repos:
#   - repo: https://github.com/astral-sh/ruff-pre-commit
#     rev: v0.1.9
#     hooks:
#       - id: ruff
#         args: [--fix, --exit-non-zero-on-fix]
#   - repo: https://github.com/psf/black
#     rev: 23.12.1
#     hooks:
#       - id: black
#   - repo: https://github.com/PyCQA/isort
#     rev: 5.12.0
#     hooks:
#       - id: isort

# Install the hooks
pre-commit install

# Run tests
pytest
```