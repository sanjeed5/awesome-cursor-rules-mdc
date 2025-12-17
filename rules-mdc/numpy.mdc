---
description: This guide provides definitive, actionable best practices for writing high-performance, maintainable, and correct NumPy code, emphasizing vectorization, explicit dtypes, and modern GPU acceleration.
globs: **/*.py
---
# numpy Best Practices

NumPy is the bedrock of numerical computing in Python. To fully leverage its power for AI/ML and data science in 2025, we must write code that is not just functional but also fast, readable, and robust. This guide outlines our team's definitive best practices for NumPy.

## Core Principles

1.  **Vectorize Everything**: Eliminate Python loops over arrays. NumPy operations are implemented in C and are orders of magnitude faster.
2.  **Be Explicit with `dtype`**: Always specify array data types to prevent unexpected casting and optimize memory/performance.
3.  **Embrace GPU Acceleration**: Utilize CuPy-compatible calls for critical, large-scale computations.

## Code Organization and Structure

Organize NumPy-heavy logic into dedicated, well-named functions and modules.

### 1. Modular Design

Isolate numerical operations into functions that accept and return NumPy arrays. This improves testability and reusability.

❌ BAD: Monolithic script
```python
import numpy as np

def process_data_script(data_path):
    data = np.loadtxt(data_path)
    # ... many lines of processing ...
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    # ... more processing ...
    return normalized_data
```

✅ GOOD: Modular functions
```python
import numpy as np

def load_numerical_data(file_path: str) -> np.ndarray:
    """Loads numerical data from a file."""
    return np.loadtxt(file_path)

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to have zero mean and unit variance."""
    if arr.size == 0:
        return arr
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    # Avoid division by zero for constant arrays
    if std_val == 0:
        return np.zeros_like(arr)
    return (arr - mean_val) / std_val

def process_pipeline(data_path: str) -> np.ndarray:
    """Orchestrates data loading and processing."""
    data = load_numerical_data(data_path)
    processed_data = normalize_array(data)
    return processed_data
```

### 2. Clear Naming and Docstrings

Adhere to PEP 8 for variable and function names (`snake_case`). Use comprehensive docstrings (PEP 257) for all public functions, classes, and modules, especially for complex numerical logic.

❌ BAD: Ambiguous names, no docs
```python
def calc(a, b):
    # What does 'a' mean? What does 'b' mean?
    # What does this function do?
    return np.dot(a, b.T)
```

✅ GOOD: Descriptive names, clear docstrings
```python
import numpy as np

def calculate_covariance_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix for a given data matrix.

    Args:
        data_matrix: A 2D NumPy array where rows are observations
                     and columns are features.

    Returns:
        A 2D NumPy array representing the covariance matrix.
    """
    if data_matrix.ndim != 2:
        raise ValueError("Input data_matrix must be 2-dimensional.")
    
    # Center the data by subtracting the mean of each feature
    centered_data = data_matrix - np.mean(data_matrix, axis=0)
    
    # Calculate covariance matrix: (X^T * X) / (n - 1)
    covariance_matrix = np.cov(data_matrix, rowvar=False)
    return covariance_matrix
```

## Common Patterns and Anti-patterns

### 1. Vectorization over Python Loops

**Always** prefer NumPy's vectorized operations and broadcasting over explicit Python `for` loops when working with arrays.

❌ BAD: Element-wise loop
```python
import numpy as np

data = np.random.rand(1_000_000)
result = np.empty_like(data)
for i in range(len(data)):
    result[i] = data[i] * 2 + 5 # Slow for large arrays
```

✅ GOOD: Vectorized operation
```python
import numpy as np

data = np.random.rand(1_000_000)
result = data * 2 + 5 # Fast and concise
```

### 2. Broadcasting

Leverage broadcasting for operations between arrays of different shapes. It's efficient and avoids unnecessary memory allocation.

❌ BAD: Manual tiling or explicit loops for scalar/vector operations
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = np.empty_like(matrix)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        result[i, j] = matrix[i, j] + scalar # Avoid
```

✅ GOOD: Broadcasting
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = matrix + scalar # NumPy handles this efficiently
```

## Performance Considerations

### 1. Explicit `dtype` Specification

**Always** specify `dtype` when creating arrays or performing operations that might change types. This prevents unexpected memory usage and performance penalties from implicit type conversions.

❌ BAD: Implicit `dtype`
```python
import numpy as np

# Python integers become int64 by default, even if int32 is sufficient
arr = np.array([1, 2, 3]) 
# float operations might implicitly upcast, consuming more memory
result = arr / 2.0 
```

✅ GOOD: Explicit `dtype`
```python
import numpy as np

# Use the smallest appropriate dtype
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Ensure consistent dtypes in operations
result = arr_int.astype(np.float32) / 2.0
```

### 2. Leverage GPU Acceleration (CuPy-compatible)

For computationally intensive tasks on large arrays, especially in AI/ML, integrate GPU-accelerated libraries like CuPy. Write functions that can seamlessly switch between NumPy and CuPy.

```python
import numpy as np
try:
    import cupy as cp
    _has_cupy = True
except ImportError:
    _has_cupy = False

def matrix_multiply(a: np.ndarray, b: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Performs matrix multiplication, optionally using GPU if available.
    """
    xp = cp if use_gpu and _has_cupy else np
    
    a_xp = xp.asarray(a)
    b_xp = xp.asarray(b)
    
    result = xp.dot(a_xp, b_xp)
    
    return result.get() if use_gpu and _has_cupy else result

# Example usage
matrix_a = np.random.rand(1000, 500)
matrix_b = np.random.rand(500, 2000)

# CPU computation
cpu_result = matrix_multiply(matrix_a, matrix_b, use_gpu=False)

# GPU computation (if CuPy installed)
if _has_cupy:
    gpu_result = matrix_multiply(matrix_a, matrix_b, use_gpu=True)
    assert np.allclose(cpu_result, gpu_result)
```

### 3. Pre-allocate Arrays

When filling an array iteratively (e.g., in a loop where vectorization isn't fully possible), pre-allocate the array with `np.empty` or `np.zeros` to avoid costly reallocations.

❌ BAD: Appending to a list then converting
```python
import numpy as np

results = []
for i in range(1000):
    results.append(np.random.rand(10))
final_array = np.array(results) # Inefficient for many appends
```

✅ GOOD: Pre-allocation
```python
import numpy as np

num_iterations = 1000
array_shape = (10,)
final_array = np.empty((num_iterations,) + array_shape, dtype=np.float32)
for i in range(num_iterations):
    final_array[i] = np.random.rand(*array_shape) # Efficient
```

## Common Pitfalls and Gotchas

### 1. Shape Mismatches

NumPy operations are strict about array shapes. Understand broadcasting rules and use `reshape`, `transpose`, `newaxis`, or `squeeze` explicitly.

❌ BAD: Implicit shape assumptions
```python
import numpy as np

vec = np.array([1, 2, 3]) # (3,)
matrix = np.array([[10, 20, 30], [40, 50, 60]]) # (2, 3)
# This will raise a ValueError if not careful, e.g., matrix + vec.T
# Or produce unexpected results due to broadcasting rules
```

✅ GOOD: Explicit shape handling
```python
import numpy as np

vec_row = np.array([1, 2, 3]) # (3,)
vec_col = np.array([1, 2, 3])[:, np.newaxis] # (3, 1)
matrix = np.array([[10, 20, 30], [40, 50, 60]]) # (2, 3)

# Correctly add a row vector to each row of a matrix
result = matrix + vec_row # (2,3) + (3,) -> (2,3)

# Correctly add a column vector to each column of a matrix
result = matrix + vec_col # (2,3) + (3,1) -> (2,3)
```

### 2. View vs. Copy

Understand when NumPy returns a view (modifying it changes the original array) versus a copy. Operations like slicing often return views. Use `.copy()` explicitly when you need an independent array.

❌ BAD: Unintended modification
```python
import numpy as np

original_data = np.arange(10)
subset = original_data[2:5] # This is a view!
subset[:] = 99 # Modifies original_data
print(original_data) # Output: [ 0  1 99 99 99  5  6  7  8  9] - DANGER!
```

✅ GOOD: Explicitly create a copy when needed
```python
import numpy as np

original_data = np.arange(10)
subset_copy = original_data[2:5].copy() # This is a copy!
subset_copy[:] = 99 # Only modifies subset_copy
print(original_data) # Output: [0 1 2 3 4 5 6 7 8 9] - SAFE!
```

## Type Hints

**Always** use type hints for function signatures involving NumPy arrays. This improves code readability, enables static analysis, and catches errors early. Use `np.ndarray` and specify `dtype` where type precision is critical.

```python
from typing import Tuple
import numpy as np

def process_sensor_data(
    readings: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes sensor readings, separating values above/below a threshold.

    Args:
        readings: A 1D NumPy array of sensor readings (expected float32).
        threshold: The threshold value.

    Returns:
        A tuple containing two 1D NumPy arrays: (above_threshold, below_threshold).
    """
    if readings.dtype != np.float32:
        raise TypeError("Input 'readings' must be of dtype np.float32")

    above = readings[readings >= threshold]
    below = readings[readings < threshold]
    return above, below

# Example usage
data = np.random.rand(100).astype(np.float32)
high_values, low_values = process_sensor_data(data, threshold=0.7)
```

## Virtual Environments

**Always** use virtual environments (`venv` or `conda`) for every project. This isolates dependencies and prevents version conflicts.

❌ BAD: Installing packages globally
```bash
pip install numpy scipy pandas # Pollutes global Python environment
```

✅ GOOD: Using a virtual environment
```bash
# For venv
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas

# For conda
conda create -n my_project_env python=3.10
conda activate my_project_env
conda install numpy scipy pandas
```

## Packaging

For any reusable NumPy code, package it as a standard Python library. Use `pyproject.toml` for modern packaging.

```toml
# pyproject.toml
[project]
name = "my_numpy_utils"
version = "0.1.0"
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.12.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## Testing Approaches

**Always** write unit tests for your NumPy code. Numerical stability and correctness are paramount.

### 1. `numpy.testing` for Numerical Comparisons

Use `numpy.testing` functions (e.g., `assert_allclose`, `assert_array_equal`) for comparing arrays, as standard equality checks can fail due to floating-point inaccuracies.

❌ BAD: Direct equality for floats
```python
import numpy as np

a = np.array([0.1 + 0.2])
b = np.array([0.3])
assert a == b # Fails due to floating point precision
```

✅ GOOD: `numpy.testing.assert_allclose`
```python
import numpy as np
from numpy.testing import assert_allclose

a = np.array([0.1 + 0.2])
b = np.array([0.3])
assert_allclose(a, b, rtol=1e-5, atol=1e-8) # Passes
```

### 2. `pytest` Integration

Integrate `numpy.testing` with `pytest` for a robust testing framework.

```python
# tests/test_my_module.py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from my_package.my_module import normalize_array, calculate_covariance_matrix

def test_normalize_array_basic():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = np.array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079], dtype=np.float32)
    result = normalize_array(arr)
    assert_allclose(result, expected, rtol=1e-6)

def test_normalize_array_constant():
    arr = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    result = normalize_array(arr)
    assert_allclose(result, expected)

def test_calculate_covariance_matrix():
    data = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.float64)
    expected_cov = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    result_cov = calculate_covariance_matrix(data)
    assert_allclose(result_cov, expected_cov, rtol=1e-9)

def test_calculate_covariance_matrix_raises_error_for_1d():
    data = np.array([1, 2, 3], dtype=np.float64)
    with np.testing.assert_raises(ValueError):
        calculate_covariance_matrix(data)
```