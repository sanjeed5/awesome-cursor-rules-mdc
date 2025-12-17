---
description: Definitive guidelines for writing high-performance, maintainable, and robust Numba-accelerated Python code, covering CPU and GPU targets.
globs: **/*.py
---
# Numba Best Practices

Numba is your go-to for accelerating numerical Python. Follow these rules to ensure your JIT-compiled code is fast, correct, and maintainable.

## Code Organization and Structure

### Isolate Numba-jitted code in dedicated modules.
Keep your Numba-accelerated functions separate from general Python logic. This creates clear boundaries, simplifies debugging, and prevents accidental object mode fallbacks.

❌ BAD: Mixing Numba with high-level logic
```python
import numba
import numpy as np
import pandas as pd

@numba.njit
def process_array(data):
    # Numba-optimized part
    return np.sqrt(data) * 2

def analyze_data(df: pd.DataFrame):
    # High-level Python logic, potentially calling Numba functions
    processed = process_array(df['value'].to_numpy())
    return pd.DataFrame({'processed': processed})
```

✅ GOOD: Dedicated module for accelerated functions
```python
# my_project/accelerated_math.py
import numba
import numpy as np

@numba.njit(fastmath=True, parallel=True)
def process_array(data: np.ndarray) -> np.ndarray:
    """Applies a numerical transformation to a NumPy array."""
    return np.sqrt(data) * 2

# my_project/data_analysis.py
import pandas as pd
from my_project.accelerated_math import process_array

def analyze_data(df: pd.DataFrame):
    processed_data = process_array(df['value'].to_numpy())
    return pd.DataFrame({'processed': processed_data})
```

## Common Patterns and Anti-patterns

### Always use `@njit` for CPU acceleration.
`@njit` (an alias for `@jit(nopython=True)`) forces Numba to compile your function entirely without the Python interpreter. If Numba cannot do this, it will raise an error, preventing silent performance degradation.

❌ BAD: Using `@jit` without `nopython=True`
```python
from numba import jit
import numpy as np

@jit # Could silently fall back to object mode, making it slow
def slow_sum(arr):
    total = 0
    for x in arr: # If 'x' becomes a Python object, this is slow
        total += x
    return total
```

✅ GOOD: Explicitly using `@njit`
```python
from numba import njit
import numpy as np

@njit # Will error if it cannot compile in nopython mode
def fast_sum(arr: np.ndarray) -> float:
    total = 0.0
    for x in arr:
        total += x
    return total
```

### Prefer `prange` for explicit CPU parallel loops with `@njit(parallel=True)`.
When `@njit(parallel=True)` is enabled, `prange` explicitly tells Numba to parallelize the loop across CPU cores, including reductions. For simple element-wise operations, Numba often auto-parallelizes, but `prange` gives you control for more complex loops.

❌ BAD: Using `range` with `parallel=True` for explicit parallelization
```python
from numba import njit
import numpy as np

@njit(parallel=True)
def sum_squares_bad(arr: np.ndarray) -> float:
    total = 0.0
    for i in range(arr.shape[0]): # Numba might not parallelize this loop as effectively
        total += arr[i] ** 2
    return total
```

✅ GOOD: Using `prange` for explicit parallelization
```python
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def sum_squares_good(arr: np.ndarray) -> float:
    total = 0.0
    for i in prange(arr.shape[0]): # Explicitly tells Numba to parallelize this loop
        total += arr[i] ** 2
    return total
```

### For CUDA, use `@cuda.jit` and manage the grid/block/thread hierarchy.
GPU kernels require explicit management of threads, blocks, and grids. Choose block sizes that are multiples of 32 (warp size) for optimal performance.

❌ BAD: Attempting to use `@njit` for GPU kernels
```python
from numba import njit
import numpy as np

@njit # This will compile for CPU, not GPU
def gpu_add_one(arr: np.ndarray):
    for i in range(arr.size):
        arr[i] += 1
```

✅ GOOD: Proper CUDA kernel declaration and invocation
```python
from numba import cuda
import numpy as np

@cuda.jit
def increment_by_one_kernel(an_array):
    idx = cuda.grid(1) # Get global linear thread index
    if idx < an_array.size:
        an_array[idx] += 1

def run_increment_on_gpu(host_array: np.ndarray):
    threadsperblock = 128 # Multiple of 32
    blockspergrid = (host_array.size + (threadsperblock - 1)) // threadsperblock

    device_array = cuda.to_device(host_array) # Transfer to device
    increment_by_one_kernel[blockspergrid, threadsperblock](device_array)
    device_array.copy_to_host(host_array) # Transfer back to host
```

## Performance Considerations

### Enable `fastmath=True` for floating-point heavy code.
If strict IEEE 754 compliance is not critical (e.g., in many AI/ML scenarios), `fastmath=True` allows Numba to perform aggressive floating-point optimizations, often leading to significant speedups.

❌ BAD: Missing `fastmath` for performance-critical calculations
```python
from numba import njit
import numpy as np

@njit # Default is fastmath=False
def calculate_trig(x: np.ndarray) -> np.ndarray:
    return np.cos(x)**2 + np.sin(x)**2
```

✅ GOOD: Enabling `fastmath`
```python
from numba import njit
import numpy as np

@njit(fastmath=True)
def calculate_trig_fast(x: np.ndarray) -> np.ndarray:
    return np.cos(x)**2 + np.sin(x)**2
```

### Install `icc_rt` for Intel SVML on x86/x64 platforms.
For Intel CPUs, installing the `icc_rt` package enables Numba to leverage Intel's Short Vector Math Library (SVML), providing highly optimized transcendental functions. This is a must-have for maximum performance.

```bash
conda install -c numba icc_rt
```

### Minimize host-device data transfers for CUDA.
Data transfer between CPU (host) and GPU (device) is a major bottleneck. Allocate device arrays once and reuse them across multiple kernel invocations (e.g., across training epochs).

❌ BAD: Repeated transfers in a loop
```python
from numba import cuda
import numpy as np

@cuda.jit
def my_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2

def process_data_bad(data_list: list[np.ndarray]):
    results = []
    for data in data_list:
        d_data = cuda.to_device(data) # Transfer every iteration
        my_kernel[..., ...](d_data)
        results.append(d_data.copy_to_host()) # Transfer back every iteration
    return results
```

✅ GOOD: Allocate once, reuse device memory
```python
from numba import cuda
import numpy as np

@cuda.jit
def my_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2

def process_data_good(data_list: list[np.ndarray]):
    if not data_list: return []
    
    # Allocate device memory once for the largest array or a fixed size
    max_size = max(arr.size for arr in data_list)
    d_buffer = cuda.device_array(max_size, dtype=data_list[0].dtype)

    results = []
    threadsperblock = 128
    for data in data_list:
        # Copy to pre-allocated device buffer
        d_buffer[:data.size] = cuda.to_device(data) 
        blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
        my_kernel[blockspergrid, threadsperblock](d_buffer[:data.size])
        results.append(d_buffer[:data.size].copy_to_host())
    return results
```

## Common Pitfalls and Gotchas

### Avoid Python objects and unsupported types inside `@njit` functions.
Numba's `@njit` mode works best with primitive types, NumPy arrays, and a subset of Python/NumPy functions. Using Python lists, dictionaries, or custom class instances inside `@njit` will either force object mode (slow) or cause compilation errors.

❌ BAD: Using Python lists or custom objects
```python
from numba import njit

class MyData:
    def __init__(self, value):
        self.value = value

@njit # Will fail or fall back to object mode
def process_objects(data_list: list[MyData]):
    total = 0
    for item in data_list: # Python list iteration
        total += item.value # Accessing custom object attribute
    return total
```

✅ GOOD: Using NumPy arrays
```python
from numba import njit
import numpy as np

@njit
def process_numpy_array(data_array: np.ndarray) -> float:
    total = 0.0
    for x in data_array: # NumPy array iteration
        total += x
    return total
```

### Time Numba code correctly by accounting for compilation overhead.
The first call to a Numba-jitted function includes compilation time. Subsequent calls with the same argument types use the cached compiled version. Always perform a "warm-up" call or use `timeit` for accurate benchmarks.

❌ BAD: Single timing measurement
```python
import time
from numba import njit
import numpy as np

@njit
def expensive_calc(arr):
    return np.sum(arr * 2 + 1)

data = np.random.rand(1_000_000)
start = time.perf_counter()
expensive_calc(data) # Includes compilation time
end = time.perf_counter()
print(f"Elapsed (BAD): {end - start:.6f}s")
```

✅ GOOD: Warm-up call or `timeit`
```python
import timeit
from numba import njit
import numpy as np

@njit
def expensive_calc(arr):
    return np.sum(arr * 2 + 1)

data = np.random.rand(1_000_000)

# Warm-up call
expensive_calc(data) 

# Now time the actual execution
elapsed = timeit.timeit(lambda: expensive_calc(data), number=100) / 100
print(f"Elapsed (GOOD, after compilation): {elapsed:.6f}s")
```

## Type Hints

### Use type hints for clarity; Numba infers types at runtime.
Numba performs its own type inference at runtime. While type hints improve code readability and enable static analysis tools, they do not directly influence Numba's compilation process.

```python
from numba import njit
import numpy as np

@njit
def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Adds two NumPy arrays element-wise."""
    return a + b
```

## Virtual Environments

### Always use `conda` for Numba installations, especially with CUDA/ROCm.
`conda` provides superior dependency management for Numba, particularly for complex native libraries like LLVM, CUDA Toolkit, and ROCm. This prevents common installation headaches.

```bash
# For CPU-only
conda install numba

# For NVIDIA CUDA GPU support
conda install numba cudatoolkit

# For AMD ROCm GPU support (Linux only)
conda install -c numba numba roctools
```

## Packaging

### Package Numba-accelerated code as standard Python packages.
Numba is a JIT compiler; it compiles at runtime. You do not need special pre-compilation steps for packaging. Distribute your code as a standard Python package, ensuring `numba` is listed as a dependency.

```toml
# pyproject.toml
[project]
name = "my_accelerated_lib"
version = "0.1.0"
dependencies = [
    "numba>=0.61.0", # Pin to latest stable release
    "numpy>=1.22.0",
]
```

## Testing Approaches

### Test Numba functions with and without the decorator.
This strategy ensures your core logic is correct even without Numba (e.g., for debugging or fallback), and provides a baseline for performance comparisons.

```python
import pytest
import numpy as np
from numba import njit

# Define the