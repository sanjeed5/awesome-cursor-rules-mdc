---
description: This guide provides opinionated, actionable best practices for developing with tinygrad, focusing on lazy execution, performance, and code structure.
globs: **/*
---
# tinygrad Best Practices

tinygrad prioritizes minimalism, performance, and a clear computation graph. Adhering to these guidelines ensures your code is efficient, readable, and aligns with the framework's core philosophy.

## 1. Embrace Lazy Evaluation

tinygrad builds a computation graph that only executes when explicitly requested. This is the single most important concept for correctness and performance.

### Explicitly Realize Tensors

Always call `.realize()` or `.numpy()` to force computation. Forgetting this is the most common pitfall.

❌ **BAD: Silent no-op**
```python
from tinygrad import Tensor
a = Tensor.rand(100)
b = Tensor.rand(100)
c = a + b # This does not compute anything
# c is just a graph node, not a computed value
```

✅ **GOOD: Force computation**
```python
from tinygrad import Tensor
a = Tensor.rand(100)
b = Tensor.rand(100)
c = a + b
c.realize() # Forces the computation of 'c'
print(c.numpy()) # Also forces computation and converts to numpy array
```

### Group Realizations for Fusion

When multiple tensors depend on a shared computation, realize them together to allow tinygrad's scheduler to fuse kernels and avoid redundant memory traffic.

❌ **BAD: Sequential realization, missed fusion opportunities**
```python
from tinygrad import Tensor
a = Tensor.rand(100)
b = Tensor.rand(100)
c = Tensor.rand(100)
out1 = a + b + c
out2 = a + b - c

out1.realize() # Computes a+b+c
out2.realize() # Recomputes a+b, then computes a+b-c
```

✅ **GOOD: Parallel realization for kernel fusion**
```python
from tinygrad import Tensor
a = Tensor.rand(100)
b = Tensor.rand(100)
c = Tensor.rand(100)
out1 = a + b + c
out2 = a + b - c

Tensor.realize(out1, out2) # Allows tinygrad to fuse (a+b) computation
```

## 2. Optimize Model Structure and Parameters

Keep models simple and leverage tinygrad's parameter management.

### Use `get_parameters` for Optimizers

Always use `tinygrad.nn.state.get_parameters` to collect all trainable parameters for your optimizer. Manually listing parameters is error-prone and not scalable.

❌ **BAD: Manual parameter listing**
```python
from tinygrad import Tensor
from tinygrad.nn.optim import SGD

class MyModel:
  def __init__(self):
    self.l1_w = Tensor.rand(10, 10, requires_grad=True)
    self.l1_b = Tensor.zeros(10, requires_grad=True)
  def __call__(self, x): return x @ self.l1_w + self.l1_b

net = MyModel()
opt = SGD([net.l1_w, net.l1_b], lr=1e-3) # Easy to miss parameters
```

✅ **GOOD: Automatic parameter collection**
```python
from tinygrad import Tensor
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

class MyModel:
  def __init__(self):
    self.l1_w = Tensor.rand(10, 10, requires_grad=True)
    self.l1_b = Tensor.zeros(10, requires_grad=True)
  def __call__(self, x): return x @ self.l1_w + self.l1_b

net = MyModel()
opt = SGD(get_parameters(net), lr=1e-3) # Robust and scalable
```

### Keep Imports Minimal

Adhere to tinygrad's minimalist ethos. Only import what you need.

❌ **BAD: Over-importing**
```python
import numpy as np
import time
from tinygrad.helpers import Timing, getenv # getenv is rarely needed
from tinygrad.nn import optim, state # Import specific classes
from tinygrad import Tensor, dtypes, Device # Device is often implicit
```

✅ **GOOD: Targeted imports**
```python
import numpy as np # Only if explicitly using numpy
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Timing
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
```

## 3. Performance Profiling

Measure, don't guess. Use tinygrad's built-in tools to identify bottlenecks.

### Use `Timing` for Benchmarking

Always wrap performance-critical sections with `Timing` to get accurate execution times, especially when comparing different approaches.

❌ **BAD: Inaccurate `time.time()` for lazy ops**
```python
import time
from tinygrad import Tensor
a = Tensor.rand(1000, 1000)
b = Tensor.rand(1000, 1000)
start = time.time()
c = a @ b # This is lazy, time.time() will be misleadingly fast
print(f"Time: {time.time() - start:.6f}s")
# Output: Time: 0.000123s (incorrect)
```

✅ **GOOD: Use `Timing` for accurate measurement**
```python
from tinygrad import Tensor
from tinygrad.helpers import Timing

a = Tensor.rand(1000, 1000)
b = Tensor.rand(1000, 1000)

with Timing("Matrix multiplication"):
  c = a @ b
  c.realize() # Ensure computation is part of the timing
# Output: Matrix multiplication: 0.012345s (example)
```

## 4. Handle Load/Store Operations

tinygrad does not natively support load/store ops to simplify backend porting. Implement them using `arange` masks.

### Implement Custom Load/Store with `arange`

If you need conditional writes or complex indexing, use `Tensor.arange` to create masks.

❌ **BAD: Attempting direct conditional assignment (not supported)**
```python
# This pattern is not directly supported in tinygrad
# x[mask] = value
```

✅ **GOOD: Using `where` with `arange` for conditional logic**
```python
from tinygrad import Tensor, dtypes

def sparse_update(x: Tensor, indices: Tensor, values: Tensor) -> Tensor:
  # Example: update specific elements in x with values based on indices
  # This is a simplified conceptual example, actual implementation
  # for complex sparse updates would be more involved.
  # For a real-world example, see sparse_categorical_crossentropy in tinygrad.
  mask = (Tensor.arange(x.numel(), dtype=dtypes.int32, requires_grad=False) == indices.flatten().unsqueeze(1)).any(axis=1).reshape(x.shape)
  return mask.where(values, x) # Conditionally replace elements
```

## 5. Reproducibility and Development Workflow

Maintain a disciplined workflow for robust and reproducible experiments.

### Commit Before Experiments

Always commit your code before running experiments. This ensures you can trace results back to a specific Git hash.

❌ **BAD: Running experiments on uncommitted changes**
```bash
# You run your training script
python train.py --epochs 10 --lr 0.001

# Later, you can't reproduce the exact code that generated the results.
```

✅ **GOOD: Version control for reproducibility**
```bash
git add .
git commit -m "Experiment: Initial MNIST training with Leaky ReLU"
# Get the commit hash: git rev-parse HEAD
# Store this hash with your experiment logs and results.
python train.py --epochs 10 --lr 0.001 --git_hash $(git rev-parse HEAD)
```