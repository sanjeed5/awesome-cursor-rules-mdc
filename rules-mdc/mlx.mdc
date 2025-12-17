---
description: This guide provides opinionated, actionable best practices for developing high-performance machine learning applications with MLX on Apple Silicon, focusing on Python and Swift.
globs: **/*
---
# mlx Best Practices

MLX is Apple's high-performance array framework for numerical computing and AI on Apple Silicon. Adhering to these guidelines ensures your MLX code is efficient, maintainable, and integrates seamlessly with Apple's ML ecosystem.

## 1. Code Organization and Structure

Organize your MLX projects into logical, reusable modules. Mirror the structure of official Apple examples (e.g., `corenet` MLX examples) for clarity and maintainability.

### 1.1 Modularize Components
Separate concerns into dedicated files: data loading, model definition, training loops, and utility functions.

❌ **BAD: Monolithic Script**
```python
# train.py
import mlx.core as mx
import mlx.nn as nn
# ... data loading, model definition, training loop all in one file ...
```

✅ **GOOD: Modular Structure**
```
my_project/
├── data/
│   └── mnist_loader.py
├── models/
│   └── simple_cnn.py
├── config/
│   └── train_config.yaml
├── utils/
│   └── metrics.py
└── train.py
```
```python
# models/simple_cnn.py
import mlx.core as mx
import mlx.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(32 * 14 * 14, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1) # Flatten
        return self.linear(x)

# train.py
from data.mnist_loader import load_mnist_data
from models.simple_cnn import SimpleCNN
# ... training logic ...
```

### 1.2 Externalize Configuration
Store training parameters, model hyperparameters, and dataset paths in version-controlled YAML or JSON files.

❌ **BAD: Hardcoded Parameters**
```python
# train.py
learning_rate = 0.001
batch_size = 32
num_epochs = 10
model = SimpleCNN(num_classes=10)
```

✅ **GOOD: Config-Driven**
```yaml
# config/train_config.yaml
training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 10
model:
  name: SimpleCNN
  num_classes: 10
data:
  path: "./data/mnist"
```
```python
# train.py
import yaml
from models.simple_cnn import SimpleCNN

with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = SimpleCNN(num_classes=config["model"]["num_classes"])
# ... use config["training"]["learning_rate"] etc.
```

## 2. Common Patterns and Anti-patterns

Leverage MLX's core features like lazy evaluation and unified memory for optimal performance.

### 2.1 Embrace Lazy Evaluation
MLX computations are lazy; they build a computation graph but execute only when results are explicitly needed. Use `mlx.eval()` or `mlx.compile()` to force execution.

❌ **BAD: Relying on Implicit Evaluation for Performance Critical Sections**
```python
# This loop might build a huge graph or trigger many small evaluations
# if intermediate results are printed or accessed.
for i in range(num_steps):
    loss, grads = value_and_grad(model, loss_fn)(model, X_batch, y_batch)
    # ... optimizer update ...
    # No explicit eval, performance can be unpredictable.
```

✅ **GOOD: Explicit Evaluation for Control**
Use `mlx.eval()` on the final outputs of a computation step to trigger execution efficiently. For static graphs, `mlx.compile()` offers further optimization.

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import value_and_grad

# Define model and loss_fn (as in 1.1)

def train_step(model, X, y, optimizer):
    loss, grads = value_and_grad(model, loss_fn)(model, X, y)
    optimizer.update(model, grads)
    return loss

# Compile the training step for static graphs
compiled_train_step = mx.compile(train_step)

# In your training loop:
optimizer = Adam(learning_rate=0.001)
for i in range(num_steps):
    X_batch, y_batch = get_batch()
    # Use the compiled step
    loss = compiled_train_step(model, X_batch, y_batch, optimizer)
    # Explicitly evaluate the loss to trigger computation and get its value
    mx.eval(model, optimizer, loss) # Evaluate model state, optimizer state, and loss
    print(f"Step {i}, Loss: {loss.item()}")
```

### 2.2 Leverage Unified Memory
MLX arrays live in shared memory, eliminating explicit data transfers between CPU and GPU. Specify the device for operations when necessary, but default behavior is often sufficient.

❌ **BAD: Unnecessary Device Management (like other frameworks)**
```python
# No need to explicitly move data to 'gpu' or 'cpu'
data = mx.array([1, 2, 3]).to("gpu") # This is not how MLX works
```

✅ **GOOD: Device-Agnostic Operations (or explicit per-op)**
```python
import mlx.core as mx

a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])

# Operations run on the default device (usually GPU if available)
c = a + b
mx.eval(c)

# You can explicitly specify device per operation if needed for specific tasks
d = mx.add(a, b, stream=mx.cpu) # Force CPU execution for this specific op
mx.eval(d)
```

### 2.3 Use Type Hints and Pure Functions
Follow Python best practices: annotate functions and variables with type hints for clarity. Design functions to be pure where possible for easier testing and composability.

## 3. Performance Considerations

Optimize for Apple Silicon by understanding MLX's execution model.

### 3.1 Batching and Data Loading
Efficient data loading and proper batching are critical for GPU utilization.

❌ **BAD: Iterating over single samples or inefficient data loading**
```python
# Inefficient for GPU
for sample_x, sample_y in dataset:
    # ... process single sample ...
```

✅ **GOOD: Batch Data and Preload**
```python
# Use a DataLoader or similar utility to yield batches of MLX arrays
def data_loader(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield mx.array(data[i:i+batch_size])

# In training loop:
for X_batch, y_batch in data_loader(train_data, batch_size):
    # ... process batch ...
```

### 3.2 Quantization for Deployment
For deploying models on-device, consider quantizing weights (e.g., 4-bit) to reduce memory footprint and improve inference speed.

```python
import mlx.core as mx
import mlx.nn as nn

# Assuming 'model' is a trained MLX model
# Quantize the model to 4-bit
quantized_model = model.quantize(group_size=32, bits=4)

# Save the quantized model
mx.save("quantized_model.safetensors", quantized_model.parameters())
```

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that can lead to unexpected behavior or poor performance.

### 4.1 Forgetting `mlx.eval()`
The most common pitfall. If you don't explicitly evaluate, your graph might not execute, or only partially.

❌ **BAD: No `eval()` on results**
```python
# This will print the MLX array object, not its computed value
result = mx.add(mx.array([1]), mx.array([2]))
print(result) # Output: array(value=..., shape=(1,), dtype=int32)
```

✅ **GOOD: Always `eval()` when you need the value**
```python
result = mx.add(mx.array([1]), mx.array([2]))
mx.eval(result)
print(result) # Output: array(3, shape=(1,), dtype=int32)
print(result.item()) # Output: 3
```

### 4.2 Over-evaluating or Under-evaluating
Finding the right balance for `mlx.eval()` is key. Too many small `eval` calls can break graph optimization. Too few can lead to memory issues or delayed execution.

**Guideline:** Evaluate at logical synchronization points, typically at the end of a training step or inference call. Use `mx.eval(model, optimizer, loss)` to evaluate all relevant states.

## 5. Testing Approaches

Ensure the reliability and correctness of your MLX models.

### 5.1 Unit Tests for Model Components
Test individual layers, custom modules, and utility functions.

```python
# tests/test_simple_cnn.py
import mlx.core as mx
import mlx.nn as nn
from models.simple_cnn import SimpleCNN

def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    # Initialize model parameters by running a dummy input
    _ = model(mx.zeros((1, 1, 28, 28)))
    mx.eval(model) # Ensure parameters are initialized

    input_tensor = mx.random.normal((2, 1, 28, 28))
    output = model(input_tensor)
    assert output.shape == (2, 10)
    print("SimpleCNN output shape test passed.")

if __name__ == "__main__":
    test_simple_cnn_output_shape()
```

### 5.2 Integration Tests for Inference
Verify that your trained models perform as expected on new data, especially when integrating with Core ML or other on-device frameworks.

### 5.3 Data Validation and Reproducibility
Implement sanity checks for all input data. Version control your data, models, and training configurations to ensure reproducible results.

```python
# utils/data_validation.py
import mlx.core as mx

def validate_image_data(images: mx.array, expected_shape: tuple, expected_dtype: mx.Dtype):
    assert images.ndim == len(expected_shape), f"Expected {len(expected_shape)} dims, got {images.ndim}"
    assert images.shape == expected_shape, f"Expected shape {expected_shape}, got {images.shape}"
    assert images.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {images.dtype}"
    assert images.min() >= 0 and images.max() <= 1, "Image pixel values out of range [0, 1]"
    print("Image data validation passed.")

# In your data loader or training script:
# ... load raw data ...
# raw_images = mx.array(...)
# validate_image_data(raw_images, (100, 1, 28, 28), mx.float32)
```

## 6. Swift-Specific Guidelines

When working with MLX Swift, adhere to Apple's official Swift style guides and API design guidelines.

### 6.1 Follow Swift API Style
Use idiomatic Swift patterns, consistent naming, and proper error handling. Refer to the `mlx-swift` examples (e.g., MNISTTrainer) as templates.

### 6.2 Manage Object Lifetime
Be mindful of object lifetimes, especially when bridging C++ MLX objects to Swift, as aggressive inlining and optimizations can affect behavior. The `mlx-swift` package includes recommendations for this.