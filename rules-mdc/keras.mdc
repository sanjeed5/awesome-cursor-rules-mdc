---
description: Definitive guidelines for writing robust, maintainable, and high-performance Keras 3 models, emphasizing the functional API, modern training patterns, and multi-backend compatibility.
globs: **/*.py
---
# keras Best Practices

Keras 3 is engineered for developer experience, emphasizing debugging speed, elegance, and deployability. Adhere to these rules for production-ready, multi-backend Keras projects.

## 1. Code Organization and Structure

**Always separate concerns:** Model definition, data pipelines, and training logic must be distinct. This improves readability, testability, and reusability.

❌ BAD: Monolithic script
```python
# model, data, and training all in one file
import keras
import numpy as np

# Data generation
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, (100, 1))

# Model definition
inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

# Training
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x_train, y_train, epochs=10)
```

✅ GOOD: Modular components
```python
# my_model.py
import keras

def build_simple_model(input_shape=(10,)):
    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Dense(32, activation="relu", name="hidden_dense")(inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output_dense")(x)
    model = keras.Model(inputs, outputs, name="simple_classifier")
    return model

# train_script.py
import keras
import numpy as np
from my_model import build_simple_model

# 1. Data Loading (e.g., using tf.data or torch.utils.data for real projects)
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, (100, 1))

# 2. Model Instantiation
model = build_simple_model()

# 3. Training Configuration
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, monitor="val_loss"),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

# 4. Training Execution
model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)
```

## 2. Common Patterns and Anti-patterns

### 2.1 Model Building: Functional API is King

**Always use the Functional API** for most models. It's explicit, easy to inspect, and handles complex topologies (multi-input/output, shared layers) gracefully. Subclassing is reserved for highly custom, non-standard layers or models with dynamic graph execution.

❌ BAD: Subclassing for simple sequential models
```python
class MySequentialModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

✅ GOOD: Functional API for clarity and debuggability
```python
inputs = keras.Input(shape=(10,), name="input_features")
x = keras.layers.Dense(32, activation="relu", name="hidden_layer")(inputs)
outputs = keras.layers.Dense(1, activation="sigmoid", name="output_prediction")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="simple_functional_model")

# Visualize the model for debugging
keras.utils.plot_model(model, "simple_functional_model.png", show_shapes=True)
```

### 2.2 Training Loops: Leverage `model.fit` and Callbacks

**Always use `model.fit()`** with Keras's powerful built-in `Callbacks`. This simplifies training, provides standard logging, and ensures best practices like early stopping and model saving are applied consistently. Only implement a custom training loop (via the "Trainer pattern") when `model.fit()` is insufficient for complex research or custom gradient logic.

❌ BAD: Manual training loop without callbacks
```python
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy()

for epoch in range(10):
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape: # Assuming TF backend
            logits = model(x_batch)
            loss_value = loss_fn(y_batch, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
```

✅ GOOD: `model.fit()` with essential callbacks
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model_epoch_{epoch:02d}.keras", save_best_only=True, monitor="val_loss"),
    keras.callbacks.TensorBoard(log_dir="./logs")
]

# Use a proper Keras dataset or NumPy arrays
model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

## 3. Performance Considerations

### 3.1 Mixed Precision Training

**Always enable mixed precision** for modern GPUs/TPUs. It significantly speeds up training and reduces memory usage with minimal impact on accuracy.

```python
# At the very beginning of your script, before any model definition
keras.mixed_precision.set_global_policy("mixed_float16")

# Your model layers will automatically use float16 where appropriate
inputs = keras.Input(shape=(224, 224, 3))
x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
outputs = keras.layers.Dense(10, activation="softmax", dtype="float32")(x) # Output layer should be float32 for stability
model = keras.Model(inputs, outputs)
```

### 3.2 Multi-Backend Execution

**Always configure your backend explicitly** using the `KERAS_BACKEND` environment variable. This ensures consistency and allows easy switching between TensorFlow, JAX, or PyTorch.

❌ BAD: Relying on default or implicit backend
```python
import keras # Backend might be TF by default, but not guaranteed or explicit
```

✅ GOOD: Explicit backend configuration (before `import keras`)
```python
import os
os.environ["KERAS_BACKEND"] = "jax" # Or "tensorflow", "torch"
import keras
# Now Keras will use JAX as its backend
```

## 4. Common Pitfalls and Gotchas

### 4.1 Reproducibility

**Always set seeds** for NumPy, Python's random module, and the Keras backend for reproducible results, especially during development and experimentation.

```python
import os
import random
import numpy as np
import tensorflow as tf # Or jax, torch

# 1. Python's random module
random.seed(42)
# 2. NumPy
np.random.seed(42)
# 3. Keras backend (TensorFlow example)
tf.random.set_seed(42)
# For JAX: jax.random.PRNGKey(42) and pass to ops
# For PyTorch: torch.manual_seed(42)

# Ensure deterministic operations if using TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import keras # Import Keras AFTER setting seeds
```

### 4.2 Keras 2 vs. Keras 3

**Always use `import keras`** for Keras 3. Avoid `from tensorflow import keras` if you need to guarantee Keras 3 behavior, especially with older TensorFlow versions. Ensure `pip install --upgrade keras` is run.

❌ BAD: Potential Keras 2 import
```python
# If TF < 2.16, this might give you Keras 2
from tensorflow import keras
```

✅ GOOD: Explicit Keras 3 import
```python
import keras # This will always be Keras 3 if installed correctly
```

## 5. Type Hints

**Always use type hints** for function signatures and class attributes. This improves code clarity, enables static analysis, and reduces bugs.

❌ BAD: Untyped function
```python
def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(10)(inputs)
    return keras.Model(inputs, x)
```

✅ GOOD: Type-hinted function
```python
import keras
from typing import Tuple

def build_model(input_shape: Tuple[int, ...]) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(10, activation="relu")(inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="typed_model")
```

## 6. Virtual Environments

**Always use a virtual environment** (e.g., `venv`, `conda`) for every project. This isolates dependencies and prevents conflicts.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install Keras and backend
pip install --upgrade keras tensorflow # Or jax, torch
```

## 7. Packaging

**Package reusable components** (custom layers, models, data utilities) into a local package. This promotes modularity and easy sharing across projects.

```
my_project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── custom_layers.py  # Contains custom Keras layers
│   └── data/
│       ├── __init__.py
│       └── data_loader.py    # Contains data loading functions
├── tests/
│   └── test_models.py
└── train.py
```

## 8. Testing Approaches

**Always write unit tests** for custom layers, preprocessing functions, and other standalone components. **Implement integration tests** for full model training and inference pipelines.

```python
# tests/test_models.py
import unittest
import keras
from src.models.custom_layers import MyCustomLayer # Example

class TestCustomLayer(unittest.TestCase):
    def test_output_shape(self):
        layer = MyCustomLayer(units=10)
        inputs = keras.ops.zeros((1, 5)) # Use keras.ops for backend-agnostic ops
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (1, 10))

    def test_trainable_weights(self):
        layer = MyCustomLayer(units=10)
        _ = layer(keras.ops.zeros((1, 5)))
        self.assertGreater(len(layer.trainable_weights), 0)

# src/models/custom_layers.py
import keras

class MyCustomLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w) + self.b
```