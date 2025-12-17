---
description: This guide defines the definitive best practices for writing TensorFlow code, focusing on `tf.keras` for models, `tf.data` for input pipelines, and `@tf.function` for performance, ensuring reproducible, scalable, and production-ready ML systems.
globs: **/*.py
---
# tensorflow Best Practices

This document outlines our team's definitive guidelines for writing TensorFlow code. Adhering to these practices ensures maintainable, performant, and reproducible machine learning systems.

## 1. Code Organization and Structure

Always structure your TensorFlow projects for clarity and modularity. Separate concerns into distinct files or functions.

### 1.1. Model Definition: Use `tf.keras`

Always define models using the `tf.keras` API. It's the high-level, declarative standard for TensorFlow. Prefer the Functional API for complex models and `Sequential` for simple stacks.

❌ BAD: Raw `tf.Variable` and custom loops for model architecture.
```python
import tensorflow as tf

class BadCustomModel:
    def __init__(self):
        self.w1 = tf.Variable(tf.random.normal([784, 128]))
        self.b1 = tf.Variable(tf.zeros([128]))
        # ... more manual variables
```

✅ GOOD: `tf.keras.Model` or `tf.keras.Sequential`.
```python
import tensorflow as tf

def build_model(input_shape: tuple[int, ...], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Usage:
model = build_model((28, 28), 10)
```

### 1.2. Data Pipelines: Use `tf.data`

Always use the `tf.data` API for building robust and efficient input pipelines. This is critical for scaling to large datasets and optimizing I/O.

❌ BAD: Loading all data into memory or using `numpy` arrays for large datasets.
```python
import numpy as np
# ... load huge_data into numpy array
# x_train, y_train = np.load('huge_data.npy')
# model.fit(x_train, y_train, batch_size=32) # Inefficient for large data
```

✅ GOOD: Stream data with `tf.data.Dataset`.
```python
import tensorflow as tf
import tensorflow_datasets as tfds

def create_dataset(split: str, batch_size: int) -> tf.data.Dataset:
    ds = tfds.load('mnist', split=split, as_supervised=True)
    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    ds = ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Usage:
train_ds = create_dataset('train', 128)
test_ds = create_dataset('test', 128)
```

## 2. Common Patterns and Anti-patterns

### 2.1. Performance Critical Code: Use `@tf.function`

Always wrap pure TensorFlow operations and custom training loops in `@tf.function` to enable graph compilation and performance optimizations (e.g., XLA). Understand its graph-building semantics.

❌ BAD: Relying solely on eager execution for production training/inference.
```python
# Slower due to Python overhead per operation
for epoch in range(epochs):
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

✅ GOOD: Compile with `@tf.function`.
```python
import tensorflow as tf

@tf.function
def train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
               loss_fn: tf.keras.losses.Loss, x_batch: tf.Tensor, y_batch: tf.Tensor):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Usage in training loop:
# for epoch in range(epochs):
#     for x_batch, y_batch in train_ds:
#         train_step(model, optimizer, loss_fn, x_batch, y_batch)
```

### 2.2. Reproducibility: Pin Random Seeds

Always set global and operational random seeds to ensure reproducibility across runs. This is non-negotiable for reliable experimentation.

❌ BAD: Not setting any seeds, leading to non-deterministic results.
```python
# Results will vary between runs
model = tf.keras.Sequential([...])
```

✅ GOOD: Set all relevant seeds.
```python
import tensorflow as tf
import numpy as np
import random

def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # For CUDA operations, if applicable
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_all_seeds(42)
model = tf.keras.Sequential([...])
```

## 3. Performance Considerations

### 3.1. Data Preprocessing: Optimize `tf.data`

Leverage `tf.data` features like `cache()`, `prefetch()`, and `num_parallel_calls` for `map()` to prevent I/O bottlenecks.

✅ GOOD: Efficient `tf.data` pipeline.
```python
import tensorflow as tf

def optimized_dataset(filepaths: list[str], batch_size: int) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(filepaths) # Or from_tensor_slices, etc.
    ds = ds.map(parse_example_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache() # Cache data after initial processing
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE) # Preload next batch
    return ds
```

## 4. Common Pitfalls and Gotchas

### 4.1. `tf.function` Side Effects

Avoid Python side effects (e.g., printing, appending to lists) within `@tf.function` decorated functions if you expect them to execute every time. Graph compilation can hoist or ignore them.

❌ BAD: Expecting Python `print` to execute on every graph call.
```python
@tf.function
def my_func(x):
    tf.print("Tracing or executing once!") # Only prints on first call/trace
    return x + 1
```

✅ GOOD: Use `tf.print` for debugging inside graphs.
```python
@tf.function
def my_func(x):
    tf.print("This prints every time the graph executes:", x)
    return x + 1
```

## 5. Type Hints

Always use Python type hints for all function signatures and variables. This improves code readability, maintainability, and enables static analysis. Use `tf.Tensor` for TensorFlow tensors.

❌ BAD: Untyped functions.
```python
def process_data(data, label):
    return data / 255.0, label
```

✅ GOOD: Fully type-hinted functions.
```python
import tensorflow as tf

def process_data(data: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scales image data and returns it with the label."""
    return tf.cast(data, tf.float32) / 255.0, label
```

## 6. Virtual Environments

Always use a virtual environment (`venv` or `conda`) for every project. This isolates dependencies and prevents conflicts.

✅ GOOD: Project-specific virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow mlflow
```

## 7. Packaging and Experiment Tracking

### 7.1. Project Packaging

Structure your project as a Python package (`src/my_project/`). Use `pyproject.toml` for dependency management and `setuptools` for packaging.

✅ GOOD: Standard Python package structure.
```
my_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── models.py
│       ├── data_pipeline.py
│       └── train.py
├── tests/
│   ├── test_models.py
│   └── test_data.py
├── pyproject.toml
└── README.md
```

### 7.2. Experiment Tracking with MLflow

Always integrate MLflow for experiment tracking. Log parameters, metrics, and model artifacts. This is crucial for reproducibility and collaboration.

✅ GOOD: MLflow integration.
```python
import mlflow
import tensorflow as tf

# Configure MLflow tracking URI (e.g., local, remote server)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist_classification")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("epochs", 5)
    mlflow.log_param("optimizer", "adam")

    model = tf.keras.models.Sequential([...])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Use MLflow Keras callback for automatic metric logging
    mlflow.tensorflow.autolog() # Enables auto-logging of metrics, params, model

    model.fit(train_ds, epochs=5, validation_data=test_ds)

    # Evaluate and log final metrics if not using autolog
    # results = model.evaluate(test_ds)
    # mlflow.log_metric("test_loss", results[0])
    # mlflow.log_metric("test_accuracy", results[1])

    # Save the model
    # mlflow.tensorflow.log_model(model, "mnist_model")
```

## 8. Testing Approaches

Always implement comprehensive testing for your TensorFlow code.

### 8.1. Unit Tests

Write unit tests for custom layers, preprocessing functions, and utility modules. Use `tf.test.TestCase` for tests involving TensorFlow operations.

✅ GOOD: Unit test for a custom layer.
```python
import tensorflow as tf
from tensorflow.python.platform import test

class MyCustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs * 2

class MyCustomLayerTest(test.TestCase):
    def test_output_shape(self):
        layer = MyCustomLayer()
        input_tensor = tf.zeros((10, 5))
        output_tensor = layer(input_tensor)
        self.assertEqual(output_tensor.shape, (10, 5))

    def test_output_values(self):
        layer = MyCustomLayer()
        input_tensor = tf.constant([1.0, 2.0, 3.0])
        output_tensor = layer(input_tensor)
        self.assertAllEqual(output_tensor, tf.constant([2.0, 4.0, 6.0]))

if __name__ == '__main__':
    test.main()
```

### 8.2. Data Sanity Checks

Implement sanity checks for all input data sources to catch issues early.

✅ GOOD: Basic data validation.
```python
import tensorflow as tf

def validate_image_dataset(dataset: tf.data.Dataset, expected_shape: tuple, expected_dtype: tf.DType):
    for images, labels in dataset.take(1):
        tf.Assert(tf.equal(tf.shape(images)[1:], expected_shape), ["Image shape mismatch"])
        tf.Assert(tf.equal(images.dtype, expected_dtype), ["Image dtype mismatch"])
        tf.Assert(tf.reduce_all(images >= 0.0) and tf.reduce_all(images <= 1.0), ["Image pixel values out of range"])
        tf.print("Data sanity check passed for one batch.")

# Usage:
# validate_image_dataset(train_ds, (28, 28, 1), tf.float32)
```

### 8.3. Integration Tests

Test the full data pipeline and model training loop end-to-end with small, representative datasets.