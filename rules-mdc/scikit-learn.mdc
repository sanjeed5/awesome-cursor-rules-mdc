---
description: Definitive guidelines for writing robust, maintainable, and performant scikit-learn code, emphasizing consistent preprocessing, API adherence, and data leakage prevention.
globs: **/*.py
---
# scikit-learn Best Practices

This guide outlines our team's definitive best practices for using and extending scikit-learn. Adhering to these rules ensures consistent, reproducible, and production-ready machine learning code.

## 1. Code Organization and Structure

### 1.1. Always Use Pipelines for Preprocessing and Models

**Pipelines are mandatory.** They prevent data leakage, ensure consistent transformations across training and inference, and simplify hyperparameter tuning.

❌ BAD: Inconsistent manual transformations

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X, y = np.random.rand(100, 5), np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression().fit(X_train_scaled, y_train)

# Forgetting to scale X_test leads to incorrect predictions
y_pred = model.predict(X_test)
print(f"MSE (BAD): {mean_squared_error(y_test, y_pred):.2f}")
```

✅ GOOD: Encapsulate all steps in a `Pipeline`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X, y = np.random.rand(100, 5), np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# All steps are chained and applied consistently
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MSE (GOOD): {mean_squared_error(y_test, y_pred):.2f}")
```

### 1.2. Custom Estimators Must Adhere to the scikit-learn API

When creating custom transformers or models, strictly follow the scikit-learn estimator API for seamless integration with pipelines and model selection tools.

-   Inherit from `BaseEstimator` and relevant mixins (`TransformerMixin`, `ClassifierMixin`, `RegressorMixin`).
-   `__init__` stores *only* hyperparameters as keyword arguments with defaults. No logic or data processing here.
-   `fit(self, X, y=None)` learns parameters from data. Store learned attributes with a trailing underscore (e.g., `self.mean_`). Return `self`.
-   `transform(self, X)` (for transformers) or `predict(self, X)`/`predict_proba(self, X)` (for predictors) applies the learned transformation/prediction.
-   Implement `get_params` and `set_params` (usually inherited from `BaseEstimator`).

❌ BAD: Non-compliant custom estimator

```python
class BadCustomScaler:
    def __init__(self, scale_factor):
        # Logic in __init__
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
        self.scale_factor = scale_factor
        self.mean_ = None # Should be learned in fit

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Fit not called yet.")
        return (X - self.mean_) * self.scale_factor

# This won't work with GridSearchCV or Pipelines reliably
```

✅ GOOD: API-compliant custom estimator

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class GoodCustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale_factor: float = 1.0):
        # Only store hyperparameters, no logic
        self.scale_factor = scale_factor

    def fit(self, X: np.ndarray, y=None):
        # Input validation (optional, but good practice)
        X = self._validate_data(X)
        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
        self.mean_ = X.mean(axis=0) # Learned attribute with underscore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(X) # Ensure consistency
        if not hasattr(self, "mean_"):
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return (X - self.mean_) * self.scale_factor

# This estimator integrates perfectly with scikit-learn utilities.
```

## 2. Common Patterns and Anti-patterns

### 2.1. Prevent Data Leakage: Split First, Fit on Train Only

Always split your data into training and test sets *before* any preprocessing. `fit` and `fit_transform` methods must *only* be called on the training data. `transform` can then be called on both training and test data.

❌ BAD: Data leakage by fitting on all data

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.random.rand(100, 5), np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
# Fitting on all data (X) leaks information from the test set
X_scaled = scaler.fit_transform(X)
X_train_scaled = X_scaled[:len(X_train)]
X_test_scaled = X_scaled[len(X_train):]
```

✅ GOOD: Correct data splitting and transformation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.random.rand(100, 5), np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform both train and test using parameters learned from train
X_test_scaled = scaler.transform(X_test)
```

### 2.2. Reproducibility: Always Set `random_state`

For any estimator or utility that involves randomness (e.g., `train_test_split`, `RandomForestClassifier`, `KMeans`), explicitly set the `random_state` parameter for reproducible results.

❌ BAD: Non-reproducible results

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Results will change on each run
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
```

✅ GOOD: Reproducible results

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Use a fixed integer for random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
```

## 3. Performance Considerations

### 3.1. Leverage Cython for Performance-Critical Components

When extending scikit-learn with custom, performance-critical algorithms, use Cython and follow scikit-learn's internal conventions.

-   Disable bounds checking and wraparound for production code.
-   Use `sklearn.utils._typedefs` for explicit type declarations.
-   Use `sklearn.utils._openmp_helpers` for OpenMP routines.
-   Prefer memoryviews over `cnp.ndarray` when possible.

```python
# Example of Cython directives and imports (in a .pyx file)
# distutils: language=c
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.parallel import prange
from sklearn.utils._typedefs cimport float64
from sklearn.utils._openmp_helpers cimport _get_num_threads

cdef void my_fast_function(float64[:] data, int n_threads) nogil:
    cdef int i
    with nogil:
        for i in prange(data.shape[0], num_threads=n_threads, schedule='static'):
            data[i] *= 2.0
```

## 4. Common Pitfalls and Gotchas

### 4.1. Avoid Overfitting on Validation Data During Hyperparameter Tuning

Use `GridSearchCV` or `RandomizedSearchCV` *with a pipeline* to ensure that cross-validation folds are correctly handled and preprocessing steps are refitted for each fold, preventing data leakage.

❌ BAD: Tuning hyperparameters outside a pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scaling outside of GridSearchCV means the scaler sees all training data
# before cross-validation, leading to optimistic scores.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

param_grid = {'C': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)
# Scores here are optimistically biased
```

✅ GOOD: Tuning hyperparameters with a pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define the pipeline including preprocessing and the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42))
])

param_grid = {'logreg__C': [0.1, 1.0, 10.0]} # Note the 'estimator__param' syntax
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train) # Fit the entire pipeline on X_train
# Scores are realistic as preprocessing is done correctly within each fold
```

## 5. Type Hints

### 5.1. Use Type Hints Extensively

All function signatures, class attributes, and complex variable assignments must include type hints. This improves code readability, enables static analysis, and reduces bugs.

❌ BAD: Untyped function

```python
def calculate_metric(y_true, y_pred):
    # What are y_true and y_pred? Lists, arrays, Series?
    return (y_true == y_pred).mean()
```

✅ GOOD: Clearly typed function

```python
import numpy as np
from typing import Union

def calculate_metric(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Calculates accuracy for binary classification."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return (y_true_arr == y_pred_arr).mean()
```

## 6. Virtual Environments

### 6.1. Isolate Project Dependencies with Virtual Environments

Always use a dedicated virtual environment (`venv`, `conda`, `poetry`, etc.) for each project. This prevents dependency conflicts and ensures consistent environments.

❌ BAD: Global `pip install`

```bash
pip install scikit-learn pandas numpy # Pollutes global environment
```

✅ GOOD: Project-specific virtual environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Using conda
conda create -n my_ml_env python=3.9 scikit-learn pandas numpy
conda activate my_ml_env
```

## 7. Packaging

### 7.1. Use `scikit-learn-contrib` Template for Shareable Estimators

If you develop a custom estimator intended for public use or sharing across projects, leverage the `scikit-learn-contrib template`. It provides a robust structure, testing, and CI/CD setup, ensuring your estimator is fully compliant and easily consumable.

❌ BAD: Ad-hoc custom estimator distribution

```python
# Just a .py file with a custom estimator, no proper packaging
# Difficult for others to install, test, or integrate
```

✅ GOOD: Structured project using `scikit-learn-contrib`

```bash
# Follow the template to create a new project:
# https://github.com/scikit-learn-contrib/project-template
# This provides setup.py, tests, documentation structure, etc.
```

## 8. Testing Approaches

### 8.1. Implement Comprehensive Unit and Integration Tests

For custom estimators, write unit tests to verify individual methods and use `sklearn.utils.estimator_checks.check_estimator` to ensure API compliance. For pipelines, write integration tests that cover the entire workflow.

❌ BAD: No tests or only manual verification

```python
# No tests for MyCustomEstimator, relying on manual runs
# to confirm it works.
```

✅ GOOD: Automated testing with `check_estimator`

```python
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset: float = 0.0):
        self.offset = offset
    def fit(self, X, y=None):
        self._validate_data(X)
        return self
    def transform(self, X):
        self._validate_data(X, reset=False)
        return X + self.offset

# Test API compliance
@pytest.mark.parametrize(
    "estimator", [MyCustomTransformer()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)

# Add specific unit tests for logic
def test_my_custom_transformer_offset():
    X = np.array([[1, 2], [3, 4]])
    transformer = MyCustomTransformer(offset=10)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert np.array_equal(transformed_X, np.array([[11, 12], [13, 14]]))
```