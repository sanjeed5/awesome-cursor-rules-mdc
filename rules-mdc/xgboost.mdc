---
description: Definitive guidelines for writing robust, performant, and maintainable Python code using the xgboost library, focusing on modern best practices and common pitfalls.
globs: **/*.py
---
# xgboost Best Practices

This document outlines the definitive coding standards and best practices for developing with `xgboost` in Python. Adhering to these guidelines ensures your code is efficient, scalable, maintainable, and aligned with the project's official recommendations and modern ML workflows.

## 1. Code Organization and Structure

Organize your `xgboost` code for clarity, reusability, and maintainability. Separate concerns: data preparation, model definition, training, evaluation, and logging.

### 1.1. Modularity with `DMatrix` and `Booster`

For optimal performance and control, especially with large datasets or custom objectives, use `xgboost.DMatrix` for data handling and `xgboost.Booster` for the core model. This separates data management from model logic.

**❌ BAD: Repeatedly converting data**
```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X, y are loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Inefficient for repeated operations or large datasets
model = xgb.train(
    {'objective': 'binary:logistic'},
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=100
)
# Later, if you need to retrain or predict:
model.predict(xgb.DMatrix(X_test)) # Another conversion
```

**✅ GOOD: Pre-process data into `DMatrix` once**
```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume X, y are loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix once for efficiency
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist', # Modern, faster tree method
    'device': 'cuda' # Leverage GPU if available
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'eval')]
)
preds = model.predict(dtest)
```

### 1.2. Adhere to Python Style Guides

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style and use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all functions and classes. This is enforced by `xgboost`'s own contribution guidelines.

**❌ BAD: Inconsistent formatting, missing docstrings**
```python
def train_model(X,y):
    model=xgb.XGBClassifier()
    model.fit(X,y)
    return model
```

**✅ GOOD: PEP 8 compliant, NumPy-style docstrings**
```python
import xgboost as xgb
import numpy as np
from typing import Tuple

def train_xgboost_classifier(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier model.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    n_estimators : int, optional
        Number of boosting rounds, by default 100.

    Returns
    -------
    xgb.XGBClassifier
        The trained XGBoost classifier model.
    """
    model = xgb.XGBClassifier(n_estimators=n_estimators, tree_method='hist', random_state=42)
    model.fit(X_train, y_train)
    return model
```

## 2. Common Patterns and Anti-patterns

### 2.1. Scikit-learn API vs. Native API

Use the Scikit-learn API (`XGBClassifier`, `XGBRegressor`) for seamless integration into standard ML pipelines (e.g., `GridSearchCV`, `Pipeline`). Opt for the native `xgboost.train` API when you need fine-grained control, custom objectives/metrics, or to leverage `DMatrix` features like external memory.

**✅ GOOD: Scikit-learn API for standard use cases**
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    tree_method='hist', # Always prefer 'hist'
    enable_categorical=True, # Enable native categorical support
    random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
```

**✅ GOOD: Native API for advanced control (e.g., custom objective, `DMatrix` features)**
```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from typing import Tuple