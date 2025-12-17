---
description: This guide provides opinionated, actionable best practices for using LightGBM in production-grade Python ML pipelines, focusing on performance, reproducibility, and maintainability.
globs: **/*.py
---
# lightgbm Best Practices

LightGBM is our go-to for high-performance tabular modeling. These rules ensure our LightGBM implementations are fast, reliable, and maintainable.

## 1. Code Structure & Reproducibility

Always encapsulate model creation and ensure runs are repeatable.

### 1.1. Use Scikit-learn API & Model Builder Functions

Always use `LGBMClassifier` or `LGBMRegressor` for `sklearn` compatibility. Wrap model instantiation in a function for clean, version-controlled hyperparameter management.

❌ **BAD: Inline model creation with hardcoded parameters**
```python
# my_script.py
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=7)
model.fit(X_train, y_train)
```

✅ **GOOD: Function-based model creation with type hints and external config**
```python
# models/lgbm_model.py
import lightgbm as lgb
from typing import Dict, Any
import yaml # Or use a dataclass for params

def build_lgbm_classifier(params: Dict[str, Any]) -> lgb.LGBMClassifier:
    """Builds a LightGBM Classifier with specified hyperparameters."""
    return lgb.LGBMClassifier(**params)

# config/lgbm_params.yaml
# classifier_v1:
#   objective: binary
#   n_estimators: 500
#   learning_rate: 0.05
#   num_leaves: 31
#   max_depth: 6
#   random_state: 42
#   n_jobs: -1
#   colsample_bytree: 0.8
#   subsample: 0.8
#   reg_alpha: 0.1
#   reg_lambda: 0.1

# my_script.py
import yaml
from models.lgbm_model import build_lgbm_classifier

with open("config/lgbm_params.yaml", "r") as f:
    params = yaml.safe_load(f)['classifier_v1']

model = build_lgbm_classifier(params)
model.fit(X_train, y_train)
```

### 1.2. Set Reproducibility Flags

Ensure `random_state` (or `seed`) and `deterministic=True` are always set for consistent results.

❌ **BAD: Non-deterministic training**
```python
model = lgb.LGBMClassifier(n_estimators=100) # Results vary on each run
```

✅ **GOOD: Fully reproducible training**
```python
model = lgb.LGBMClassifier(n_estimators=100, random_state=42, deterministic=True)
```

## 2. Data Handling

Leverage LightGBM's native capabilities for efficiency.

### 2.1. Native Categorical Feature Handling

Feed raw categorical columns directly. LightGBM handles them efficiently, saving memory and speeding up training.

❌ **BAD: Manual one-hot or label encoding**
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
# Then concatenate with numerical features
model.fit(X_train_encoded, y_train)
```

✅ **GOOD: Use `categorical_feature` parameter**
```python
categorical_feature_names = ['feature_A', 'feature_B']
categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_feature_names]

model = lgb.LGBMClassifier(categorical_feature=categorical_feature_indices, **params)
model.fit(X_train, y_train)
```

### 2.2. Missing Values

LightGBM handles missing values natively. While it can split on them, consider imputing sensible defaults (e.g., mean, target encoding) for improved stability in some cases.

✅ **GOOD: Let LightGBM handle, or impute strategically**
```python
# Option 1: Let LightGBM handle (default behavior)
model = lgb.LGBMClassifier(**params)
model.fit(X_train_with_nans, y_train)

# Option 2: Impute for potential stability/performance gains
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_with_nans)
model.fit(X_train_imputed, y_train)
```

## 3. Hyperparameter Tuning & Training

Systematic tuning and robust training practices are non-negotiable.

### 3.1. Automated Hyperparameter Tuning

Always use automated tools like Optuna or FLAML for systematic tuning. Prioritize `learning_rate`, `num_leaves`, `max_depth`, and `min_data_in_leaf`.

✅ **GOOD: Optuna for efficient hyperparameter search**
```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best trial: {study.best_trial.value}")
print(f"Best parameters: {study.best_params}")
```

### 3.2. `num_leaves` vs `max_depth`

Always ensure `num_leaves < 2^max_depth` to prevent the leaf-wise tree from over-growing and overfitting.

❌ **BAD: `num_leaves` too large for `max_depth`**
```python
model = lgb.LGBMClassifier(num_leaves=127, max_depth=6) # 2^6 = 64. 127 is too large.
```

✅ **GOOD: Balanced `num_leaves` and `max_depth`**
```python
model = lgb.LGBMClassifier(num_leaves=31, max_depth=6) # 2^6 = 64. 31 is a good balance.
```

### 3.3. Early Stopping

Always use a validation set with early stopping to prevent overfitting and optimize training time.

❌ **BAD: Training for a fixed, potentially excessive number of estimators**
```python
model = lgb.LGBMClassifier(n_estimators=1000)
model.fit(X_train, y_train) # Might overfit or train longer than necessary
```

✅ **GOOD: Early stopping with a validation set**
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='auc',
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])
```

### 3.4. Callbacks for Monitoring

Use `log_evaluation` and custom callbacks for transparent and debuggable training.

✅ **GOOD: Log evaluation during training**
```python
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='auc',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=10)])
```

### 3.5. Advanced Sampling

Enable GOSS (`boosting_type='goss'`) and Exclusive Feature Bundling (EFB) for large datasets to reduce data size and speed up training without significant accuracy loss.

✅ **GOOD: Leverage GOSS for efficiency**
```python
model = lgb.LGBMClassifier(boosting_type='goss', **params)
model.fit(X_train, y_train)
# EFB is handled automatically by LightGBM if features are detected as exclusive.
```

## 4. Performance & Scalability

Optimize for speed and memory.

### 4.1. `num_threads` & GPU

Set `n_jobs=-1` (or `num_threads`) to utilize all available CPU cores. For very large datasets, use a GPU-enabled LightGBM build.

✅ **GOOD: Maximize CPU utilization**
```python
model = lgb.LGBMClassifier(n_jobs=-1, **params) # Uses all available CPU cores
# For GPU: ensure LightGBM is built with GPU support and set device='gpu'
# model = lgb.LGBMClassifier(device='gpu', **params)
```

### 4.2. Distributed Training

For scaling beyond a single node, select the appropriate parallel algorithm (`tree_learner`) based on data vs. feature size.

✅ **GOOD: Configure for distributed training (e.g., Dask)**
```python
# For Dask integration, use lightgbm.dask estimators
import lightgbm.dask as lgb_dask
from distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4, threads_per_worker=2) # At least 2 threads per worker
client = Client(cluster)

dask_model = lgb_dask.DaskLGBMClassifier(n_estimators=100, client=client, **params)
dask_model.fit(dask_X_train, dask_y_train,
               eval_set=[(dask_X_val, dask_y_val)],
               eval_metric='auc',
               callbacks=[lgb.early_stopping(50)])
client.close()
cluster.close()
```
For Spark, use SynapseML's `LightGBMClassifier`.

## 5. Common Pitfalls (❌ BAD vs ✅ GOOD)

Avoid these common mistakes.

### 5.1. Imbalanced Datasets

Always address class imbalance in classification tasks.

❌ **BAD: Ignoring imbalanced classes**
```python
model = lgb.LGBMClassifier(objective='binary', random_state=42)
model.fit(X_train_imbalanced, y_train_imbalanced) # Minority class performance will suffer
```

✅ **GOOD: Use `is_unbalance=True` or `class_weight='balanced'`**
```python
# Option 1: Let LightGBM adjust weights
model = lgb.LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)
model.fit(X_train_imbalanced, y_train_imbalanced)

# Option 2: Scikit-learn style balanced weights
model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=42)
model.fit(X_train_imbalanced, y_train_imbalanced)
```

## 6. Testing & Deployment Readiness

Ensure models are robust and ready for production.

### 6.1. Cross-Validation

Always use cross-validation for robust model evaluation and hyperparameter tuning.

✅ **GOOD: Scikit-learn `cross_val_score` or `GridSearchCV`**
```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = lgb.LGBMClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC scores: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### 6.2. Virtual Environments & Packaging

Always use virtual environments (e.g., `venv`, `conda`) and manage dependencies with `pip-tools` or `poetry`. Package your application for deployment using standard Python tools.

✅ **GOOD: Isolated environment and explicit dependencies**
```bash
# In your project root
python -m venv .venv
source .venv/bin/activate
pip install pip-tools # Or poetry
pip-compile requirements.in
pip install -r requirements.txt
```