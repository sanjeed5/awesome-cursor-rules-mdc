---
description: This guide provides opinionated, actionable best practices for using statsmodels in Python, focusing on code organization, common patterns, performance, and modern development workflows to ensure robust and reproducible statistical analysis.
globs: **/*.py
---
# `statsmodels` Best Practices

`statsmodels` is the definitive library for rigorous statistical modeling and inference in our AI/ML pipelines. This guide ensures consistent, high-quality, and reproducible statistical analysis across our projects.

## 1. Code Organization and Imports

Always use the recommended import aliases for clarity and consistency. Keep model specifications in dedicated, well-named scripts or notebooks.

### ✅ GOOD: Standardized Imports

```python
# For general models and datasets
import statsmodels.api as sm
import pandas as pd
import numpy as np

# For R-style formula interface
import statsmodels.formula.api as smf
```

### ❌ BAD: Inconsistent or Wildcard Imports

```python
from statsmodels.api import * # Pollutes namespace
import statsmodels as sm_lib # Non-standard alias
```

## 2. Data Handling and Model Specification

`statsmodels` excels with `pandas` DataFrames and R-style formulas. Leverage these for readability and robustness.

### 2.1. Use Pandas DataFrames

Always prepare your data as `pandas.DataFrame` objects. This ensures clear column names and simplifies formula specification.

### ✅ GOOD: Pandas DataFrame for Data

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.DataFrame({
    'y_response': np.random.rand(100),
    'x_feature1': np.random.rand(100) * 10,
    'x_feature2': np.random.randint(0, 2, 100),
    'group': np.random.choice(['A', 'B'], 100)
})

# Use formula API for convenience
model = smf.ols('y_response ~ x_feature1 + C(x_feature2) + group', data=data).fit()
print(model.summary())
```

### ❌ BAD: Raw NumPy Arrays (unless absolutely necessary)

```python
# Avoid this unless you need fine-grained control or non-standard designs
X = np.random.rand(100, 2)
y = np.random.rand(100)
model = sm.OLS(y, X).fit() # Column names are lost in summary
```

### 2.2. Always Add a Constant (Intercept)

For most regression models, an intercept term is crucial. The formula API handles this automatically; for array-based models, explicitly add it.

### ✅ GOOD: Explicitly Adding a Constant

```python
import statsmodels.api as sm
import numpy as np

X_raw = np.random.rand(100, 2)
X = sm.add_constant(X_raw) # Adds a column of ones for the intercept
y = np.random.rand(100)

model = sm.OLS(y, X).fit()
```

### ❌ BAD: Forgetting the Constant

```python
# This model will be fitted without an intercept, which is often incorrect
X = np.random.rand(100, 2)
y = np.random.rand(100)
model = sm.OLS(y, X).fit()
```

## 3. Reproducibility and Debugging

Ensure your analyses are reproducible and easy to debug by including version information.

### ✅ GOOD: Reporting Version Information

```python
import statsmodels.api as sm

print(f"Statsmodels Version: {sm.version.full_version}")
sm.show_versions() # Crucial for dependency debugging
```

### ❌ BAD: Omitting Version Details

```python
# When reporting issues or sharing code, this makes debugging harder
# print("Model results...")
```

## 4. Type Hints for Clarity

Use type hints to improve code readability, maintainability, and enable static analysis.

### ✅ GOOD: Using Type Hints

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

def fit_linear_model(data: pd.DataFrame, formula: str) -> RegressionResultsWrapper:
    """Fits an OLS model and returns the results."""
    model = sm.formula.ols(formula, data=data)
    results = model.fit()
    return results

# Usage
df = pd.DataFrame({'y': np.random.rand(10), 'x': np.random.rand(10)})
ols_results = fit_linear_model(df, 'y ~ x')
print(ols_results.summary())
```

### ❌ BAD: Untyped Code

```python
def fit_linear_model(data, formula): # Lacks clarity on expected types
    model = sm.formula.ols(formula, data=data)
    results = model.fit()
    return results
```

## 5. Virtual Environments and Packaging

Always isolate project dependencies using virtual environments and manage them explicitly.

### ✅ GOOD: Dedicated Virtual Environments and `pyproject.toml`

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

# Install dependencies
pip install pandas numpy statsmodels

# Manage dependencies with pyproject.toml (preferred) or requirements.txt
# pyproject.toml example:
# [project]
# name = "my-stats-project"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "statsmodels>=0.15"
# ]
```

### ❌ BAD: Global Package Installation

```bash
pip install pandas numpy statsmodels # Pollutes global Python environment
```

## 6. Testing Approaches

Thorough testing is non-negotiable. Ensure statistical models are robust and produce expected outcomes.

### ✅ GOOD: Unit and Integration Tests with Fixed Seeds

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def test_ols_model_coefficients():
    np.random.seed(42) # Fix seed for reproducibility
    data = pd.DataFrame({
        'y': 2 + 3 * np.random.rand(100) + np.random.normal(0, 0.5, 100),
        'x': np.random.rand(100)
    })
    model = smf.ols('y ~ x', data=data).fit()

    # Assert expected coefficients within a tolerance
    assert np.isclose(model.params['Intercept'], 2.0, atol=0.5)
    assert np.isclose(model.params['x'], 3.0, atol=0.5)

# Run tests as part of CI (e.g., pytest)
# Ensure CI pipeline includes ruff, isort, and other linters.
```

### ❌ BAD: Untested Models or Unreproducible Tests

```python
# No tests for model logic or coefficient validation
# Tests that rely on random data without a fixed seed will be flaky
```

## 7. Docstrings and Comments

Follow the NumPy/SciPy docstring standard and use concise, clear comments for complex logic.

### ✅ GOOD: NumPy-style Docstrings

```python
def analyze_sales_data(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Analyzes sales data using Ordinary Least Squares (OLS).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'sales', 'ad_spend', and 'promotions' columns.

    Returns
    -------
    sm.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model results.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """
    required_cols = ['sales', 'ad_spend', 'promotions']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    model = smf.ols('sales ~ ad_spend + promotions', data=df)
    results = model.fit()
    return results
```

### ❌ BAD: Missing or Poor Docstrings

```python
def analyze_sales_data(df): # What does it do? What are inputs/outputs?
    model = smf.ols('sales ~ ad_spend + promotions', data=df)
    results = model.fit()
    return results
```

## 8. Git Workflow

Adhere to our standard Git practices for maintainability and clear history.

### ✅ GOOD: Focused Branches and Clear Commits

```bash
# Work on a single feature per branch
git checkout -b feature/add-robust-regression

# Write concise, descriptive commit messages
git commit -m "feat: Implement Huber-M-estimator for RLM" -m "Adds robust linear model with Huber loss function to handle outliers in financial data analysis."
```

### ❌ BAD: "Kitchen Sink" Branches and Vague Commits

```bash
git checkout -b dev # Generic branch
git commit -m "Update code" # Uninformative
```