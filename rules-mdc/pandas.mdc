---
description: This guide outlines definitive best practices for writing high-performance, maintainable, and robust pandas code, focusing on modern patterns and avoiding common pitfalls.
globs: **/*.py
---
# pandas Best Practices

Pandas is the backbone of data analysis in Python. Adhere to these guidelines to write efficient, readable, and scalable code that integrates seamlessly into modern AI/ML pipelines.

## 1. Code Organization & Structure

### 1.1 Standard Imports
Always import pandas with its conventional alias for consistency.

❌ BAD
```python
import pandas
from pandas import DataFrame
```

✅ GOOD
```python
import pandas as pd
import numpy as np # For numerical operations
```

### 1.2 Method Chaining for Readability
Chain operations to create clear, sequential data transformations. Use `.pipe()` for custom functions or when intermediate steps improve clarity.

❌ BAD
```python
df_filtered = df[df['value'] > 10]
df_grouped = df_filtered.groupby('category')
df_result = df_grouped['metric'].mean().reset_index()
```

✅ GOOD
```python
df_result = (
    df[df['value'] > 10]
    .groupby('category')['metric']
    .mean()
    .reset_index()
    .rename(columns={'metric': 'avg_metric'}) # Add a descriptive rename
)
```

## 2. Common Patterns & Anti-patterns

### 2.1 Avoid Python Loops – Embrace Vectorization
Iterating over DataFrames row-by-row (`.iterrows()`, `.apply(axis=1)`) is a major performance bottleneck. Pandas operations are optimized C-level functions.

❌ BAD (Slow for large DataFrames)
```python
# Calculating a new column based on existing ones
for index, row in df.iterrows():
    df.loc[index, 'new_col'] = row['col_a'] + row['col_b']

# Complex conditional logic with .apply(axis=1)
def calculate_status(row):
    if row['score'] > 90 and row['grade'] == 'A':
        return 'Excellent'
    return 'Good'
df['status'] = df.apply(calculate_status, axis=1)
```

✅ GOOD (Vectorized and performant)
```python
# Calculating a new column
df['new_col'] = df['col_a'] + df['col_b']

# Complex conditional logic using np.select or boolean indexing
conditions = [
    (df['score'] > 90) & (df['grade'] == 'A'),
    (df['score'] > 80) & (df['grade'] == 'B')
]
choices = ['Excellent', 'Very Good']
df['status'] = np.select(conditions, choices, default='Good')
```

### 2.2 Use `.loc` and `.iloc` for Explicit Access
Always use `.loc` (label-based) or `.iloc` (integer-based) for setting values to prevent `SettingWithCopyWarning` and ensure predictable behavior.

❌ BAD (Chained indexing can return a view, leading to `SettingWithCopyWarning`)
```python
df[df['city'] == 'London']['population'] = 8_000_000
```

✅ GOOD (Explicitly modifies the original DataFrame)
```python
df.loc[df['city'] == 'London', 'population'] = 8_000_000
```

### 2.3 Leverage `df.eval()` and `df.query()`
For complex filtering or column creation, `eval()` and `query()` can be significantly faster than boolean indexing, as they use the underlying C engine.

❌ BAD
```python
df_filtered = df[(df['col_a'] > 10) & (df['col_b'] < 20) | (df['col_c'] == 'X')]
df['new_col'] = df['col_a'] * df['col_b'] + df['col_c'] / df['col_d']
```

✅ GOOD
```python
df_filtered = df.query("(col_a > 10 and col_b < 20) or (col_c == 'X')")
df['new_col'] = df.eval("col_a * col_b + col_c / col_d")
```

## 3. Performance Considerations

### 3.1 Optimize Data Loading with `read_csv`
Specify `dtype` and `usecols` upfront to reduce memory footprint and speed up loading, especially for large files. Let `read_csv` infer optimal defaults (`low_memory=False` is often better).

❌ BAD
```python
df = pd.read_csv("large_data.csv") # Pandas infers all dtypes, loads all columns
```

✅ GOOD
```python
# Define dtypes and select only necessary columns
dtypes = {
    'store_id': 'Int32', # Use nullable integer
    'product_id': 'Int32',
    'category': 'category', # For repeated strings
    'price': 'Float32',
    'timestamp': 'datetime64[ns]'
}
cols_to_use = ['store_id', 'product_id', 'category', 'price', 'timestamp']

df = pd.read_csv(
    "large_data.csv",
    dtype=dtypes,
    usecols=cols_to_use,
    parse_dates=['timestamp'], # Parse dates explicitly
    low_memory=False # Allows pandas to infer dtypes more accurately
)
```

### 3.2 Use Optimal Data Types
Explicitly set the smallest appropriate dtype. `category` for repeated strings, `Int64` (nullable integer), `StringDtype` (nullable string), and smaller floats/integers save memory and improve performance.

❌ BAD
```python
df['id_col'] = df['id_col'].astype('int64') # Default, but maybe not needed
df['product_name'] = df['product_name'].astype('object') # Default string
```

✅ GOOD
```python
df['id_col'] = df['id_col'].astype('Int32') # Use nullable, smaller int
df['product_name'] = df['product_name'].astype('StringDtype') # Modern nullable string
df['region'] = df['region'].astype('category') # For categorical data
df['sales'] = df['sales'].astype('Float32') # Smaller float if precision allows
```

## 4. Common Pitfalls & Gotchas

### 4.1 Handle Missing Values Explicitly
Always address `NaN` values. Use `.fillna()`, `.dropna()`, or `.interpolate()` with clear strategies.

❌ BAD
```python
# Operations on columns with NaNs can lead to unexpected results or errors
df['revenue_per_sale'] = df['total_revenue'] / df['num_sales'] # NaNs propagate
```

✅ GOOD
```python
# Fill NaNs before calculation or drop them
df['num_sales'] = df['num_sales'].fillna(0) # Assume 0 sales if missing
df['revenue_per_sale'] = df['total_revenue'] / df['num_sales']

# Or drop rows with NaNs in critical columns
df_cleaned = df.dropna(subset=['total_revenue', 'num_sales'])
```

### 4.2 Be Mindful of Implicit Type Conversions
Operations can silently change dtypes. Monitor with `df.info()` and explicitly cast when necessary.

❌ BAD
```python
# Adding a float to an Int64 column can convert it to float64, losing nullable int
df['int_col'] = pd.Series([1, 2, pd.NA], dtype='Int64')
df['int_col'] = df['int_col'] + 0.5 # Converts to float64
```

✅ GOOD
```python
df['int_col'] = pd.Series([1, 2, pd.NA], dtype='Int64')
# If you need float, explicitly convert or accept the float result
df['float_col'] = df['int_col'].astype('Float64') + 0.5
```

## 5. Type Hints

### 5.1 Type Your Pandas Functions
Use `pd.DataFrame`, `pd.Series`, and specific dtypes for function signatures to improve code clarity, enable static analysis, and reduce bugs.

❌ BAD
```python
def process_data(data):
    # ... untyped operations ...
    return data
```

✅ GOOD
```python
from typing import Dict, Any
import pandas as pd

def process_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes raw sales data to calculate aggregated metrics.
    """
    df['total_price'] = df['quantity'] * df['unit_price']
    return df.groupby('product_id')['total_price'].sum().reset_index()

def create_empty_dataframe(columns: Dict[str, Any]) -> pd.DataFrame:
    """Creates an empty DataFrame with specified columns and dtypes."""
    return pd.DataFrame(columns=list(columns.keys())).astype(columns)

# Example usage with type hints
schema = {
    'product_id': 'Int64',
    'quantity': 'Int32',
    'unit_price': 'Float32'
}
empty_df = create_empty_dataframe(schema)
```

## 6. Virtual Environments

### 6.1 Always Use Virtual Environments
Isolate project dependencies using `venv` or `conda`. This prevents dependency conflicts and ensures reproducible environments.

```bash
# Create a virtual environment
python -m venv .venv
# Activate it
source .venv/bin/activate # On Windows: .venv\Scripts\activate
# Install pandas
pip install pandas ruff black isort pytest
```

## 7. Packaging

### 7.1 Manage Dependencies with `pyproject.toml`
For any project beyond a simple script, use `pyproject.toml` (with `poetry` or `setuptools`) to declare dependencies, including `pandas`.

```toml
# pyproject.toml
[project]
name = "my_data_pipeline"
version = "0.1.0"
dependencies = [
    "pandas>=2.3.0",
    "numpy",
    "pyarrow", # Recommended for faster I/O and nullable dtypes
]

[tool.ruff]
line-length = 88
target-version = "py310"
```

## 8. Testing Approaches

### 8.1 Write Comprehensive Tests with `pytest`
Ensure data transformations are correct and robust. Use `pandas.testing` for DataFrame and Series comparisons.

```python
# tests/test_data_processing.py
import pandas as pd
from pandas.testing import assert_frame_equal
from my_package.data_processor import process_sales_data

def test_process_sales_data():
    # Arrange
    raw_data = pd.DataFrame({
        'product_id': [1, 1, 2, 2],
        'quantity': [10, 5, 20, 10],
        'unit_price': [1.0, 2.0, 0.5, 1.5]
    })
    expected_data = pd.DataFrame({
        'product_id': [1, 2],
        'total_price': [20.0, 25.0]
    })

    # Act
    result_df = process_sales_data(raw_data)

    # Assert
    assert_frame_equal(result_df, expected_data)

def test_process_sales_data_with_nans():
    # Arrange
    raw_data_with_nan = pd.DataFrame({
        'product_id': [1, 1, 2, 2, 3],
        'quantity': [10, 5, 20, 10, pd.NA],
        'unit_price': [1.0, 2.0, 0.5, 1.5, 10.0]
    }, dtype={'quantity': 'Int64'}) # Use nullable Int64

    # Act (assuming process_sales_data handles NaNs gracefully, e.g., drops rows or fills)
    # For this example, let's assume it drops rows with NA in quantity
    result_df = process_sales_data(raw_data_with_nan.dropna(subset=['quantity']))

    expected_data = pd.DataFrame({
        'product_id': [1, 2],
        'total_price': [20.0, 25.0]
    })
    assert_frame_equal(result_df, expected_data)
```