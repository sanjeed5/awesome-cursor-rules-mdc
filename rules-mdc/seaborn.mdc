---
description: This guide defines best practices for using Seaborn in Python for data visualization, emphasizing modern, reproducible, and performant approaches in AI/ML pipelines.
globs: **/*.py
---
# seaborn Best Practices

Seaborn is the definitive library for statistical data visualization in Python. This guide outlines our team's mandatory best practices to ensure consistent, reproducible, and high-quality plots in all AI/ML projects.

## 1. Code Organization & Project-Wide Styling

Always configure Seaborn's global style settings once per project. Centralize this in a dedicated `styles.py` module and import it. This guarantees a uniform visual language across all outputs.

❌ BAD: Ad-hoc styling in every script/notebook
```python
# my_script.py
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis") # In every file!
# ... plotting code
```

✅ GOOD: Centralized and imported styling
```python
# styles.py
import seaborn as sns
import matplotlib.pyplot as plt

def apply_seaborn_defaults():
    """Applies project-wide Seaborn and Matplotlib styling."""
    sns.set_theme(
        context="talk",      # Readable text for presentations/reports
        style="whitegrid",   # Clean background with light grid
        palette="deep"       # Perceptually uniform, colorblind-friendly
    )
    plt.rcParams["figure.figsize"] = (8, 5)    # Consistent figure size
    plt.rcParams["figure.dpi"] = 150           # On-screen clarity

# my_script.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .styles import apply_seaborn_defaults # Adjust import path as needed

apply_seaborn_defaults()
# ... plotting code
```

## 2. Prefer Figure-Level Functions

For complex layouts, especially with faceting, always use Seaborn's figure-level functions (`relplot`, `catplot`, `pairplot`, `lmplot`). They handle figure creation, axis management, and legends automatically, ensuring consistency.

❌ BAD: Manual `subplots` with axes-level functions for faceting
```python
# Hard to manage multiple subplots, shared axes, and legends manually
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df[df['category'] == 'A'], x='x', y='y', ax=axes[0])
sns.scatterplot(data=df[df['category'] == 'B'], x='x', y='y', ax=axes[1])
plt.tight_layout()
plt.show()
```

✅ GOOD: `relplot` for automatic faceting and consistent sizing
```python
import pandas as pd
import numpy as np

# Example data
data = {
    'x': np.random.rand(100),
    'y': np.random.rand(100) + np.random.rand(100),
    'category': np.random.choice(['A', 'B'], 100),
    'group': np.random.choice(['X', 'Y'], 100)
}
df = pd.DataFrame(data)

g = sns.relplot(
    data=df, x='x', y='y',
    col='category', # Facet by category
    hue='group',    # Color by group
    kind='scatter',
    height=4, aspect=1.2
)
g.set_axis_labels("Feature X", "Feature Y")
g.set_titles("Category: {col_name}")
plt.tight_layout()
plt.show()
```

## 3. Data Format: Long-Form is King

Seaborn thrives on "tidy" or "long-form" data. Ensure your Pandas DataFrames are structured with one observation per row and variables as columns. This allows Seaborn to infer groupings automatically, reducing boilerplate.

❌ BAD: Wide-form data requiring `melt` or manual iteration
```python
# Data where 'value_A' and 'value_B' are separate columns for the same metric
wide_df = pd.DataFrame({
    'id': [1, 2], 'value_A': [10, 20], 'value_B': [15, 25]
})
# To plot, you'd need:
melted_df = wide_df.melt(id_vars=['id'], var_name='metric', value_name='value')
sns.barplot(data=melted_df, x='metric', y='value')
plt.show() # Added plt.show()
```

✅ GOOD: Data already in long-form
```python
long_df = pd.DataFrame({
    'id': [1, 2, 1, 2],
    'metric': ['value_A', 'value_A', 'value_B', 'value_B'],
    'value': [10, 20, 15, 25]
})
sns.barplot(data=long_df, x='metric', y='value')
plt.show()
```

## 4. Performance for Large Datasets

For datasets with millions of rows, plotting every point can be slow and create over-plotting. Optimize for clarity and speed.

*   **Down-sample**: Plot a representative subset for initial exploration.
*   **Binning/Aggregation**: Summarize data before plotting (e.g., `histplot` with `bins`, `kdeplot`, `lineplot` on aggregated data).
*   **Limit `hue`**: Excessive `hue` categories lead to cluttered legends and slow rendering.
*   **Efficient Data Loading**: Use `polars` or `duckdb` for faster initial data processing if `pandas` is a bottleneck.

```python
# Assume large_df is a very large DataFrame
# ✅ GOOD: Down-sampling for quick exploration
sampled_df = large_df.sample(n=100_000, random_state=42)
sns.scatterplot(data=sampled_df, x='feature1', y='feature2', alpha=0.5)
plt.show()

# ✅ GOOD: Aggregating for trends
daily_avg = large_df.groupby('date')['value'].mean().reset_index()
sns.lineplot(data=daily_avg, x='date', y='value')
plt.show()
```

## 5. Type Hints for Clarity

Always use explicit type hints for functions that generate plots. This improves code readability, maintainability, and enables static analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union

def plot_feature_distribution(
    df: pd.DataFrame,
    feature_col: str,
    hue_col: Optional[str] = None,
    title: str = "Feature Distribution"
) -> plt.Axes:
    """
    Generates a histogram and KDE plot for a given feature.

    Args:
        df: Input DataFrame.
        feature_col: Name of the column to plot.
        hue_col: Optional column for grouping.
        title: Plot title.

    Returns:
        The Matplotlib Axes object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=feature_col, hue=hue_col, kde=True, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return ax

# Example usage
tips = sns.load_dataset("tips")
plot_feature_distribution(tips, "total_bill", hue_col="sex", title="Total Bill Distribution by Sex")
plt.show()
```

## 6. Reproducibility: Virtual Environments & Packaging

Ensure full reproducibility of your plotting environment.

*   **Virtual Environments**: Always use `venv` or `conda` environments.
*   **`requirements.txt`**: Pin *all* dependencies (e.g., `seaborn`, `matplotlib`, `pandas`, `numpy`) to exact versions using `pip freeze > requirements.txt`.

```bash
# In your project root
python -m venv .venv
source .venv/bin/activate
pip install seaborn==0.13.2 matplotlib==3.8.2 pandas==2.1.4 numpy==1.26.2
pip freeze > requirements.txt
```

## 7. Testing: Visual Regression

For critical visualizations in production pipelines, implement visual regression tests. Compare hashes of generated PNGs or use a dedicated library (e.g., `pytest-mpl`) to detect unintended aesthetic changes caused by library updates or code modifications.

```python
# Example (conceptual)
# test_plots.py
import hashlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from your_project.styles import apply_seaborn_defaults # Adjust import

def generate_plot_image(filename: str):
    apply_seaborn_defaults()
    df = sns.load_dataset("iris")
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
    plt.savefig(filename, bbox_inches="tight")
    plt.close() # Crucial to close figures after saving

def test_iris_scatter_plot_regression():
    expected_image_path = "tests/baseline_iris_scatter.png"
    generated_image_path = "tests/temp_iris_scatter.png"

    generate_plot_image(generated_image_path)

    with open(expected_image_path, "rb") as f_expected:
        expected_hash = hashlib.md5(f_expected.read()).hexdigest()
    with open(generated_image_path, "rb") as f_generated:
        generated_hash = hashlib.md5(f_generated.read()).hexdigest()

    assert expected_hash == generated_hash, "Iris scatter plot has changed visually!"
    os.remove(generated_image_path) # Clean up
```

## 8. Documentation: Clear Docstrings

Document every plotting function with a clear docstring. Describe the data source, plot type, purpose, statistical assumptions, parameters, return values, and any side effects (e.g., `plt.show()`). This makes code audit-friendly and maintainable.

```python
# See example in Section 5 for a well-documented plotting function.
```