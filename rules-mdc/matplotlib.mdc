---
description: This guide outlines definitive best practices for writing clean, performant, and maintainable matplotlib code, emphasizing the object-oriented API and modern data science workflows.
globs: **/*.py
---
# matplotlib Best Practices

Matplotlib is the bedrock of Python data visualization. Adhering to these guidelines ensures your plots are not just visually appealing, but also robust, performant, and easily integrated into production AI/ML and data science pipelines.

## 1. Code Organization & Structure

### 1.1 Standard Imports

Always use the conventional aliases. This improves readability and consistency across projects.

❌ BAD:
```python
import numpy
import matplotlib.pyplot
import matplotlib
```

✅ GOOD:
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
```

### 1.2 Object-Oriented API First

Prioritize the object-oriented API (`Figure` and `Axes` objects) over `pyplot`'s stateful interface. This leads to more explicit, reproducible, and testable code, especially in functions or classes.

❌ BAD (Stateful `pyplot`):
```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("My Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show() # Blocks execution
```

✅ GOOD (Object-Oriented):
```python
from typing import List
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes

def create_my_plot(data_x: List[float], data_y: List[float], title: str, x_label: str, y_label: str) -> mpl_figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data_x, data_y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig

# Usage in a script or notebook
if __name__ == "__main__":
    x = [1, 2, 3]
    y = [4, 5, 6]
    my_figure = create_my_plot(x, y, "My Plot", "X-axis", "Y-axis")
    my_figure.savefig("my_plot.png") # Non-blocking save
    # If interactive display is needed, use:
    # plt.show()
```

### 1.3 Consistent Variable Naming

Use the standard variable names for `Figure` and `Axes` objects.

❌ BAD:
```python
my_figure_object, my_axis_object = plt.subplots()
```

✅ GOOD:
```python
fig, ax = plt.subplots()
# For multiple axes
fig, axs = plt.subplots(2, 2)
```

### 1.4 `rcParams` Access

Always access `rcParams` via `mpl.rcParams` to avoid import issues in early-loaded modules.

❌ BAD:
```python
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
```

✅ GOOD:
```python
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 6)
```

## 2. Common Patterns & Anti-patterns

### 2.1 Handling Dense Data

For dense scatter plots, use `alpha` for transparency. For very large collections, rasterize them to keep file sizes manageable.

✅ GOOD:
```python
# Dense scatter plot with transparency
ax.scatter(X, Y, s=40, c='C1', alpha=0.1)

# Rasterizing large collections for PDF/SVG output
ax.scatter(X_large, Y_large, rasterized=True)
fig.savefig("rasterized_figure.pdf", dpi=600)
```

### 2.2 Publication-Quality Styling

Use custom linestyles with rounded caps and text path effects for professional figures.

✅ GOOD:
```python
# Rounded dotted lines
ax.plot([0, 1], [0, 0], "C1", linestyle=(0, (0.01, 1)), dash_capstyle="round")

# Text outline for visibility
import matplotlib.patheffects as fx
text = ax.text(0.5, 0.1, "Label", transform=ax.transAxes)
text.set_path_effects([
    fx.Stroke(linewidth=3, foreground='white'),
    fx.Normal()
])
```

### 2.3 Managing Plot Margins

Always call `fig.tight_layout()` before saving or showing a figure to automatically adjust subplot parameters for a tight layout.

❌ BAD:
```python
fig, ax = plt.subplots()
ax.plot(x, y)
fig.savefig("plot_with_margins.png") # May have excessive whitespace
```

✅ GOOD:
```python
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout() # Adjusts subplot params for tight layout
fig.savefig("plot_tight.png")
```

### 2.4 `**kwargs` Usage

Reserve `**kwargs` for pass-through arguments. Explicitly define keyword-only arguments for parameters consumed locally.

❌ BAD:
```python
def my_plot_function(x, y, **kwargs):
    label = kwargs.pop('label', 'default') # Consuming from kwargs
    ax.plot(x, y, label=label, **kwargs)
```

✅ GOOD:
```python
def my_plot_function(x, y, *, label='default', **kwargs): # Keyword-only 'label'
    ax.plot(x, y, label=label, **kwargs)
```

## 3. Performance Considerations

### 3.1 Avoid `plt.show()` in Libraries/Functions

`plt.show()` blocks execution and is intended for interactive display. In reusable code, return the `Figure` object or save it directly. This allows callers to decide how to handle the figure.

❌ BAD:
```python
def generate_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show() # Blocks execution, not suitable for libraries
```

✅ GOOD:
```python
def generate_plot(data) -> mpl_figure.Figure:
    fig, ax = plt.subplots()
    ax.plot(data)
    return fig

# In main script:
if __name__ == "__main__":
    my_data = [1, 5, 2, 8]
    plot_fig = generate_plot(my_data)
    plot_fig.savefig("my_generated_plot.pdf")
    # If interactive display is desired:
    # plt.show()
```

### 3.2 Explicitly Close Figures

When creating many figures programmatically (e.g., in loops or batch processing), explicitly close them to prevent memory leaks.

✅ GOOD:
```python
for i in range(100):
    fig, ax = plt.subplots()
    ax.plot([0, i], [i, 0])
    fig.savefig(f"plot_{i}.png")
    plt.close(fig) # Essential for memory management
```

## 4. Type Hints

Embrace type hints for all new and modified public APIs. Use stub files (`.pyi`) where appropriate, or inline for simpler modules like `pyplot.py`.

✅ GOOD:
```python
from typing import List
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes

def plot_data(x: List[float], y: List[float], color: str = 'blue') -> mpl_figure.Figure:
    """Plots x and y data on a new figure."""
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color)
    return fig
```

## 5. Virtual Environments & Packaging

Always use virtual environments (`venv`, `conda`) for dependency management. Pin Matplotlib to a specific stable version in `requirements.txt` or `pyproject.toml` to ensure reproducibility.

✅ GOOD (`requirements.txt`):
```
matplotlib==3.10.0
numpy==1.26.0
pandas==2.1.0
seaborn==0.13.0
```

## 6. Testing Approaches

Integrate `ruff check --fix` into your CI/CD pipeline. Add specific tests for new features and bug fixes, especially image comparison tests for visual changes.

✅ GOOD (Example test structure):
```python
# tests/test_plotting.py
import matplotlib.pyplot as plt
import numpy as np
from my_module import generate_plot # Assuming generate_plot is in my_module

def test_generate_plot_output():
    data = np.array([1, 2, 3, 4, 5])
    fig = generate_plot(data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    # Add more assertions, e.g., check title, labels, data points
    plt.close(fig) # Clean up figure after test

# For visual regression testing (requires specific tooling like pytest-mpl)
# def test_plot_visual_regression(mpl_image_compare):
#     fig = generate_plot(np.array([1, 5, 2, 8]))
#     assert mpl_image_compare(fig, "expected_plot_output")
```