---
description: Definitive guidelines for writing maintainable, performant, and reproducible Plotly visualizations using modern Python best practices, especially for AI/ML workflows.
globs: **/*.py
---
# Plotly Best Practices

Plotly 6.5.0 is a powerful tool for interactive, publication-quality visualizations, crucial for modern AI/ML workflows. These rules ensure your Plotly code is maintainable, reproducible, and production-ready, leveraging the latest features including AI-assisted Studio tools.

## Code Organization and Structure

### 1. Separate Data Preparation from Visualization
Keep data preprocessing, model outputs, and metric calculations distinct from plotting logic. This improves debugging and reusability.

❌ BAD:
```python
import pandas as pd
import plotly.express as px

def analyze_and_plot_data(raw_data: pd.DataFrame):
    # Data processing mixed with plotting
    processed_df = raw_data[raw_data['value'] > 0]
    processed_df['normalized'] = processed_df['value'] / processed_df['value'].max()
    fig = px.scatter(processed_df, x='timestamp', y='normalized', title='Normalized Data')
    fig.show()
```

✅ GOOD:
```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data for visualization."""
    processed_df = raw_data[raw_data['value'] > 0]
    processed_df['normalized'] = processed_df['value'] / processed_df['value'].max()
    return processed_df

def create_normalized_scatter_plot(data: pd.DataFrame) -> go.Figure:
    """Creates a scatter plot from processed data."""
    fig = px.scatter(data, x='timestamp', y='normalized', title='Normalized Data Over Time')
    return fig

# Usage example:
# raw_df = pd.read_csv('my_data.csv')
# clean_df = preprocess_data(raw_df)
# plot_fig = create_normalized_scatter_plot(clean_df)
# plot_fig.show()
```

### 2. Modular Figure Construction
Treat the `Figure` object as a modular data structure. Build traces, then define the layout. This mirrors Plotly's internal design and enhances readability.

❌ BAD:
```python
import plotly.graph_objects as go

# Hard-to-read, deeply nested dictionary for figure definition
fig = go.Figure(
    data=[{'type': 'bar', 'x': [1, 2, 3], 'y': [10, 15, 13]}],
    layout={'title': {'text': 'Sales Data'}, 'xaxis': {'title': {'text': 'Month'}}}
)
```

✅ GOOD:
```python
import plotly.graph_objects as go

# Separate data (traces) and layout definition
trace1 = go.Bar(x=[1, 2, 3], y=[10, 15, 13], name='Product A')
trace2 = go.Bar(x=[1, 2, 3], y=[12, 10, 18], name='Product B')

fig = go.Figure(data=[trace1, trace2])
fig.update_layout(
    title_text='Monthly Sales Data',
    xaxis_title_text='Month',
    yaxis_title_text='Revenue ($)',
    legend_title_text='Product'
)
```

## Common Patterns and Anti-patterns

### 3. Prefer Plotly Express for Most Charts
Use `plotly.express` for common chart types. It's terse, consistent, and returns a full `go.Figure` object, allowing easy customization with `graph_objects` methods.

❌ BAD:
```python
import plotly.graph_objects as go
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 1, 2], 'category': ['A', 'B', 'A']})
# Manually creating a scatter plot with graph_objects for a simple case
fig = go.Figure(data=[
    go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color=df['category'].map({'A': 'red', 'B': 'blue'})))
])
fig.update_layout(title_text='Custom Scatter Plot')
```

✅ GOOD:
```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 1, 2], 'category': ['A', 'B', 'A']})
# Plotly Express handles common mappings automatically
fig = px.scatter(df, x='x', y='y', color='category', title='Simple Scatter Plot')
# Further customization is still possible
fig.update_traces(marker_size=10)
```

### 4. Apply Consistent Labeling and Theming
Ensure all plots meet publication-quality standards with clear titles, descriptive axis labels, and consistent fonts. Leverage `labels` in `px` and `update_layout` for global styling.

❌ BAD:
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_length', y='sepal_width') # No title, default labels
fig.show()
```

✅ GOOD:
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df,
                 x='sepal_length',
                 y='sepal_width',
                 color='species',
                 labels={
                     'sepal_length': 'Sepal Length (cm)',
                     'sepal_width': 'Sepal Width (cm)',
                     'species': 'Iris Species'
                 },
                 title='Iris Sepal Dimensions by Species')

fig.update_layout(
    font_family="Arial",
    font_color="darkblue",
    title_font_size=20,
    hoverlabel_font_family="monospace"
)
fig.show()
```

## Performance Considerations

### 5. Optimize for Large Datasets
For extremely large datasets, consider data aggregation or sampling before plotting to maintain interactivity. Plotly.js can handle many points, but browser rendering has limits. Use `go.Scattergl` for hardware-accelerated rendering of large scatter plots.

❌ BAD:
```python
import plotly.express as px
import numpy as np
import pandas as pd

# Plotting 10 million points directly, which might be slow
large_df = pd.DataFrame({'x': np.random.rand(10_000_000), 'y': np.random.rand(10_000_000)})
fig = px.scatter(large_df, x='x', y='y')
fig.show() # Might freeze browser or be very sluggish
```

✅ GOOD:
```python
import plotly.graph_objects as go # Use go for Scattergl
import numpy as np
import pandas as pd

# For very large scatter plots, use go.Scattergl for WebGL acceleration
large_df = pd.DataFrame({'x': np.random.rand(10_000_000), 'y': np.random.rand(10_000_000)})

fig = go.Figure(data=[go.Scattergl(x=large_df['x'], y=large_df['y'], mode='markers')])
fig.update_layout(title_text='Large Data Scatter Plot (WebGL)')
fig.show()

# Alternatively, for other chart types, consider sampling or aggregation
# sampled_df = large_df.sample(n=100_000)
# fig = px.scatter(sampled_df, x='x', y='y', title='Sampled Data Scatter Plot')
# fig.show()
```

### 6. Choose Appropriate Output Format
For static reports or non-interactive embedding, export figures as images. For interactive dashboards, use `fig.show()` or integrate with Dash.

```python
import plotly.express as px
import pandas as pd

df = px.data.iris()
fig = px.scatter(df, x='sepal_length', y='sepal_width', color='species')

# ✅ GOOD: For interactive display in notebooks/scripts
fig.show()

# ✅ GOOD: For static image export (requires kaleido: pip install kaleido)
fig.write_image("iris_scatter.png", width=800, height=600)

# ✅ GOOD: For interactive HTML export
fig.write_html("iris_scatter.html")
```

## Common Pitfalls and Gotchas

### 7. Avoid Direct Dictionary Manipulation for Figures
While figures can be represented as dictionaries, directly building or modifying them this way bypasses Plotly's validation and convenience methods. Always use `plotly.graph_objects.Figure` or `plotly.express`.

❌ BAD:
```python
# No validation, easy to make typos
fig = {
    "data": [{"type": "bar", "x": [1, 2], "y": [3, 4]}],
    "layout": {"title": {"text": "My Bar Chart"}}
}
import plotly.io as pio
pio.show(fig)
```

✅ GOOD:
```python
import plotly.graph_objects as go
# Benefits from validation, docstrings, and update methods
fig = go.Figure(data=[go.Bar(x=[1, 2], y=[3, 4])])
fig.update_layout(title_text="My Bar Chart")
fig.show()
```

### 8. Leverage `update_layout` and `update_traces`
These methods provide a clean, efficient way to modify existing figures without recreating them. Use "magic underscores" for concise attribute setting.

❌ BAD:
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_length', y='sepal_width')

# Reassigning individual attributes directly is verbose and less robust
fig.layout.title.text = 'Iris Data'
fig.layout.xaxis.title.text = 'Length'
fig.layout.yaxis.title.text = 'Width'
```

✅ GOOD:
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_length', y='sepal_width')

# Use update methods with magic underscores for clarity and efficiency
fig.update_layout(
    title_text='Iris Sepal Measurements',
    xaxis_title_text='Sepal Length (cm)',
    yaxis_title_text='Sepal Width (cm)',
    font_size=12
)
fig.update_traces(marker_opacity=0.7, selector=dict(type='scatter')) # Apply to all scatter traces
```

## Type Hints

### 9. Use Type Hints for Clarity and Maintainability
Explicitly type hint Plotly `Figure` objects and related data structures (`pd.DataFrame`) in function signatures. This improves code readability and enables static analysis.

❌ BAD:
```python
import plotly.express as px
def create_plot(data): # Untyped input and output
    fig = px.line(data, x='date', y='value')
    return fig
```

✅ GOOD:
```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_time_series_plot(data: pd.DataFrame) -> go.Figure:
    """Generates a time series line plot."""
    fig = px.line(data, x='date', y='value', title='Time Series Analysis')
    return fig
```

## Virtual Environments

### 10. Always Use Virtual Environments
Isolate your project dependencies. This is a fundamental Python best practice, critical for reproducibility, especially in AI/ML where library versions can significantly impact results.

✅ GOOD:
```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install plotly pandas kaleido
```

## Packaging

### 11. Package Visualization Code
When sharing or deploying your visualization logic, package it properly (e.g., using `pyproject.toml` and `setuptools`). This ensures consistent installation and dependency management.

✅ GOOD:
```toml
# pyproject.toml example
[project]
name = "my_ml_viz"
version = "0.1.0"
dependencies = [
    "plotly>=6.5.0",
    "pandas",
    "kaleido"
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## Testing Approaches

### 12. Implement Testing for Visualization Logic
Test functions that generate figures, focusing on data input validation and layout configuration. For complex visual changes, consider visual regression testing.

✅ GOOD:
```python
import pytest
import pandas as pd
import plotly.graph_objects as go
# Assuming create_time_series_plot is in a module named my_viz_module
# from my_viz_module import create_time_series_plot 

def create_time_series_plot(data: pd.DataFrame) -> go.Figure:
    """Generates a time series line plot (for testing purposes)."""
    fig = px.line(data, x='date', y='value', title='Time Series Analysis')
    return fig

def test_create_time_series_plot():
    # Test data
    test_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'value': [10, 12, 8]
    })

    fig = create_time_series_plot(test_data)

    # Assertions about the figure structure and content
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == 'scatter' # px.line creates scatter traces
    assert fig.layout.title.text == 'Time Series Analysis'
    assert fig.data[0].x.tolist() == test_data['date'].tolist()
    assert fig.data[0].y.tolist() == test_data['value'].tolist()

    # For visual regression, you'd compare fig.to_json() or a generated