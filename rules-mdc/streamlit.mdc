---
description: Enforce modern, performant, and maintainable Streamlit development practices, focusing on caching, modularity, and robust dependency management.
globs: **/*.py
---
# Streamlit Best Practices

This guide outlines the definitive best practices for building robust, performant, and maintainable Streamlit applications. Adhere to these rules to ensure your apps are efficient, scalable, and easy to collaborate on.

## 1. Code Organization and Structure

Always separate UI logic from data processing and business logic. Modularize your application into distinct files or directories.

### 1.1 Project Structure

Organize your project for clarity and maintainability.

❌ BAD: Monolithic `app.py`
```python
# app.py
import streamlit as st
import pandas as pd

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    # ... complex processing ...
    return df

st.title("My Data App")
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    data = load_and_process_data(uploaded_file)
    st.dataframe(data)
    # ... more UI and logic ...
```

✅ GOOD: Modularized Structure
```
my_streamlit_app/
├── app.py                  # Main entry point, orchestrates pages
├── pages/
│   └── dashboard_page.py   # Specific page UI and logic
├── services/
│   └── data_service.py     # Data loading and processing functions
├── components/
│   └── custom_widgets.py   # Reusable UI components
├── requirements.txt
└── .gitignore
```

`app.py`:
```python
# app.py
import streamlit as st
from pages import dashboard_page # Assuming dashboard_page.py has a main() function

st.set_page_config(layout="wide")

# Example of multi-page app structure
PAGES = {
    "Dashboard": dashboard_page,
    # "Another Page": another_page_module,
}

with st.sidebar:
    st.title('Navigation')
    selection = st.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.main() # Each page module must have a main() function
```

`pages/dashboard_page.py`:
```python
# pages/dashboard_page.py
import streamlit as st
from services.data_service import load_data, process_data

def main():
    st.title("Dashboard Overview")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        raw_df = load_data(uploaded_file)
        st.write("Raw Data:")
        st.dataframe(raw_df.head())

        processed_df = process_data(raw_df)
        st.write("Processed Data:")
        st.dataframe(processed_df)

        # Further UI elements specific to the dashboard
```

`services/data_service.py`:
```python
# services/data_service.py
import pandas as pd
import streamlit as st
from typing import Union

@st.cache_data
def load_data(file_path: Union[str, bytes]) -> pd.DataFrame:
    """Loads data from a file-like object or path."""
    return pd.read_csv(file_path)

@st.cache_data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs complex data processing."""
    # Example: clean, transform, aggregate
    processed_df = df.dropna().copy()
    processed_df['new_col'] = processed_df['value'] * 2 # Example transformation
    return processed_df
```

## 2. Performance Considerations

Streamlit reruns the entire script on every interaction. Efficient caching and state management are critical.

### 2.1 Caching Expensive Operations

Always cache functions that load data, perform heavy computations, or interact with external resources. Use `st.cache_data` for serializable data and `st.cache_resource` for unserializable objects like database connections or ML models.

❌ BAD: Uncached data loading
```python
import streamlit as st
import pandas as pd

def get_large_dataset():
    # Simulates a slow data load
    return pd.read_csv("https://example.com/large_data.csv")

df = get_large_dataset() # Runs on every interaction
st.dataframe(df)
st.button("Rerun App")
```

✅ GOOD: Cached data loading
```python
import streamlit as st
import pandas as pd

@st.cache_data
def get_large_dataset() -> pd.DataFrame:
    """Loads a large dataset from a URL."""
    # Simulates a slow data load
    return pd.read_csv("https://example.com/large_data.csv")

df = get_large_dataset() # Only runs once per unique URL/function code
st.dataframe(df)
st.button("Rerun App")
```

### 2.2 Session State for User-Specific Data

Use `st.session_state` to store and persist user-specific variables across reruns. This is essential for interactive elements where state must be maintained.

❌ BAD: Global variables or re-initializing state
```python
import streamlit as st

if 'counter' not in st.session_state:
    counter = 0 # This will reset on every rerun!

st.session_state.counter = counter # Incorrect assignment

if st.button("Increment"):
    st.session_state.counter += 1 # This will fail if counter is not in session_state initially

st.write(f"Counter: {st.session_state.counter}")
```

✅ GOOD: Proper `st.session_state` initialization and usage
```python
import streamlit as st

# Initialize state only if it doesn't exist
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.write(f"Counter: {st.session_state.counter}")

if st.button("Increment"):
    st.session_state.counter += 1

# Example for input widgets
name = st.text_input("Enter your name", key="user_name_input")
st.write(f"Hello, {st.session_state.user_name_input}!")
```

## 3. Common Patterns and Anti-patterns

### 3.1 Separate UI from Logic

Keep Streamlit UI calls (`st.write`, `st.button`, etc.) distinct from core business logic.

❌ BAD: Intermingled UI and logic
```python
import streamlit as st

def calculate_and_display(a, b):
    result = a + b
    st.write(f"The sum is: {result}") # UI call inside logic

num1 = st.number_input("First number", value=1)
num2 = st.number_input("Second number", value=2)
calculate_and_display(num1, num2)
```

✅ GOOD: Clear separation
```python
import streamlit as st

def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two numbers."""
    return a + b

num1 = st.number_input("First number", value=1)
num2 = st.number_input("Second number", value=2)

sum_result = calculate_sum(num1, num2)
st.write(f"The sum is: {sum_result}") # UI call outside logic
```

### 3.2 Use Context Managers for Layout

Leverage `st.sidebar`, `st.columns`, `st.expander`, `st.container`, and `st.tabs` as context managers for cleaner layout code.

❌ BAD: Manual layout with repeated elements
```python
import streamlit as st

col1, col2 = st.columns(2)
col1.write("Column 1 content")
col2.write("Column 2 content")

st.sidebar.header("Sidebar")
st.sidebar.write("Sidebar content")
```

✅ GOOD: Context managers for structured layout
```python
import streamlit as st

with st.sidebar:
    st.header("App Controls")
    option = st.selectbox("Choose an option", ["A", "B", "C"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Input")
    st.write(f"Selected option: {option}")
    # ... more content ...

with col2:
    st.subheader("Visualization")
    st.line_chart([1, 2, 3])
    # ... more content ...
```

## 4. Common Pitfalls and Gotchas

### 4.1 Modifying Cached Objects

`st.cache_data` returns a *copy* of the cached object, preventing unintended mutations. `st.cache_resource` returns the *original* object, so be cautious with mutations.

❌ BAD: Modifying `st.cache_resource` return value without care
```python
import streamlit as st

@st.cache_resource
def get_shared_list():
    return [1, 2, 3]

my_list = get_shared_list()
st.write(f"Original list: {my_list}")

if st.button("Append to list"):
    my_list.append(4) # This modifies the cached object for ALL users/sessions!
    st.write("Appended 4")

st.write(f"Current list: {my_list}")
```

✅ GOOD: Treat `st.cache_resource` objects as immutable or explicitly copy
```python
import streamlit as st

@st.cache_resource
def get_shared_list():
    return [1, 2, 3]

# Option 1: Treat as immutable
st.write(f"Shared list: {get_shared_list()}")

# Option 2: Create a local copy if modification is needed
my_list_copy = list(get_shared_list())
st.write(f"Original copy: {my_list_copy}")

if st.button("Append to list"):
    my_list_copy.append(4) # Only modifies the copy for this session
    st.write("Appended 4")

st.write(f"Current copy: {my_list_copy}")
```

### 4.2 Widget Keys

Always provide unique `key` arguments for widgets that appear multiple times or whose state needs to be explicitly managed. This prevents unexpected behavior when widgets are dynamically rendered or reordered.

❌ BAD: Missing or duplicate keys
```python
import streamlit as st

for i in range(3):
    st.checkbox(f"Option {i}") # Keys will be implicitly generated, might conflict

st.text_input("Enter value")
st.text_input("Enter another value") # These will implicitly get the same key if not specified
```

✅ GOOD: Unique and descriptive keys
```python
import streamlit as st

for i in range(3):
    st.checkbox(f"Option {i}", key=f"checkbox_{i}")

st.text_input("Enter value", key="first_input")
st.text_input("Enter another value", key="second_input")
```

## 5. Type Hints

Always use type hints for function signatures and complex variables. This improves code readability, enables static analysis, and aids collaboration.

❌ BAD: Untyped function
```python
def process_data(data):
    # What is 'data'? A list, DataFrame, dict?
    return len(data)
```

✅ GOOD: Type-hinted function
```python
import pandas as pd
from typing import List, Dict, Union

def process_data(data: Union[pd.DataFrame, List[Dict]]) -> int:
    """Processes data and returns its length."""
    if isinstance(data, pd.DataFrame):
        return len(data)
    elif isinstance(data, list):
        return len(data)
    else:
        raise TypeError("Unsupported data type")
```

## 6. Packaging and Dependencies

Manage your dependencies rigorously using `requirements.txt` for Python packages and `packages.txt` for system-level dependencies.

### 6.1 Python Dependencies (`requirements.txt`)

Pin exact versions or use clear version constraints.

❌ BAD: Vague `requirements.txt`
```
pandas
streamlit
matplotlib
```

✅ GOOD: Specific `requirements.txt`
```
streamlit==1.52.1
pandas>=2.1.0,<3.0.0
matplotlib~=3.8.0
```

### 6.2 System Dependencies (`packages.txt`)

If your app requires non-Python libraries (e.g., for image processing, database connectors), list them in `packages.txt` for Streamlit Community Cloud.

`packages.txt` example:
```
build-essential
libgl1-mesa-glx
```

## 7. Testing Approaches

While Streamlit apps are primarily UI-driven, you must test your underlying logic.

### 7.1 Unit Testing Core Logic

Isolate and unit test all non-UI functions (e.g., data processing, calculations, API calls). Use standard Python testing frameworks like `pytest`.

`test_data_service.py`:
```python
# tests/test_data_service.py
import pandas as pd
from services.data_service import process_data

def test_process_data_dropna():
    df = pd.DataFrame({'value': [1, 2, None, 4]})
    processed_df = process_data(df)
    assert len(processed_df) == 3
    assert 'new_col' in processed_df.columns
    assert processed_df['new_col'].iloc[0] == 2

def test_process_data_empty_df():
    df = pd.DataFrame({'value': []})
    processed_df = process_data(df)
    assert processed_df.empty
```

### 7.2 End-to-End Testing (Optional, for complex apps)

For complex applications, consider tools like `Playwright` or `Selenium` to simulate user interactions and verify UI behavior. However, prioritize robust unit tests for core logic first, as E2E testing adds significant overhead.