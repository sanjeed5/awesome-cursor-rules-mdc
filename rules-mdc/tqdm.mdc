---
description: This guide defines best practices for using the `tqdm` library to implement clear, efficient, and robust progress bars in Python applications, ensuring consistent user feedback and maintainable code.
globs: **/*.py
---
# tqdm Best Practices

`tqdm` is the definitive library for adding progress bars to Python applications. It's lightweight, fast, and works across various environments (terminal, Jupyter). This guide outlines the mandatory practices for its use.

## 1. Import Strategy

Always import `tqdm` directly from the library for clarity. Use `trange` as a convenient shortcut for `tqdm(range(N))`.

❌ BAD: Generic import
```python
import tqdm
for i in tqdm.tqdm(range(100)):
    pass
```

✅ GOOD: Direct and clear imports
```python
from tqdm import tqdm, trange

for i in trange(100):  # Use trange for simple range loops
    pass

data = [1, 2, 3] * 10
for item in tqdm(data, desc="Processing items"): # Use tqdm for iterables
    pass
```

## 2. Prefer Context Managers

Use `tqdm` as a context manager (`with ... as pbar:`) whenever possible. This ensures the progress bar is properly closed and resources are released, even if exceptions occur.

❌ BAD: Manual `close()` (prone to errors)
```python
pbar = tqdm(total=100)
for i in range(100):
    pbar.update(1)
    # What if an error occurs here? pbar.close() might be skipped.
pbar.close()
```

✅ GOOD: Context manager for robust cleanup
```python
with tqdm(total=100, desc="Calculating") as pbar:
    for i in range(100):
        # Simulate work
        if i == 50:
            # Example of an error, pbar will still close
            # raise ValueError("Something went wrong!")
            pass
        pbar.update(1)
```

## 3. Manual Updates for Non-Iterables or Custom Logic

When wrapping an iterable isn't feasible (e.g., callbacks, async operations, or custom progress logic), use `pbar.update(n)`. **Always provide `total`** when using manual updates.

❌ BAD: Missing `total` with manual updates (bar won't show percentage/ETA)
```python
pbar = tqdm(desc="Downloading")
# ... later in a callback ...
pbar.update(chunk_size)
```

✅ GOOD: Explicit `total` for manual updates
```python
import time
import random

file_size = 1024 * 1024 * 50  # 50 MB
downloaded_bytes = 0

with tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading file") as pbar:
    while downloaded_bytes < file_size:
        chunk = random.randint(1024, 1024 * 10) # Simulate chunk download
        chunk = min(chunk, file_size - downloaded_bytes)
        downloaded_bytes += chunk
        pbar.update(chunk)
        time.sleep(0.01)
```

## 4. Use Descriptive Labels and Postfix Information

Enhance readability by providing a `desc` (description) and dynamic `postfix` information. `set_description()` and `set_postfix()` are preferred for updating these dynamically.

❌ BAD: Static or cluttered description
```python
for epoch in tqdm(range(10), desc="Epoch"):
    loss = 0.5 - epoch * 0.01
    # This doesn't update dynamically without re-creating the bar
    for batch in tqdm(range(100), desc=f"Epoch {epoch} Loss: {loss:.4f}", leave=False):
        pass
```

✅ GOOD: Dynamic description and postfix updates
```python
import random

for epoch in tqdm(range(10), desc="Training Epochs"):
    total_loss = 0.0
    for batch in trange(100, desc=f"Batch {epoch+1}", leave=False):
        current_loss = random.uniform(0.1, 0.5)
        total_loss += current_loss
        # Update postfix with real-time metrics
        tqdm.set_postfix(loss=f"{current_loss:.4f}", avg_loss=f"{total_loss/(batch+1):.4f}")
        # Simulate work
        time.sleep(0.005)
```

## 5. Handle Output with `tqdm.write()`

Never use `print()` directly when a `tqdm` bar is active. It will corrupt the progress bar display. Use `tqdm.write()` instead, which intelligently prints messages without interfering.

❌ BAD: Corrupts progress bar
```python
for i in trange(10):
    if i == 5:
        print("Mid-loop message!") # This will mess up the bar
    time.sleep(0.1)
```

✅ GOOD: Preserves progress bar integrity
```python
for i in trange(10, desc="Processing"):
    if i == 5:
        tqdm.write("Mid-loop message: Halfway there!")
    time.sleep(0.1)
```

## 6. Fine-tune Display Options

Use `leave=False` for inner loops to remove them upon completion, keeping the terminal clean. Leverage `unit`, `unit_scale`, and `dynamic_ncols` for better presentation.

❌ BAD: Cluttered terminal with nested bars
```python
for i in tqdm(range(3), desc="Outer"):
    for j in tqdm(range(5), desc="Inner"): # leave=True by default
        time.sleep(0.05)
```

✅ GOOD: Clean nested bars and scaled units
```python
for i in tqdm(range(3), desc="Outer Loop"):
    for j in tqdm(range(5), desc="Inner Loop", leave=False): # Bar disappears after inner loop finishes
        time.sleep(0.05)

# Example with units
with tqdm(total=100_000_000, unit="B", unit_scale=True, dynamic_ncols=True, desc="Transferring") as pbar:
    for _ in range(100):
        pbar.update(1_000_000)
        time.sleep(0.01)
```

## 7. Integrate with Data Science Libraries

`tqdm.pandas()` and `tqdm.auto` provide seamless integration for common data science workflows.

### Pandas DataFrames

Use `tqdm.pandas()` to add a `.progress_apply()` method to DataFrames.

```python
import pandas as pd
from tqdm.auto import tqdm # Use tqdm.auto for notebook compatibility

tqdm.pandas() # Register tqdm with pandas

df = pd.DataFrame({'col': range(100000)})

# Applying a function with a progress bar
df['new_col'] = df['col'].progress_apply(lambda x: x * 2)
```

### Jupyter Notebooks

`tqdm.auto` automatically detects if it's running in a Jupyter environment and uses the appropriate display.

```python
# In a .py file that might be run in a notebook or terminal
from tqdm.auto import tqdm, trange

def long_running_task():
    for i in trange(100, desc="Auto-detected progress"):
        time.sleep(0.01)

long_running_task()
```

## 8. Performance Considerations

`tqdm` is highly optimized, but for extremely fast loops (e.g., millions of iterations per second), consider batching updates or disabling the bar if not strictly necessary.

```python
# For loops where each iteration is extremely fast
total_iterations = 10_000_000
batch_size = 1000

with tqdm(total=total_iterations, desc="Batch Processing") as pbar:
    for i in range(0, total_iterations, batch_size):
        # Perform batch_size operations here
        # ...
        pbar.update(batch_size) # Update once per batch
```