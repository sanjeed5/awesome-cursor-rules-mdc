---
description: This guide provides opinionated, actionable best practices for writing performant, maintainable, and idiomatic Dask code, focusing on real-world scenarios and modern patterns.
globs: **/*.py
---
# Dask Best Practices

Dask is a powerful library for scaling Python, but it introduces new paradigms. Writing effective Dask code requires understanding its lazy execution model and distributed computing overhead. This guide outlines critical best practices to ensure your Dask pipelines are reliable, performant, and easy to maintain.

## Critical Guidelines

### 1. Start Small and Profile First

Before reaching for Dask, ensure your problem genuinely requires distributed computing. Optimize your serial code first. Dask adds overhead; it's not a magic bullet for inefficient algorithms.

**✅ GOOD:** Profile your Pandas/NumPy code to identify bottlenecks. Only then consider Dask for the specific slow parts.

```python
import pandas as pd
import numpy as np
import time

# Simulate a large dataset operation
data = np.random.rand(10_000_000, 10)
df_small = pd.DataFrame(data)

start_time = time.time()
result_small = df_small[df_small[0] > 0.5].groupby(1).mean()
print(f"Pandas execution time: {time.time() - start_time:.2f}s")

# If this is too slow, then consider Dask for the bottleneck.
# Don't start with Dask if Pandas is fast enough.
```

### 2. Explicit Client Management

Always create an explicit `dask.distributed.Client` at the top of your script or notebook. Use it as a context manager to ensure proper shutdown and pass it explicitly to any Dask-aware libraries. Avoid relying on implicit global state.

**❌ BAD:** Implicit client usage or manual `client.close()`.

```python
# Relying on implicit client or forgetting to close
from dask.distributed import Client
client = Client() # Client might not be closed on error
# ... Dask operations ...
# client.close() # Easily forgotten
```

**✅ GOOD:** Use `Client` as a context manager and pass it explicitly.

```python
from dask.distributed import Client, LocalCluster

# For local development, use LocalCluster
with LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='2GB') as cluster:
    with Client(cluster) as client:
        print(f"Dask Dashboard: {client.dashboard_link}")
        # Your Dask operations here, passing 'client' if needed
        import dask.dataframe as dd
        df = dd.read_csv("s3://bucket/data/*.csv", blocksize="64MB")
        # ...
```

### 3. Chunk Data Wisely

Choose chunk (partition) sizes that fit comfortably in worker memory (typically 100 MiB to 1 GiB) and align with your underlying storage layout (e.g., Parquet row groups). Too small chunks lead to high task overhead; too large chunks lead to memory pressure.

**❌ BAD:** Default chunking on large files or arbitrary small chunks.

```python
import dask.dataframe as dd
# Default blocksize might be too large or too small for your data/cluster
df = dd.read_csv("large_file.csv")

# Explicitly small chunks lead to massive task graphs
df = dd.read_csv("large_file.csv", blocksize="1MB")
```

**✅ GOOD:** Tune `blocksize` based on data size, worker memory, and file format.

```python
import dask.dataframe as dd
# Aim for 100MB-1GB chunks. Adjust based on your cluster's memory profile.
# For Parquet, align with row group size if possible.
df = dd.read_parquet("s3://bucket/data/*.parquet", blocksize="256MB")
print(f"Number of partitions: {df.npartitions}")
```

### 4. Persist Intermediate Results

If a Dask collection is reused across multiple steps, `persist()` it to keep its computed results in worker memory. This prevents redundant re-computation of the same task graph. Call `.compute()` only once at the very end of your pipeline.

**❌ BAD:** Re-computing the same graph multiple times.

```python
import dask.dataframe as dd
df = dd.read_parquet("data/*.parquet")
filtered_df = df[df['value'] > 0.5]

# This will re-compute 'filtered_df' twice
mean_val = filtered_df['column_A'].mean().compute()
std_val = filtered_df['column_B'].std().compute()
```

**✅ GOOD:** `persist()` intermediate results that are reused.

```python
import dask.dataframe as dd
df = dd.read_parquet("data/*.parquet")
filtered_df = df[df['value'] > 0.5].persist() # Keep in memory

# Now 'filtered_df' is computed once and reused
mean_val = filtered_df['column_A'].mean().compute()
std_val = filtered_df['column_B'].std().compute()
```

### 5. Leverage Dask-Joblib for Scikit-learn

For scikit-learn models, use the Dask-Joblib backend to distribute training, grid search, or cross-validation across your Dask cluster. Wrap your scikit-learn code in `with joblib.parallel_backend('dask'):`.

**❌ BAD:** Running scikit-learn serially or with `n_jobs=-1` on a single machine when a cluster is available.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# ... data loading ...
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, n_jobs=-1) # Only local parallelism
grid_search.fit(X, y)
```

**✅ GOOD:** Distribute scikit-learn workloads with Dask-Joblib.

```python
from dask.distributed import Client
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# ... data loading (can be Dask DataFrames/Arrays) ...

with Client() as client: # Or connect to a remote cluster
    print(f"Dask Dashboard: {client.dashboard_link}")
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1) # n_jobs=-1 is fine here

    with joblib.parallel_backend('dask'):
        grid_search.fit(X, y) # This will now use the Dask cluster
    print(f"Best parameters: {grid_search.best_params_}")
```

### 6. Avoid Mixing Eager Pandas/NumPy with Lazy Dask

Do not mix eager Pandas/NumPy objects with lazy Dask collections inside loops or frequently. This forces Dask to compute repeatedly, negating its benefits. Keep the entire workflow in Dask until the final result is needed.

**❌ BAD:** Iterating over Dask partitions and converting to Pandas.

```python
import dask.dataframe as dd
df = dd.read_parquet("data/*.parquet")
results = []
for partition in df.partitions: # This forces computation of each partition
    pandas_df = partition.compute()
    results.append(pandas_df['col'].mean())
final_result = sum(results)
```

**✅ GOOD:** Use Dask's built-in methods or `map_partitions`/`map_blocks`.

```python
import dask.dataframe as dd
df = dd.read_parquet("data/*.parquet")

# Use Dask's native operations
mean_col = df['col'].mean().compute()

# Or map a function over partitions if custom logic is needed
def process_partition(part):
    return part['col'].mean()

mean_col_mapped = df.map_partitions(process_partition, meta=('col', 'float64')).compute()
```

### 7. Use Efficient Binary Formats

Prefer binary data formats like Parquet or Zarr over CSV or JSON. These formats are column-oriented, support schema evolution, and enable efficient filtering and projection, reducing I/O and serialization overhead.

**❌ BAD:** Reading large CSV files.

```python
import dask.dataframe as dd
# CSV is slow to read, especially with many columns or complex types
df = dd.read_csv("s3://bucket/large_data.csv")
```

**✅ GOOD:** Use Parquet or Zarr.

```python
import dask.dataframe as dd
# Parquet is highly optimized for Dask DataFrames
df = dd.read_parquet("s3://bucket/large_data.parquet")

# Zarr is excellent for Dask Arrays
import dask.array as da
arr = da.from_zarr("s3://bucket/large_array.zarr")
```

## Advanced Considerations

### 8. Monitor with the Dask Dashboard

The Dask dashboard (`client.dashboard_link`) is your most valuable tool for debugging and optimizing Dask workloads. Use it to identify bottlenecks, visualize task graphs, spot memory issues, and understand worker activity.

**✅ GOOD:** Always have the dashboard open when developing Dask code.

```python
from dask.distributed import Client, LocalCluster
with LocalCluster() as cluster:
    with Client(cluster) as client:
        print(f"Dask Dashboard: {client.dashboard_link}")
        # Run your Dask code and observe the dashboard
        import dask.array as da
        x = da.random.random((10000, 10000), chunks=(1000, 1000))
        y = x.mean(axis=1)
        y.compute() # Watch the dashboard during computation
```

### 9. Manage Graph Size

Very large task graphs (millions of tasks) can overwhelm the scheduler. If your graph becomes too large, consider:
*   **Increasing chunk size:** Reduces the number of partitions/tasks.
*   **Fusing operations:** Combine multiple small operations into a single `map_partitions`/`map_blocks` function.
*   **Breaking up computation:** For extremely large workloads, compute and save intermediate results in stages.

**❌ BAD:** Chaining many fine-grained operations on tiny chunks.

```python
# This creates a very deep graph with many small tasks
df = dd.read_parquet("data/*.parquet", blocksize="1MB") # Too small chunks
for col in df.columns:
    df[col] = df[col] * 2 + 1 # Many small operations
df.compute()
```

**✅ GOOD:** Use `map_partitions` for complex, per-partition logic.

```python
def complex_transform(partition):
    # Perform multiple operations within a single Pandas function call
    for col in partition.columns:
        partition[col] = partition[col] * 2 + 1
    return partition

df = dd.read_parquet("data/*.parquet", blocksize="256MB") # Appropriate chunk size
df_transformed = df.map_partitions(complex_transform, meta=df).compute()
```

### 10. Type Hints

Use type hints consistently in your Dask-related functions, especially for `map_partitions` and `map_blocks` callbacks. This improves readability, maintainability, and enables static analysis tools to catch errors early.

**✅ GOOD:** Type-hinting functions passed to Dask.

```python
import pandas as pd
import dask.dataframe as dd

def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Applies a transformation to a Pandas DataFrame chunk."""
    return df[df['value'] > 0.5] * 2

ddf = dd.read_parquet("data/*.parquet")
result_ddf = ddf.map_partitions(process_chunk, meta=ddf)
```

### 11. Testing Approaches

When testing Dask collections, use `dask.array.utils.assert_eq` for Dask Arrays and `dask.dataframe.utils.assert_eq` for Dask DataFrames. These functions compare Dask collections against their NumPy/Pandas equivalents, ensuring correctness of lazy computations.

**✅ GOOD:** Using Dask's assertion utilities.

```python
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq

def test_array_sum():
    np_arr = np.array([1, 2, 3, 4, 5])
    da_arr = da.from_array(np_arr, chunks=2)
    assert_eq(np_arr.sum(), da_arr.sum())

import pandas as pd
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq

def test_dataframe_mean():
    pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    dd_df = dd.from_pandas(pd_df, npartitions=2)
    assert_eq(pd_df.mean(), dd_df.mean())
```