---
description: Definitive guidelines for writing high-performance, functionally pure, and maintainable JAX code, focusing on common pitfalls and optimal patterns for accelerators.
globs: **/*.py
---
# jax Best Practices

JAX is the backbone of our AI/ML and numerical computing projects. Adhere to these principles for high-performance, reproducible, and maintainable JAX code.

## 1. Functional Purity: The Absolute Core

JAX transformations (`jit`, `grad`, `vmap`, `pmap`) operate exclusively on **functionally pure** code. This means functions must be free of side-effects: all inputs explicit, all results returned.

*   **Avoid mutable global state:** JAX captures global values at first `jit` compilation, leading to stale values.
    ❌ BAD:
    ```python
    g = 0
    def impure_uses_globals(x):
        return x + g # `g` is captured at first jit
    # ... later g = 10, but jit(impure_uses_globals) still uses g=0
    ```
    ✅ GOOD: Pass all state explicitly.
    ```python
    def pure_uses_globals(x, g_val):
        return x + g_val
    # ... pass g_val=0, then g_val=10
    ```

*   **No in-place array mutation:** JAX arrays are immutable. Use the `.at[]` syntax for functional updates.
    ❌ BAD:
    ```python
    import jax.numpy as jnp
    arr = jnp.zeros((3,3))
    arr[1, :] = 1.0 # TypeError!
    ```
    ✅ GOOD:
    ```python
    import jax.numpy as jnp
    arr = jnp.zeros((3,3))
    updated_arr = arr.at[1, :].set(1.0) # Returns a new array
    ```

*   **Avoid Python iterators in `jit`ted code:** Iterators introduce state.
    ❌ BAD:
    ```python
    from jax import jit
    def sum_iterator(it):
        total = 0
        for x in it: # Python loop with iterator
            total += x
        return total
    # jit(sum_iterator)(iter(range(10))) # Will fail or give unexpected results
    ```
    ✅ GOOD: Use JAX control flow primitives.
    ```python
    from jax import lax
    import jax.numpy as jnp
    def sum_array(arr):
        # Use lax.scan or lax.fori_loop for JAX-compatible loops
        return lax.fori_loop(0, arr.shape[0], lambda i, x: x + arr[i], 0)
    # jit(sum_array)(jnp.arange(10))
    ```

## 2. Numerical Type Discipline

Prioritize `float32` for performance on accelerators. Avoid implicit `float64` promotion.

*   **Explicit `dtype` for constants:**
    ❌ BAD: Implicitly typed Python floats can lead to `float64` promotion.
    ```python
    import jax.numpy as jnp
    x = jnp.ones(5, dtype=jnp.float32)
    y = x * 2.0 # 2.0 is a Python float, can cause unwanted promotion
    ```
    ✅ GOOD: Use 0-D `jnp.array` with explicit `dtype` or `jnp.float32()`.
    ```python
    import jax.numpy as jnp
    x = jnp.ones(5, dtype=jnp.float32)
    y = x * jnp.array(2.0, dtype=jnp.float32)
    # Or more concisely:
    z = x * jnp.float32(2.0)
    ```

## 3. Performance Considerations: Stable Compilation

Prevent costly recompilations and leverage XLA effectively.

*   **Static shapes:** Keep input shapes static or pass them via `static_argnums` to `jit`. Dynamic shapes force recompilation.
    ❌ BAD:
    ```python
    from jax import jit
    def dynamic_shape_func(x):
        return x.sum()
    # jit(dynamic_shape_func)(jnp.ones(5))
    # jit(dynamic_shape_func)(jnp.ones(10)) # Recompiles!
    ```
    ✅ GOOD:
    ```python
    from jax import jit
    @jit
    def static_shape_func(x):
        return x.sum()
    # Call with consistent shapes
    # static_shape_func(jnp.ones(5))
    # static_shape_func(jnp.ones(5)) # Uses cached compilation
    ```
    If shapes *must* vary, consider `static_argnums` for non-array arguments that determine shape.

*   **JAX control flow primitives:** Always use `lax.scan`, `lax.while_loop`, `lax.cond` inside `jit`ted functions. Python control flow breaks XLA compilation.
    ❌ BAD:
    ```python
    from jax import jit
    @jit
    def python_loop(x, n):
        for _ in range(n): # Python loop inside jit
            x = x * 2
        return x
    ```
    ✅ GOOD:
    ```python
    from jax import jit, lax
    @jit
    def jax_loop(x, n):
        # lax.fori_loop(start, stop, body_fn, init_val)
        return lax.fori_loop(0, n, lambda i, val: val * 2, x)
    ```

*   **Vectorization with `vmap`:** For data-parallel batching, `vmap` is your tool.
    ```python
    from jax import vmap
    import jax.numpy as jnp
    def elementwise_op(x, y):
        return x + y # Operates on scalars or single elements
    batched_op = vmap(elementwise_op)
    # batched_op(jnp.array([1,2,3]), jnp.array([4,5,6])) # Applies elementwise
    ```

## 4. Code Organization and Randomness

*   **Standard Imports:**
    ```python
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax.lax as lax
    ```

*   **Randomness Management:** Use `jax.random` and explicitly split PRNGKeys. Never reuse a key.
    ❌ BAD:
    ```python
    import jax.random as jr
    key = jr.PRNGKey(0)
    val1 = jr.normal(key, (5,)) # Key used
    val2 = jr.normal(key, (5,)) # Key reused, not independent!
    ```
    ✅ GOOD:
    ```python
    import jax.random as jr
    key = jr.PRNGKey(0)
    key, subkey1 = jr.split(key)
    val1 = jr.normal(subkey1, (5,))
    key, subkey2 = jr.split(key)
    val2 = jr.normal(subkey2, (5,)) # Independent random values
    ```

## 5. Type Hints

Use standard Python type hints, especially `jax.Array` for JAX arrays. This improves readability and enables static analysis.

```python
import jax.numpy as jnp
from jax import Array, jit

@jit
def add_arrays(a: Array, b: Array) -> Array:
    """Adds two JAX arrays."""
    return a + b

# Example usage:
# result = add_arrays(jnp.ones(5), jnp.ones(5))
```

## 6. Testing Approaches

*   **`pytest` is standard:** Use `pytest` for unit and integration tests.
*   **Gradient checking:** For custom operations or complex functions, use `jax.test_util.check_grads` to verify gradients.

```python
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

def my_complex_func(x):
    return jnp.sin(x) * jnp.exp(x)

# Test gradients numerically vs. analytically
# check_grads(my_complex_func, (jnp.array(1.0),), order=1)
```