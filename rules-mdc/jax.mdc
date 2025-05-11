---
description: This rule provides best practices and coding standards for the JAX library, emphasizing functional programming, JIT compilation, automatic differentiation, and immutable data structures. It also covers performance considerations, common pitfalls, and tooling recommendations.
globs: *.py
---
- **Functional Programming**: JAX emphasizes a functional programming style. Ensure functions are pure (no side effects, no reliance on global variables) for consistent JIT compilation and optimization.

- **JIT Compilation**: Use `@jax.jit` to decorate functions for performance optimization, especially those called multiple times with the same input shapes. Understand the implications of tracing and static arguments.

- **Automatic Differentiation**: Leverage `jax.grad()` for efficient gradient calculations and higher-order derivatives.  Be mindful of how JAX handles gradients with control flow and other transformations.

- **Vectorization with `vmap`**: Utilize `jax.vmap()` for automatic vectorization instead of explicit loops. This improves performance by mapping operations across array dimensions.

- **Immutable Data Structures**: JAX arrays are immutable. Use `.at[].set()` for updates. Understand that these operations are out-of-place, returning a new array. In-place updates *may* occur under jit-compilation if the original value isn't reused, but rely on the functional style.

- **Random Number Generation**: Use `jax.random` for random number generation. Employ explicit PRNG state management. Always split the key using `jax.random.split()` before generating random numbers to avoid reusing the same state.

- **Control Flow**: Be aware of limitations when using Python control flow with `jax.jit`. Use `static_argnums` to specify arguments for tracing on concrete values if necessary. Consider structured control flow primitives like `lax.cond`, `lax.while_loop`, `lax.fori_loop`, and `lax.scan` for better traceability and avoiding large loop unrolling.  Understand which control flow constructs are differentiable.

- **Dynamic Shapes**: Avoid dynamic shapes within JAX transformations like `jax.jit`, `jax.vmap`, and `jax.grad`.  The shapes of output arrays must not depend on values within other arrays. Use techniques like `jnp.where` to work around the need for dynamically-sized arrays.

- **NaN Handling**: Use the NaN-checker during debugging by setting `JAX_DEBUG_NANS=True` or using `jax.config.update("jax_debug_nans", True)`.  Be aware that this adds overhead and should be disabled in production.

- **Double Precision**: Enable double-precision numbers by setting the `jax_enable_x64` configuration variable at startup (`JAX_ENABLE_X64=True` environment variable, `jax.config.update("jax_enable_x64", True)`, or `jax.config.parse_flags_with_absl()`).  Note that not all backends support 64-bit convolutions.

- **Non-array Inputs**: JAX generally expects NumPy arrays as inputs to its API functions. Avoid passing Python lists or tuples directly; convert them to arrays first.

- **Out-of-Bounds Indexing**: JAX handles out-of-bounds indexing by clamping the index to the bounds of the array for retrieval operations. Array update operations at out-of-bounds indices are skipped. Use the optional parameters of `ndarray.at` for finer-grained control.

- **Miscellaneous Divergences from NumPy**: Be aware of differences in type promotion rules, unsafe type casts, and other corner cases where JAX's behavior may differ from NumPy.

## Code Organization and Structure:

- **Directory Structure**: Consider a structure like this:
  
  project_root/
  ├── src/
  │   ├── __init__.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── model_a.py
  │   │   └── model_b.py
  │   ├── layers/
  │   │   ├── __init__.py
  │   │   ├── layer_a.py
  │   │   └── layer_b.py
  │   ├── utils/
  │   │   ├── __init__.py
  │   │   ├── data_loading.py
  │   │   └── evaluation.py
  │   ├── train.py
  │   └── predict.py
  ├── tests/
  │   ├── __init__.py
  │   ├── models/
  │   │   ├── test_model_a.py
  │   │   └── test_model_b.py
  │   ├── utils/
  │   │   ├── test_data_loading.py
  │   │   └── test_evaluation.py
  │   └── test_train.py
  ├── notebooks/
  │   ├── exploration.ipynb
  │   └── analysis.ipynb
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── models/
  │   └── saved_models/
  ├── .gitignore
  ├── README.md
  ├── requirements.txt
  └── setup.py
  

- **File Naming**: Use descriptive, lowercase names with underscores (e.g., `data_loading.py`, `model_a.py`).

- **Module Organization**: Group related functionalities into modules.  Use clear and concise module names.

- **Component Architecture**: Favor a modular architecture. Decouple components where possible. Use interfaces or abstract base classes when appropriate.

- **Code Splitting**: Split large files into smaller, more manageable modules.  Consider splitting based on functionality or abstraction level.

## Common Patterns and Anti-patterns:

- **Design Patterns**: Consider patterns like:
    - **Strategy**: For selecting different computation strategies at runtime.
    - **Factory**: For creating JAX models or layers.
    - **Observer**: For monitoring training progress.

- **Recommended Approaches**: 
    - Using `jax.tree_util` for working with nested data structures (pytrees).
    - Caching intermediate results with `lru_cache` (with caution, as it introduces state).
    - Using `jax.experimental.optimizers` for defining optimization schedules.

- **Anti-patterns**: 
    - Mutable global state within JIT-compiled functions.
    - Excessive use of `static_argnums` (leads to recompilation).
    - Ignoring the immutability of JAX arrays.
    - Using Python loops instead of vectorized operations (`vmap`).
    - Unnecessary host-device data transfers.
    - Reusing PRNG keys without splitting.

- **State Management**: 
    - Avoid global mutable state. Pass state explicitly as function arguments.
    - Consider using immutable data structures for state.
    - If mutable state is absolutely necessary (e.g., in some RL settings), use JAX's stateful computation tools carefully (e.g., `jax.lax.scan` with a carry).

- **Error Handling**: 
    - Use `try...except` blocks for handling potential errors during data loading or preprocessing.
    - Employ assertions to check for invalid input or unexpected conditions.
    - Utilize logging to track errors and debug issues.
    - The `JAX_DEBUG_NANS` and `JAX_DEBUG_NANS=True` flag is critical when debugging `NaN`.

## Performance Considerations:

- **Optimization Techniques**: 
    - JIT compilation (`jax.jit`).
    - Vectorization (`jax.vmap`).
    - Parallelization (`jax.pmap`).
    - Fusion of operations.
    - Reducing host-device data transfers.
    - Choose appropriate data types (e.g., `float32` instead of `float64` if sufficient).
    - Use `jax.numpy` functions instead of NumPy functions where possible.
    - Avoid scalar operations inside JIT-compiled loops; use array operations.

- **Memory Management**: 
    - Minimize the creation of large intermediate arrays.
    - Reuse arrays where possible (using `.at[].set()` for in-place updates if safe to do so).
    - Be aware of memory fragmentation.
    - If you are dealing with datasets larger than memory, explore data streaming/sharding approaches.

- **Rendering Optimization**:  (Relevant if using JAX for visualization, e.g., with OpenGL or similar libraries)
    - Batch rendering operations.
    - Optimize data transfer to the rendering device (e.g., GPU).
    - Consider using specialized rendering libraries that are compatible with JAX arrays.

- **Bundle Size Optimization**: (For JAX-based applications deployed in web environments)
    - Tree-shake unused code.
    - Minify JavaScript bundles.
    - Use code splitting to load only necessary modules.
    - Compress assets (e.g., images, data files).

- **Lazy Loading**: 
    - Load data or models only when needed.
    - Use iterators or generators for large datasets.
    - Implement a loading indicator to provide feedback to the user.

## Security Best Practices:

- **Common Vulnerabilities**: 
    - Input data poisoning.
    - Model inversion attacks.
    - Adversarial examples.
    - Side-channel attacks (less relevant in typical JAX applications, but important in cryptographic applications).

- **Input Validation**: 
    - Validate input data types and ranges.
    - Sanitize input data to prevent injection attacks.
    - Use schema validation libraries (e.g., `cerberus`, `jsonschema`).

- **Authentication and Authorization**:  (If building APIs or web services with JAX)
    - Implement secure authentication mechanisms (e.g., OAuth 2.0, JWT).
    - Use role-based access control (RBAC) to restrict access to sensitive resources.
    - Protect API endpoints with appropriate authentication and authorization checks.

- **Data Protection**: 
    - Encrypt sensitive data at rest and in transit.
    - Use secure storage mechanisms for models and data.
    - Implement data masking or anonymization techniques to protect personally identifiable information (PII).

- **Secure API Communication**: 
    - Use HTTPS for all API communication.
    - Implement proper rate limiting to prevent denial-of-service attacks.
    - Use secure coding practices to prevent common web vulnerabilities (e.g., cross-site scripting, SQL injection).

## Testing Approaches:

- **Unit Testing**: 
    - Test individual functions or classes in isolation.
    - Use `pytest` or `unittest` for writing unit tests.
    - Mock external dependencies to isolate the code under test.
    - Test different input scenarios and edge cases.
    - Use `jax.random.PRNGKey` with a fixed seed for reproducible tests.

- **Integration Testing**: 
    - Test the interaction between different components or modules.
    - Verify that data flows correctly between components.
    - Test the integration with external services or databases.

- **End-to-End Testing**: 
    - Test the entire application flow from start to finish.
    - Simulate real user interactions.
    - Verify that the application meets the specified requirements.
    - Use tools like Selenium or Cypress for end-to-end testing.

- **Test Organization**: 
    - Organize tests into separate directories or modules.
    - Use descriptive test names.
    - Follow a consistent testing style.

- **Mocking and Stubbing**: 
    - Use `unittest.mock` or `pytest-mock` for mocking external dependencies.
    - Create stubs for complex or time-consuming operations.
    - Mock JAX functions (e.g., `jax.random.normal`) to control the output of random number generators.

## Common Pitfalls and Gotchas:

- **Frequent Mistakes**: 
    - Mutable global state in JIT-compiled functions.
    - Incorrect use of `static_argnums`.
    - Ignoring JAX array immutability.
    - Using Python loops instead of `vmap`.
    - Reusing PRNG keys without splitting.

- **Edge Cases**: 
    - Handling `NaN` and `Inf` values.
    - Dealing with numerical instability.
    - Working with sparse data.
    - Optimizing for different hardware backends (CPU, GPU, TPU).

- **Version-Specific Issues**: 
    - Be aware of breaking changes between JAX versions.
    - Consult the JAX documentation for the latest updates and bug fixes.

- **Compatibility Concerns**: 
    - Ensure compatibility between JAX and other libraries (e.g., TensorFlow, PyTorch).
    - Be aware of potential conflicts between JAX and NumPy versions.

- **Debugging Strategies**: 
    - Use the JAX debugger (`jax.debug.print`).
    - Set `JAX_DEBUG_NANS=True` to detect `NaN` values.
    - Use `jax.make_jaxpr` to inspect the JAX program representation.
    - Use `jax.debug.visualize_jaxpr` to visualize the JAX program flow (requires graphviz).
    - Profile your code to identify performance bottlenecks.
    - Use a debugger (e.g., `pdb`) to step through your code and inspect variables.

## Tooling and Environment:

- **Recommended Development Tools**: 
    - VS Code with the Python extension.
    - Jupyter Notebook or JupyterLab for interactive development.
    - IPython for a powerful interactive shell.
    - TensorBoard for visualizing training progress.

- **Build Configuration**: 
    - Use `requirements.txt` or `setup.py` to manage dependencies.
    - Specify JAX version requirements.
    - Use a virtual environment (e.g., `venv`, `conda`) to isolate dependencies.

- **Linting and Formatting**: 
    - Use `flake8` or `pylint` for linting.
    - Use `black` or `autopep8` for formatting.
    - Configure your editor to automatically lint and format code on save.

- **Deployment Best Practices**: 
    - Package your application into a Docker container.
    - Use a cloud platform (e.g., AWS, Google Cloud, Azure) for deployment.
    - Use a deployment framework (e.g., Flask, FastAPI) for serving APIs.
    - Monitor your application for performance and errors.

- **CI/CD Integration**: 
    - Use a CI/CD platform (e.g., GitHub Actions, Travis CI, CircleCI).
    - Automate testing, linting, and formatting.
    - Automate deployment to staging and production environments.


This comprehensive guide should help developers write better JAX code, avoid common pitfalls, and build high-performance machine learning applications.