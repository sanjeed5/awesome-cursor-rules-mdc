---
description: This rule provides comprehensive best practices and coding standards for the Numba library, covering code organization, performance optimization, security, testing, and common pitfalls. It aims to help developers write efficient, maintainable, and secure Numba code.
globs: *.py
---
# Numba Best Practices and Coding Standards

This document outlines best practices and coding standards for the Numba library to help developers write efficient, maintainable, and secure code.

## Library Information:

- Name: numba
- Tags: ai, ml, gpu-computing, python, jit-compiler

## 1. Code Organization and Structure

### 1.1. Directory Structure

- **`src/`:** Contains the main source code. Subdivide into modules or packages based on functionality (e.g., `src/core`, `src/cuda`, `src/npyimpl`).
- **`tests/`:** Contains unit, integration, and end-to-end tests.  Mirror the `src/` structure to easily locate corresponding tests.
- **`examples/`:** Provides example scripts and notebooks demonstrating how to use numba features.
- **`docs/`:** Stores documentation in reStructuredText or Markdown format.
- **`benchmarks/`:** Contains benchmark scripts for performance evaluation.
- **`scripts/`:**  Utilities and helper scripts for development, testing, or deployment.
- **`.cursor/rules/`:** Store custom rules for AI-assisted code generation (e.g., this file).

Example:


my_project/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── functions.py
│   │   ├── types.py
│   ├── cuda/
│   │   ├── __init__.py
│   │   ├── kernels.py
│   │   ├── device.py
│   ├── npyimpl/
│   │   ├── __init__.py
│   │   ├── array_methods.py
│   │   ├── reductions.py
├── tests/
│   ├── __init__.py
│   ├── core/
│   │   ├── test_functions.py
│   │   ├── test_types.py
│   ├── cuda/
│   │   ├── test_kernels.py
│   │   ├── test_device.py
│   ├── npyimpl/
│   │   ├── test_array_methods.py
│   │   ├── test_reductions.py
├── examples/
│   ├── mandelbrot.py
│   ├── gaussian_blur.ipynb
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   ├── Makefile
├── benchmarks/
│   ├── bench_matmul.py
├── scripts/
│   ├── build_extensions.py
├── .cursor/
│   ├── rules/
│   │   ├── numba_rules.mdc
├── setup.py
├── README.md
├── LICENSE


### 1.2. File Naming Conventions

- Python files: Use descriptive lowercase names with underscores (e.g., `array_methods.py`, `type_inference.py`).
- Test files: Prefix with `test_` and mirror the module/file being tested (e.g., `test_array_methods.py`).
- Class names: Use PascalCase (e.g., `NumbaTypeManager`, `CUDADeviceContext`).
- Function and variable names: Use lowercase with underscores (e.g., `jit_compile`, `input_array`).
- Constants: Use uppercase with underscores (e.g., `DEFAULT_NUM_THREADS`, `MAX_ARRAY_SIZE`).

### 1.3. Module Organization

- Group related functionality into modules. For example, put all type-related code in a `types` module.
- Use `__init__.py` files to define packages and control namespace imports.
- Keep modules focused and avoid large, monolithic modules.
- Use relative imports within packages (e.g., `from . import functions`).

### 1.4. Component Architecture

- Design with clear separation of concerns (e.g., type inference, code generation, runtime execution).
- Employ interfaces or abstract base classes for loose coupling between components.
- Use dependency injection to manage component dependencies.
- Favor composition over inheritance to promote code reuse and flexibility.

### 1.5. Code Splitting

- Break down large functions into smaller, more manageable functions.
- Group related functions into classes or modules.
- Use decorators to add functionality without modifying the original code (e.g., `@jit`, `@vectorize`).
- Extract common code into reusable utility functions.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

- **Decorator Pattern:** Used extensively in Numba via the `@jit`, `@njit`, `@vectorize`, and `@cuda.jit` decorators for applying transformations and optimizations to functions.
- **Factory Pattern:** Can be used to create different types of compiled functions based on runtime conditions or configurations.
- **Strategy Pattern:** Employ different compilation strategies based on target architecture or optimization level.

### 2.2. Recommended Approaches

- **Use `@njit` whenever possible:**  This enforces no-python mode, maximizing performance.  Only fall back to `@jit` if no-python mode is not feasible.
- **Explicitly specify types:** When possible, define input and output types using Numba's type system for increased performance and type safety.
- **Utilize `prange` for parallel loops:** When using `parallel=True`, use `numba.prange` instead of `range` for loops that can be executed in parallel.
- **Minimize data transfer between CPU and GPU:** When using CUDA, keep data on the GPU as much as possible and pre-allocate output arrays.
- **Use structured array when possible:** When working with complex data, leverage the power of structured arrays to improve data access times and readability.

### 2.3. Anti-patterns

- **Relying on object mode:**  Object mode is significantly slower than no-python mode. Avoid it by ensuring your code is compatible with no-python mode.
- **Excessive use of global variables:** Global variables can hinder type inference and optimization.  Pass data as function arguments instead.
- **Unnecessary data copying:** Avoid copying data between CPU and GPU or within memory unless absolutely necessary.
- **Ignoring profiling results:**  Don't guess where performance bottlenecks are.  Use profiling tools to identify and address them.
- **Using Python data structures like dictionaries and lists directly inside `@njit` decorated code:** These data structures usually trigger object mode and hinder performance. Opt for `typed-list` and `typed-dict` when possible.

### 2.4. State Management

- Avoid mutable global state within Numba-compiled functions. Mutating global state can lead to unpredictable behavior and make debugging difficult.
- If state needs to be maintained, consider using a class with methods decorated with `@njit` to encapsulate the state and operations.
- When working with CUDA, manage device memory explicitly to avoid memory leaks and ensure efficient resource utilization.

### 2.5. Error Handling

- Use `try...except` blocks to handle potential exceptions within Numba-compiled functions.
- Propagate errors gracefully and provide informative error messages.
- Consider using Numba's error reporting mechanisms for custom error handling.
- Handle CUDA errors properly when working with GPU code.

## 3. Performance Considerations

### 3.1. Optimization Techniques

- **Just-In-Time (JIT) Compilation:** Leverage `@jit` and `@njit` decorators for automatic code optimization.
- **Vectorization:** Utilize `@vectorize` decorator for element-wise operations on arrays.
- **Parallelization:** Employ `parallel=True` and `prange` for parallel loop execution on multi-core CPUs.
- **CUDA Acceleration:** Use `@cuda.jit` decorator to compile and execute code on NVIDIA GPUs.
- **`fastmath`:** Use `fastmath=True` when strict IEEE 754 compliance is not required to enable aggressive optimizations.
- **Loop Unrolling:** Manually unroll loops to reduce loop overhead when appropriate.
- **Memory Alignment:** Ensure data is properly aligned in memory for efficient access.
- **Kernel Fusion:** Combine multiple operations into a single kernel to reduce memory access and improve performance.
- **Use Intel SVML:** If possible, install and use the Intel Short Vector Math Library (SVML) for optimized transcendental functions.

### 3.2. Memory Management

- **Minimize data transfer between CPU and GPU:** Keep data on the GPU as much as possible.
- **Pre-allocate output arrays:** Use `cuda.device_array` or `np.empty` to pre-allocate memory.
- **Use CUDA Device Arrays:** Operate directly on device arrays to avoid implicit data transfers.
- **Use `del` keyword to free unused arrays:** Use the `del` keyword to release large arrays or variables and free up memory after they're no longer needed.

### 3.3. Bundle Size Optimization

- Numba itself doesn't directly impact front-end bundle sizes (as it operates on the backend Python code).  However, minimizing dependencies and using efficient numerical algorithms can indirectly reduce the overall application size.

### 3.4. Lazy Loading

- Numba JIT compilation performs lazy loading by default. Functions are compiled only when they are first called.
- This can be beneficial for large applications where not all functions need to be compiled immediately.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

- **Code Injection:** Avoid using `eval()` or `exec()` with user-supplied input, as this can lead to arbitrary code execution.
- **Integer Overflow:** Be mindful of potential integer overflows when performing arithmetic operations.
- **Buffer Overflow:** Prevent buffer overflows by validating array dimensions and sizes before performing operations.
- **Denial of Service (DoS):** Limit resource usage (e.g., memory, CPU time) to prevent DoS attacks.

### 4.2. Input Validation

- Validate all user-supplied input before passing it to Numba-compiled functions.
- Check data types, ranges, and sizes to prevent unexpected behavior or errors.
- Sanitize input to remove potentially malicious characters or code.

### 4.3. Data Protection

- Avoid storing sensitive data in plain text. Encrypt sensitive data at rest and in transit.
- Implement access controls to restrict access to sensitive data to authorized users only.

### 4.4. Secure API Communication

- Use HTTPS for all API communication to encrypt data in transit.
- Implement authentication and authorization mechanisms to verify the identity of clients and restrict access to resources.
- Protect against Cross-Site Scripting (XSS) and Cross-Site Request Forgery (CSRF) attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

- Write unit tests for individual functions and classes to verify their correctness.
- Use the `unittest` or `pytest` frameworks for writing and running tests.
- Mock external dependencies to isolate the code being tested.
- Test different input values and edge cases to ensure robustness.

### 5.2. Integration Testing

- Write integration tests to verify the interaction between different components or modules.
- Test the integration between Numba-compiled functions and other parts of the application.
- Use realistic test data to simulate real-world scenarios.

### 5.3. End-to-End Testing

- Write end-to-end tests to verify the overall functionality of the application.
- Test the entire workflow from user input to output to ensure that all components are working correctly.
- Use automated testing tools to run tests regularly.

### 5.4. Test Organization

- Organize tests in a directory structure that mirrors the source code.
- Use descriptive test names to indicate the functionality being tested.
- Write clear and concise test cases that are easy to understand and maintain.

### 5.5. Mocking and Stubbing

- Use mocking and stubbing to replace external dependencies with controlled substitutes during testing.
- This allows you to isolate the code being tested and simulate different scenarios.
- Use libraries like `unittest.mock` or `pytest-mock` for mocking and stubbing.
- Make sure mocks respect type constraints used within the tested code.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

- **Incorrect data types:**  Numba relies on type inference, so providing incorrect data types can lead to unexpected results or errors. Explicitly specify types when possible.
- **Unsupported Python features:**  Numba does not support all Python features. Be aware of the limitations and use supported alternatives.
- **Not using no-python mode:** Forgetting to use `@njit` or `nopython=True` can result in significantly slower performance due to object mode compilation.
- **Ignoring performance profiling:**  Failing to profile your code can lead to inefficient optimizations or missed opportunities for performance improvements.
- **Over-optimizing too early:** Focus on writing correct and readable code first, then optimize only when necessary.
- **Incompatible NumPy versions:** Make sure you are using a compatible NumPy version that Numba supports.

### 6.2. Edge Cases

- **Division by zero:** Handle potential division by zero errors gracefully.
- **NaN and Inf values:** Be aware of how NaN and Inf values can propagate through calculations.
- **Large arrays:**  Be mindful of memory usage when working with large arrays, especially on GPUs.
- **Multithreading issues:** When using `parallel=True`, be aware of potential race conditions and data synchronization issues.

### 6.3. Version-Specific Issues

- Consult the Numba documentation and release notes for any version-specific issues or compatibility concerns.
- Pay attention to deprecation warnings and update your code accordingly.

### 6.4. Compatibility Concerns

- Ensure compatibility between Numba and other libraries you are using, such as NumPy, SciPy, and CUDA.
- Be aware of potential conflicts between different versions of these libraries.

### 6.5. Debugging Strategies

- Use the `print()` function or a debugger to inspect variables and execution flow.
- Check the Numba documentation and community forums for troubleshooting tips.
- Use the `numba.errors.DEBUG` environment variable to enable debug output from Numba.
- Use `numba --sysinfo` to get details about your system for inclusion in bug reports or when seeking help.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

- **IDE:** VS Code with the Python extension, PyCharm, or Jupyter Notebook.
- **Profiler:** `cProfile` or `line_profiler` for identifying performance bottlenecks.
- **Debugger:** `pdb` or an IDE-integrated debugger for stepping through code and inspecting variables.
- **Linter:** `flake8` or `pylint` for enforcing code style and identifying potential errors.
- **Formatter:** `black` or `autopep8` for automatically formatting code according to PEP 8.

### 7.2. Build Configuration

- Use `setup.py` or `pyproject.toml` for managing project dependencies and build configuration.
- Specify Numba as a dependency in your project's configuration file.
- Consider using a virtual environment (e.g., `venv` or `conda`) to isolate project dependencies.

### 7.3. Linting and Formatting

- Use `flake8` or `pylint` to enforce code style and identify potential errors.
- Configure the linter to check for Numba-specific best practices, such as using `@njit` whenever possible.
- Use `black` or `autopep8` to automatically format code according to PEP 8.

### 7.4. Deployment Best Practices

- Package your application and its dependencies into a self-contained executable or container.
- Use a deployment platform that supports Numba and its dependencies (e.g., AWS Lambda, Google Cloud Functions, Docker).
- Optimize your application for the target deployment environment.

### 7.5. CI/CD Integration

- Integrate Numba into your Continuous Integration/Continuous Delivery (CI/CD) pipeline.
- Run unit tests, integration tests, and end-to-end tests automatically on every commit.
- Use a CI/CD tool like GitHub Actions, GitLab CI, or Jenkins to automate the build, test, and deployment process.
- Perform performance testing as part of the CI/CD pipeline to detect regressions.