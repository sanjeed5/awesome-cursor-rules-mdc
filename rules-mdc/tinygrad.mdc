---
description: This rule file provides comprehensive best practices for developing with tinygrad, covering code organization, performance, testing, and security.  It aims to improve code quality, maintainability, and prevent common pitfalls when working with tinygrad.
globs: *.py
---
- **Introduction**
  This document outlines best practices for developing and maintaining code within the tinygrad library and projects that use tinygrad. Adhering to these guidelines will ensure code readability, maintainability, and optimal performance.  It draws from established software engineering principles, AI/ML coding standards, and the specific nuances of the tinygrad library.

- **General Principles**
  - **Readability is Paramount:**  Code should be easy to understand, even for those unfamiliar with the specific problem domain.  Use clear, descriptive variable and function names. Comment liberally, explaining the *why* more than the *what*.
  - **Simplicity First:** Favor simple, straightforward solutions over complex, clever ones.  Optimize for readability and maintainability before raw performance. Premature optimization is the root of all evil.
  - **Test-Driven Development (TDD):** Write tests before writing code. This clarifies requirements and ensures that the code behaves as expected. Tests should be automated and run frequently.
  - **Version Control:** All code must be managed with Git. Use meaningful commit messages. Branch frequently for new features and bug fixes.  Pull requests should be reviewed by at least one other developer.
  - **Continuous Integration/Continuous Deployment (CI/CD):** Automate the build, test, and deployment process.  Use a CI/CD system like GitHub Actions or GitLab CI.
  - **Documentation:** Document all public APIs.  Include examples of how to use the code. Keep documentation up-to-date.
  - **SOLID Principles:** Attempt to adhere to the SOLID design principles as appropriate for Machine Learning.

- **Code Organization and Structure**
  - **Directory Structure:** A suggested directory structure for tinygrad projects is:
    
    project_root/
    ├── tinygrad/
    │   ├── __init__.py
    │   ├── tensor.py      # Core tensor implementation
    │   ├── ops.py         # Basic operations (add, multiply, etc.)
    │   ├── device.py      # Abstraction for different hardware devices (CPU, GPU)
    │   ├── nn/            # Neural network modules
    │   │   ├── __init__.py
    │   │   ├── layers.py
    │   │   └── optim.py
    │   ├── mlops.py       # Matrix library operations
    │   ├──codegen/         # Code generation related files
    │   │   ├── __init__.py
    │   │   ├── linearizer.py
    │   │   └── opencl.py
    │   └── ...
    ├── examples/       # Example usage of tinygrad
    ├── tests/          # Unit and integration tests
    ├── README.md
    ├── setup.py
    └── requirements.txt
    
  - **File Naming Conventions:**
    - Python files: Use lowercase with underscores (e.g., `tensor.py`, `nn_layers.py`).
    - Classes: Use PascalCase (e.g., `Tensor`, `NeuralNetworkLayer`).
    - Functions and variables: Use lowercase with underscores (e.g., `add`, `learning_rate`).
    - Constants: Use UPPER_CASE with underscores (e.g., `DEFAULT_DEVICE`, `EPSILON`).
  - **Module Organization:**
    - Group related functionality into modules. For example, all neural network layers should be in the `nn` module.
    - Use `__init__.py` files to define the public API of each module.
    - Avoid circular dependencies between modules.
  - **Component Architecture:**
    - Design components with clear interfaces and responsibilities.
    - Favor composition over inheritance.
    - Use dependency injection to decouple components.
  - **Code Splitting:**
    - Break down large files into smaller, more manageable ones.
    - Split code based on functionality or responsibility.
    - Use lazy loading for modules that are not immediately needed.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - **Factory Pattern:** Use factories to create instances of tensors on different devices.
    - **Strategy Pattern:** Use the strategy pattern to implement different optimization algorithms.
    - **Observer Pattern:** Implement an observer pattern for monitoring training progress.
  - **Recommended Approaches:**
    - When implementing new operators, consider their performance implications on different hardware backends.  Optimize for the common case.
    - Use NumPy-like syntax where possible to improve code readability for those familiar with NumPy.
  - **Anti-patterns and Code Smells:**
    - **God Classes:** Avoid creating classes that do too much. Split them into smaller, more focused classes.
    - **Long Methods:** Break down long methods into smaller, more manageable ones.
    - **Duplicated Code:** Extract duplicated code into reusable functions or classes.
    - **Magic Numbers:** Use named constants instead of hardcoding numerical values.
    - **Excessive Nesting:** Reduce nesting by using helper functions or early returns.
  - **State Management:**
    - Avoid global state. Pass state explicitly to functions and classes.
    - Use immutable data structures where possible.
    - Consider using a state management library for complex applications.
  - **Error Handling:**
    - Use exceptions to handle errors.
    - Provide informative error messages.
    - Catch exceptions at the appropriate level and handle them gracefully.
    - Avoid swallowing exceptions.

- **Performance Considerations**
  - **Optimization Techniques:**
    - **Operator Fusion:** Fuse multiple operations into a single kernel to reduce memory transfers.
    - **Loop Unrolling:** Unroll loops to improve instruction-level parallelism.
    - **Vectorization:** Use SIMD instructions to perform operations on multiple data elements in parallel.
    - **Code generation:** Use code generation to tailor the implementation to the specific hardware backend. Utilize libraries like LLVM.
  - **Memory Management:**
    - Minimize memory allocations and deallocations.
    - Use memory pools to reuse memory.
    - Be aware of memory alignment requirements for different hardware backends.
    - Optimize data layout for cache efficiency.
  - **Lazy Loading:**
    - Only load data when it is needed.
    - Use generators to process large datasets in chunks.

- **Security Best Practices**
  - **Input Validation:**
    - Validate all inputs to prevent malicious data from corrupting the system.
    - Check for valid data types, ranges, and formats.
    - Sanitize inputs to remove potentially harmful characters.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use strong authentication and authorization mechanisms to protect data access.

- **Testing Approaches**
  - **Unit Testing:**
    - Test individual components in isolation.
    - Use mocking and stubbing to isolate dependencies.
    - Write tests for all public APIs.
    - Aim for high code coverage.
  - **Integration Testing:**
    - Test the interactions between different components.
    - Test the integration with external systems.
  - **End-to-End Testing:**
    - Test the entire application from end to end.
    - Simulate real-world user scenarios.
  - **Test Organization:**
    - Organize tests into a directory structure that mirrors the code structure.
    - Use meaningful test names.
    - Keep tests independent of each other.
  - **Mocking and Stubbing:**
    - Use mocking to replace dependencies with controlled substitutes.
    - Use stubbing to provide predefined responses to method calls.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - Neglecting to release allocated memory.
    - Using incorrect data types.
    - Incorrectly handling exceptions.
    - Premature optimization.
  - **Edge Cases:**
    - Handling empty tensors.
    - Handling tensors with different shapes.
    - Handling numerical instability.
  - **Version-Specific Issues:**
    - Be aware of breaking changes between different versions of tinygrad.
    - Consult the release notes for details.
  - **Compatibility Concerns:**
    - Be aware of compatibility issues between tinygrad and other libraries.
    - Test your code with different versions of dependencies.
  - **Debugging Strategies:**
    - Use a debugger to step through the code and inspect variables.
    - Use logging to track the execution flow.
    - Use assertions to check for unexpected conditions.

- **Tooling and Environment**
  - **Development Tools:**
    - PyCharm, VS Code, or other IDE.
    - Python debugger (pdb).
    - Profiler (e.g., cProfile).
    - Memory analyzer (e.g., memory_profiler).
  - **Build Configuration:**
    - Use a build system like setuptools.
    - Define dependencies in `requirements.txt`.
  - **Linting and Formatting:**
    - Use a linter like pylint or flake8.
    - Use a code formatter like black or autopep8.
    - Configure your IDE to automatically lint and format code.
  - **Deployment:**
    - Package your application into a Docker container.
    - Use a deployment platform like Kubernetes or AWS ECS.
  - **CI/CD Integration:**
    - Integrate your build, test, and deployment process into a CI/CD pipeline.
    - Use a CI/CD system like GitHub Actions or GitLab CI.

- **Additional Best Practices for Tinygrad:**
    - **Leverage the low-level nature:**  Understand the underlying operations and memory management principles of tinygrad to write efficient code.  Don't shy away from customizing operations when necessary.
    - **Contribute Back:** If you find a bug or implement a useful feature, consider contributing back to the tinygrad project.  Help the community grow.
    - **Stay Updated:** Tinygrad is under active development.  Stay up-to-date with the latest releases and best practices.
    - **Understand the Device Abstraction Layer:**  Tinygrad's strength is its ability to target different hardware backends.  Familiarize yourself with the device abstraction layer to write portable code.

By following these best practices, you can write high-quality, maintainable, and performant code with tinygrad.