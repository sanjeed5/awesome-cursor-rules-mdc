---
description: Definitive guidelines for writing high-quality, performant, and maintainable ROS 2 code, leveraging modern C++ and Python best practices, `ament` tooling, and efficient architectural patterns like node composition and multithreaded executors.
globs: **/*
---
# ros Best Practices

This guide outlines the essential practices for developing robust, performant, and maintainable ROS 2 applications. Adherence to these rules ensures consistency, leverages ROS 2's advanced features, and aligns with the project's quality standards.

## 1. Code Quality and Static Analysis (The `ament` Way)

*   **Mandate `ament_lint_auto`**: Always integrate `ament_lint_auto` and `ament_lint_common` into your package build process. This is non-negotiable for catching style violations and common bugs early.
    *   ❌ BAD: Relying on manual checks or pre-commit hooks that can be bypassed.
    *   ✅ GOOD: Automatic enforcement during every build.

    ```cmake
    # CMakeLists.txt
    # ...
    if(BUILD_TESTING)
      find_package(ament_lint_auto REQUIRED)
      ament_lint_auto_find_test_dependencies()
    endif()
    # ...
    ```

    ```xml
    <!-- package.xml -->
    <package format="2">
      <!-- ... -->
      <test_depend>ament_lint_auto</test_depend>
      <test_depend>ament_lint_common</test_depend>
      <!-- ... -->
    </package>
    ```

*   **C++ Style Enforcement**: Use `ament_cpplint` and `ament_cppcheck`. Follow Google C++ Style Guide and modern C++17/20 features. Prioritize RAII and explicit smart pointer ownership.
    *   ❌ BAD: Raw pointers for resource management; ignoring `cpplint` warnings.
    *   ✅ GOOD: `std::unique_ptr`, `std::shared_ptr` for ownership.

    ```cpp
    // ❌ BAD: Manual memory management, potential leak
    MyObject* obj = new MyObject();
    // ...
    delete obj; // Easy to forget or double-delete

    // ✅ GOOD: RAII with smart pointers
    std::unique_ptr<MyObject> obj = std::make_unique<MyObject>();
    // Resource automatically managed
    ```

*   **Python Style Enforcement**: Adhere strictly to REP-8, which extends PEP 8. Use `black` for formatting and `flake8` for linting. Ensure proper naming, imports, and docstrings.
    *   ❌ BAD: Inconsistent formatting, missing docstrings, `camelCase` for variables.
    *   ✅ GOOD: `snake_case` for variables/functions, clear docstrings, `black`-formatted code.

    ```python
    # ❌ BAD
    def calculateVelocity(pos1, pos2, time):
        # ...
        return vel

    # ✅ GOOD
    def calculate_velocity(position1: float, position2: float, time_delta: float) -> float:
        """Calculates average velocity."""
        # ...
        return velocity
    ```

*   **Copyright and Licensing**: Use `ament_copyright` to ensure all source files have correct copyright and license headers.
    *   Run `ament_copyright --add-missing "Your Name" apache2` to automate this.

## 2. Code Organization and Structure

*   **Python Package Layout (Catkin)**: Adopt the recommended `src/<package_name>/__init__.py` structure. This ensures proper Python package discovery and avoids namespace collisions.
    *   ❌ BAD: Scattering Python files directly in the package root or relying on `roslib.load_manifest`.
    *   ✅ GOOD: Centralized, importable Python code.

    ```
    my_package/
    ├── CMakeLists.txt
    ├── package.xml
    ├── setup.py
    └── src/
        └── my_package/
            ├── __init__.py
            └── my_module.py
    ```

    ```python
    # setup.py for the above structure
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
        packages=['my_package'],
        package_dir={'': 'src'},
    )
    setup(**d)
    ```

*   **Node/Script Installation**: Use `catkin_install_python` for ROS nodes and scripts. Install to `${CATKIN_PACKAGE_BIN_DESTINATION}` to make them accessible via `rosrun` without polluting the global `PATH`.
    *   ❌ BAD: Installing scripts directly to `/usr/bin` via `setup.py` `scripts` argument.
    *   ✅ GOOD: Package-isolated executables.

    ```cmake
    # CMakeLists.txt
    # ...
    catkin_install_python(PROGRAMS
      nodes/my_node.py
      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    # ...
    ```

## 3. Performance Considerations

*   **Embrace Node Composition**: Always use node composition for performance-critical applications, especially those with large messages (images, point clouds) or resource-constrained environments. This enables intra-process communication (IPC) for zero-copy data transfer.
    *   ❌ BAD: Running every node in its own process, leading to high inter-process communication overhead.
    *   ✅ GOOD: Grouping related nodes into a single composed process.

    ```cpp
    // Example: Composing nodes in C++ (simplified)
    #include "rclcpp/rclcpp.hpp"
    #include "my_package/my_camera_node.hpp" // Component
    #include "my_package/my_processor_node.hpp" // Component

    int main(int argc, char * argv[]) {
      rclcpp::init(argc, argv);
      rclcpp::executors::SingleThreadedExecutor exec; // Or MultiThreadedExecutor

      auto camera_node = std::make_shared<my_package::MyCameraNode>();
      auto processor_node = std::make_shared<my_package::MyProcessorNode>();

      exec.add_node(camera_node);
      exec.add_node(processor_node); // IPC if topics match

      exec.spin();
      rclcpp::shutdown();
      return 0;
    }
    ```

*   **Choose the Right Executor**: For complex applications with multiple nodes or long-running callbacks, use `rclcpp::executors::MultiThreadedExecutor`. For simple, single-purpose nodes, `SingleThreadedExecutor` is sufficient.
    *   ❌ BAD: Using `SingleThreadedExecutor` for a node with many subscriptions or services that could block.
    *   ✅ GOOD: `MultiThreadedExecutor` for concurrent callback processing.

    ```cpp
    // ✅ GOOD: MultiThreadedExecutor for responsive systems
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4); // 4 threads
    exec.add_node(my_complex_node);
    exec.spin();
    ```

## 4. Common Pitfalls and Gotchas

*   **Thread Safety Analysis (C++)**: Enable Clang's Thread Safety Analysis (`-Wthread-safety`) for multithreaded C++ code. Annotate mutex-protected data with `RCPPUTILS_TSA_GUARDED_BY` and functions with `RCPPUTILS_TSA_REQUIRES`. Use `libcxx` for `std::mutex` annotations.
    *   ❌ BAD: Unprotected shared data access, leading to data races and deadlocks.
    *   ✅ GOOD: Compiler-assisted detection of threading issues.

    ```cmake
    # CMakeLists.txt
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      add_compile_options(-Wthread-safety)
      # add_compile_options(-Wthread-safety-negative) # For negative capability analysis
    endif()
    ```

    ```cpp
    // ✅ GOOD: Thread-safe data access with annotations
    #include <rcpputils/thread_safety_annotations.hpp>
    #include <mutex>

    class MyThreadSafeClass {
    public:
      void increment() RCPPUTILS_TSA_REQUIRES(mutex_) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_ RCPPUTILS_TSA_GUARDED_BY(mutex_)++;
      }
    private:
      mutable std::mutex mutex_;
      int data_ RCPPUTILS_TSA_GUARDED_BY(mutex_) = 0;
    };
    ```

## 5. Testing Approaches

*   **Unit Tests**: Every package must include unit tests. Use `gtest` for C++ and `pytest` for Python. Ensure high code coverage.
    *   Integrate tests into your `CMakeLists.txt` using `ament_add_gtest` or `ament_add_pytest`.
    *   ❌ BAD: Untested code, relying solely on integration tests.
    *   ✅ GOOD: Small, focused unit tests for individual components.

    ```cmake
    # CMakeLists.txt for C++ gtest
    if(BUILD_TESTING)
      find_package(ament_cmake_gtest REQUIRED)
      ament_add_gtest(my_cpp_test test/test_my_module.cpp)
      target_link_libraries(my_cpp_test PRIVATE my_package_library)
    endif()
    ```

*   **Continuous Integration (CI)**: Implement CI pipelines (GitHub Actions, Azure Pipelines) to automatically build, lint, and test your code on every push/pull request. This is crucial for maintaining code quality and catching regressions.
    *   Ensure CI runs `colcon test --packages-select <your_package> --event-handlers console_direct+`.

*   **Documentation**: Generate documentation (Doxygen for C++, Sphinx for Python) and keep it up-to-date. Clear documentation is vital for maintainability and onboarding.
    *   Integrate documentation generation into your build or CI process.

This comprehensive guide ensures your ROS 2 projects are built on a solid foundation of quality, performance, and maintainability. Adhere to these principles to contribute to a robust and reliable robotics ecosystem.