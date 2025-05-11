---
description: Enforces CUDA coding standards, performance optimizations, and best practices to ensure efficient and maintainable GPU-accelerated code. This rule provides guidance on code organization, memory management, error handling, and more.
globs: *.cu
---
- # CUDA Best Practices Guide
  This document outlines best practices for CUDA development, focusing on code organization, performance optimization, and common pitfalls.  It is based on NVIDIA's CUDA C++ Best Practices Guide and expands on it with more detailed recommendations.

- ## 1. Code Organization and Structure

  - ### 1.1 Directory Structure
    - Organize your CUDA project with a clear directory structure.  A common structure includes:
      
      project_root/
      ├── src/
      │   ├── kernels/      # CUDA kernel source files (.cu)
      │   ├── host/         # Host-side code (.cpp, .h)
      │   ├── common/       # Shared utility code (.h, .cpp)
      │   └── include/      # Header files
      ├── build/            # Build output directory
      ├── data/             # Input and output data
      ├── tests/           # Unit and integration tests
      └── CMakeLists.txt  # CMake build configuration
      

  - ### 1.2 File Naming Conventions
    - Use descriptive file names that clearly indicate the purpose of the file.
      - Kernel files: `kernel_name.cu` (e.g., `matrix_multiply.cu`)
      - Host files: `module_name.cpp`, `module_name.h` (e.g., `data_loader.cpp`, `data_loader.h`)
      - Common files: `utility.h`, `error_handling.cpp`

  - ### 1.3 Module Organization
    - Divide your code into logical modules based on functionality.
    - Use namespaces to avoid naming conflicts and improve code organization.
    - Encapsulate CUDA kernel launches within well-defined functions or classes.

  - ### 1.4 Component Architecture
    - Design your application with a modular component architecture to facilitate code reuse and maintainability.
    - Decouple host-side code from CUDA kernels as much as possible.
    - Use abstraction layers to hide CUDA-specific details from higher-level components.

  - ### 1.5 Code Splitting Strategies
    - Split large CUDA kernels into smaller, more manageable functions.
    - Use separate files for different kernels or related functionalities.
    - Consider using template metaprogramming to generate specialized kernels at compile time.

- ## 2. Common Patterns and Anti-patterns

  - ### 2.1 Design Patterns Specific to CUDA
    - **CUDA Stream Pattern:** Use CUDA streams to overlap data transfers and kernel execution.
    - **Memory Pooling Pattern:** Implement memory pools to reduce the overhead of frequent memory allocations and deallocations.
    - **Tiling Pattern:** Divide large data structures into smaller tiles to improve data locality and cache utilization.
    - **Reduction Pattern:** Use parallel reduction algorithms to efficiently compute aggregate values.

  - ### 2.2 Recommended Approaches for Common Tasks
    - **Error Handling:** Use the CUDA error handling API to check for errors after each CUDA function call.
    - **Memory Allocation:** Use `cudaMalloc`, `cudaFree`, and related functions for memory allocation on the device.
    - **Data Transfer:** Use `cudaMemcpy` to transfer data between host and device memory.
    - **Kernel Launch:** Use the `<<<gridDim, blockDim, sharedMem>>>` syntax to launch CUDA kernels.

  - ### 2.3 Anti-patterns and Code Smells to Avoid
    - **Synchronous Memory Transfers:** Avoid blocking memory transfers that stall the GPU.
    - **Excessive Global Memory Access:** Minimize global memory access by using shared memory and registers.
    - **Thread Divergence:** Avoid conditional branches that cause threads within a warp to execute different code paths.
    - **Uncoalesced Memory Access:** Ensure that threads access memory in a coalesced manner to maximize memory bandwidth.
    - **CPU-GPU Synchronization Bottlenecks:**  Minimize the number of synchronization points between the CPU and GPU.

  - ### 2.4 State Management Best Practices
    - Encapsulate CUDA context and device management within a dedicated class or module.
    - Avoid global state variables that can lead to unexpected behavior and concurrency issues.
    - Use RAII (Resource Acquisition Is Initialization) to ensure that CUDA resources are properly released.

  - ### 2.5 Error Handling Patterns
    - Check the return value of every CUDA function call and handle errors appropriately.
    - Use `cudaGetLastError` to retrieve the last error that occurred on the device.
    - Implement custom error handling routines for specific error conditions.
    - Log error messages with file name, line number, and a descriptive error message.

- ## 3. Performance Considerations

  - ### 3.1 Optimization Techniques
    - **Kernel Fusion:** Combine multiple kernels into a single kernel to reduce kernel launch overhead and data transfers.
    - **Loop Unrolling:** Unroll loops to improve instruction-level parallelism.
    - **Instruction Scheduling:** Optimize instruction scheduling to reduce pipeline stalls.
    - **Constant Memory Usage:** Store frequently accessed read-only data in constant memory.
    - **Texture Memory Usage:** Utilize texture memory for spatially coherent data access patterns.

  - ### 3.2 Memory Management
    - **Minimize Data Transfers:** Reduce the amount of data transferred between host and device.
    - **Asynchronous Data Transfers:** Use asynchronous memory transfers with CUDA streams to overlap computation and communication.
    - **Zero-Copy Memory:** Use zero-copy memory to directly access host memory from the GPU (use with caution due to performance implications).
    - **Pinned Memory (Page-Locked Memory):** Use pinned memory for efficient asynchronous data transfers.

  - ### 3.3 CUDA Profiler
    - Use the NVIDIA Nsight Systems and Nsight Compute profilers to identify performance bottlenecks.

- ## 4. Security Best Practices

  - ### 4.1 Common Vulnerabilities and How to Prevent Them
    - **Buffer Overflows:** Carefully validate input sizes to prevent buffer overflows in CUDA kernels.
    - **Integer Overflows:** Check for potential integer overflows in calculations involving data sizes and indices.
    - **Race Conditions:** Protect shared data with appropriate synchronization mechanisms (e.g., atomic operations, mutexes) to prevent race conditions.
    - **Injection Attacks:** Sanitize input data to prevent injection attacks that could execute arbitrary code on the GPU.

  - ### 4.2 Input Validation
    - Validate all input data received by CUDA kernels to ensure that it is within the expected range and format.
    - Check for invalid or malicious input that could lead to security vulnerabilities.

  - ### 4.3 Data Protection Strategies
    - Encrypt sensitive data stored on the GPU to protect it from unauthorized access.
    - Use secure communication channels to transfer data between host and device.

- ## 5. Testing Approaches

  - ### 5.1 Unit Testing Strategies
    - Write unit tests to verify the correctness of individual CUDA kernels and host-side functions.
    - Use a testing framework like Google Test or Catch2 to automate the testing process.
    - Mock CUDA runtime functions to isolate kernels during testing.
    - Use a separate compilation approach to test individual kernel functions.

  - ### 5.2 Integration Testing
    - Perform integration tests to verify the interaction between different components of the CUDA application.
    - Test data transfers between host and device, kernel launches, and error handling.

  - ### 5.3 Test Organization
    - Organize your tests into a logical directory structure.
    - Use descriptive test names that clearly indicate the purpose of each test.
    - Group related tests together into test suites.

  - ### 5.4 Mocking and Stubbing
    - Use mocking and stubbing techniques to isolate components during testing and simulate different scenarios.
    - Mock CUDA runtime functions to control the behavior of the GPU during testing.

- ## 6. Common Pitfalls and Gotchas

  - ### 6.1 Frequent Mistakes Developers Make
    - **Ignoring CUDA Error Codes:** Always check the return values of CUDA functions to ensure that they succeed.
    - **Incorrect Grid and Block Dimensions:** Choose appropriate grid and block dimensions for your kernels.
    - **Shared Memory Bank Conflicts:** Avoid shared memory bank conflicts to maximize memory bandwidth.
    - **Thread Divergence:** Minimize thread divergence within warps to improve performance.
    - **Uncoalesced Memory Access:** Ensure that threads access memory in a coalesced manner.

  - ### 6.2 Version-Specific Issues
    - Be aware of compatibility issues between different CUDA versions.
    - Use conditional compilation to handle version-specific code.

  - ### 6.3 Debugging Strategies
    - Use the NVIDIA Nsight Systems and Nsight Compute debuggers to debug CUDA code.
    - Insert print statements to track the execution flow and variable values.
    - Use the `cudaDeviceSynchronize` function to force the GPU to complete all operations before proceeding.

- ## 7. Tooling and Environment

  - ### 7.1 Recommended Development Tools
    - **CUDA Toolkit:** Install the latest version of the CUDA Toolkit from NVIDIA's website.
    - **NVIDIA Nsight Systems and Nsight Compute:** Use the Nsight profilers to analyze and optimize CUDA code.
    - **CMake:** Use CMake to manage the build process.
    - **Integrated Development Environment (IDE):** Use an IDE such as Visual Studio or Eclipse with CUDA support.

  - ### 7.2 Build Configuration
    - Use CMake to generate build files for your target platform.
    - Configure the CUDA compiler (nvcc) with appropriate optimization flags.

  - ### 7.3 Linting and Formatting
    - Use a linter such as clang-tidy to enforce coding standards and identify potential errors.
    - Use a code formatter such as clang-format to ensure consistent code formatting.