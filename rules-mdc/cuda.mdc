---
description: Definitive guidelines for writing high-performance, maintainable, and error-resistant CUDA C++ code, focusing on modern practices, memory management, kernel optimization, and robust error handling.
globs: **/*
---
# CUDA Best Practices

This guide outlines the essential practices for developing efficient and maintainable CUDA C++ applications. Adhere to these rules to maximize GPU throughput, reduce debugging time, and ensure code quality across projects.

## 1. Code Organization and Structure

### 1.1 Robust Error Handling
**Always wrap CUDA API calls in an error-checking macro.** This prevents silent failures and provides immediate, actionable debugging information.

❌ BAD:
```cpp
cudaMalloc(&d_data, size);
// ... potentially crash later without knowing why
```

✅ GOOD:
```cpp
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_data, size));
kernel<<<grid, block>>>(d_data);
CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
```

### 1.2 Host-Device Separation
Clearly separate host (CPU) orchestration logic from device (GPU) computation.

❌ BAD:
```cpp
// host_code.cu
void processData() {
    cudaMalloc(&d_data, size); // Mixed host/device
    kernel<<<...>>>(d_data);
}
```

✅ GOOD:
```cpp
// host_manager.cpp
void allocateAndLaunch(float* h_in, int N) {
    float *d_in;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    myKernel_kernel<<<N/256 + 1, 256>>>(d_in, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_in));
}

// device_kernels.cu
__global__ void myKernel_kernel(float* d_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { d_in[idx] *= 2.0f; }
}
```

### 1.3 Consistent Naming Conventions
Use clear suffixes and prefixes to distinguish CUDA components.
*   `_kernel`: For `__global__` functions.
*   `_device`: For `__device__` functions.
*   `smem_`: For `__shared__` variables.
*   `k` prefix: For `__constant__` memory or compile-time constants.

❌ BAD:
```cpp
void compute(float* data); // Ambiguous
__shared__ int temp;
```

✅ GOOD:
```cpp
__global__ void myCompute_kernel(float* d_data);
__device__ float calculate_device(float val);
extern __shared__ float smem_scratch[];
__constant__ const int kTileSize = 32;
```

### 1.4 Modularity
Decompose large kernels into smaller, focused `__device__` functions.

❌ BAD:
```cpp
__global__ void monolithicKernel(float* d_data, int N) { /* Do everything here */ }
```

✅ GOOD:
```cpp
__device__ float processElement_device(float val) { return val * val + 1.0f; }

__global__ void modularKernel(float* d_in, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < N) {
        d_in[global_idx] = processElement_device(d_in[global_idx]);
    }
}
```

## 2. Common Patterns and Anti-patterns

### 2.1 Modern Memory Management (RAII)
Use RAII wrappers for device memory and streams to prevent leaks.

❌ BAD:
```cpp
float* d_data; cudaMalloc(&d_data, size); // ... cudaFree(d_data); // Easy to forget
```

✅ GOOD:
```cpp
// Use a robust library or std::unique_ptr with custom deleter in production.
template<typename T> struct CudaDeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    CudaDeviceBuffer(size_t n) : count(n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~CudaDeviceBuffer() { if (ptr) CUDA_CHECK(cudaFree(ptr)); }
    T* get() { return ptr; }
};
// Usage: CudaDeviceBuffer<float> d_data(N); // Auto-freed on scope exit
```

### 2.2 Aligned Memory Allocation
For 2D arrays, use `cudaMallocPitch` for proper alignment and coalescing.

❌ BAD:
```cpp
cudaMalloc(&d_matrix, width * height * sizeof(float)); // Misaligned rows possible
```

✅ GOOD:
```cpp
float* d_matrix; size_t pitch; // Stored in bytes
CUDA_CHECK(cudaMallocPitch((void**)&d_matrix, &pitch, width * sizeof(float), height));
// Access in kernel: d_matrix[row * (pitch / sizeof(float)) + col];
```

### 2.3 Overlap Data Transfer and Computation
Utilize `cudaMemcpyAsync` with multiple streams to hide data transfer latency.

❌ BAD:
```cpp
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice); // Synchronous
kernel<<<...>>>(d_data);
```

✅ GOOD:
```cpp
cudaStream_t stream1, stream2;
CUDA_CHECK(cudaStreamCreate(&stream1));
// Segment 1: Transfer H->D, Compute, Transfer D->H
CUDA_CHECK(cudaMemcpyAsync(d_data_seg1, h_data_seg1, size_seg1, cudaMemcpyHostToDevice, stream1));
kernel_seg1<<<...>>>(d_data_seg1, stream1);
CUDA_CHECK(cudaMemcpyAsync(h_result_seg1, d_result_seg1, size_seg1, cudaMemcpyDeviceToHost, stream1));
// ... (similar for stream2) ...
CUDA_CHECK(cudaStreamSynchronize(stream1));
CUDA_CHECK(cudaStreamDestroy(stream1));
```

### 2.4 Minimize Warp Divergence
Refactor conditional logic to reduce divergent branches within a warp. Use lookup tables or predicate execution.

❌ BAD:
```cpp
if (idx % 2 == 0) { d_data[idx] *= 2; } else { d_data[idx] += 1; }
```

✅ GOOD:
```cpp
int multiplier = (idx % 2 == 0) ? 2 : 1;
int adder = (idx % 2 == 0) ? 0 : 1;
d_data[idx] = d_data[idx] * multiplier + adder;
```

### 2.5 Use `__restrict__` and `const` Qualifiers
Enable compiler optimizations by explicitly declaring non-aliased pointers and read-only data.

❌ BAD:
```cpp
__global__ void add(float* a, float* b, float* c, int N) { /* ... */ }
```

✅ GOOD:
```cpp
__global__ void add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { c[idx] = a[idx] + b[idx]; }
}
```

### 2.6 Data Layout: Structures-of-Arrays (SoA)
Organize data as SoA for better memory coalescing, especially for frequently accessed fields.

❌ BAD:
```cpp
struct Particle { float x, y, z; }; Particle* particles = new Particle[N];
```

✅ GOOD:
```cpp
struct ParticlesSoA { float *x, *y, *z; }; // Allocate each array separately
```

### 2.7 Leverage CUDA-X Libraries
For common operations (BLAS, FFT, DNN), use highly optimized CUDA-X libraries (cuBLAS, cuFFT, cuDNN, cuML).

❌ BAD:
```cpp
__global__ void myMatrixMul_kernel(...) { /* complex GEMM logic */ }
```

✅ GOOD:
```cpp
cublasHandle_t handle; CUDA_CHECK(cublasCreate(&handle));
// ... setup matrices ...
CUDA_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
CUDA_CHECK(cublasDestroy(handle));
```

### 2.8 Embrace CUDA Tile (Toolkit 13.1+)
For tile-based tensor operations, use the new CUDA Tile programming model and cuTile Python API. This simplifies complex tiling logic and targets specialized hardware.

✅ GOOD:
```cpp
// Refer to NVIDIA documentation for specific CUDA Tile API usage.
// This abstracts away explicit shared memory management and synchronization for tiling.
// E.g., using cuTile Python API to express high-level tensor operations.
```

## 3. Performance Considerations

### 3.1 Memory Coalescing
Ensure global memory accesses by threads within a warp are contiguous and aligned.

❌ BAD:
```cpp
d_matrix[threadIdx.x * N + blockIdx.x]; // Strided access
```

✅ GOOD:
```cpp
d_matrix[blockIdx.x * blockDim.x + threadIdx.x]; // Coalesced access
```

### 3.2 Shared Memory Optimization
Use shared memory to cache frequently accessed global memory data. Be mindful of bank conflicts.

❌ BAD:
```cpp
__shared__ float smem_data[256]; float val = smem_data[threadIdx.x * 2]; // Bank conflicts
```

✅ GOOD:
```cpp
__shared__ float smem_data[256]; float val = smem_data[threadIdx.x]; // No bank conflicts
```

### 3.3 Constant Memory
Use `__constant__` memory for read-only data uniform across all threads. It has a fast, cached access path.

✅ GOOD:
```cpp
__constant__ float kCoefficients[10];
__global__ void process_kernel(float* d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { d_data[idx] *= kCoefficients[0]; } // Fast access
}
```

## 4. Common Pitfalls and Gotchas

### 4.1 Unchecked API Calls
As covered in 1.1, neglecting error checks leads to hard-to-debug issues.

### 4.2 Excessive `__syncthreads()`
Overuse of `__syncthreads()` can severely limit occupancy and performance. Only synchronize when absolutely necessary.

### 4.3 Forgetting `cuda