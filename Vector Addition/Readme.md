# Step by step dissection:
---
```c
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}
```
## ```__global__```:
```__global__```: This is a CUDA-specific keyword indicating that the function is a kernel, a function that is executed on the GPU and launched by the host (CPU).

Others include: 
__device__: Device function (called from device or global functions).
__host__: Host function (executed on CPU).
__forceinline__: Forces inlining of a function ( meaning the function's code will be copied into the places where it's called, rather than being invoked through a regular function call.)
__launch_bounds__: Optimizes thread/block configurations for kernels.
__atomic_*: Atomic functions for safe concurrent memory access.

These can be used in combination with each other, for example:
__host__ __device__: Host-device function (can be called from both CPU and GPU).

## ```void``` 
In C++ (and CUDA), void is used as a return type for functions that do not produce a return value. This is similar to the way void is used in other programming languages.

You might ask why are we not returning the array and instead editing the inplace:

We do not return values from the CUDA kernel because. This stems from three reasons:

- Parallelism: CUDA threads execute in parallel, and it’s easier and more efficient for each thread to modify its own part of an array in memory rather than attempting to return values and synchronize results.
- Efficiency: Modifying values directly in the arrays avoids extra copying, synchronization, and merging, which would be complex and costly for large datasets.
- Memory management: Passing pointers to arrays allows you to efficiently work with GPU memory and avoid the overhead of return values or managing intermediate results across many threads.
