#include "solve.h"
#include <cuda_runtime.h>

__global__ void dotProductKernel(const float *A, const float *B, float *result, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // grid stride : https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += A[i] * B[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


void solve(const float* A, const float* B, float* result, int N) {

    float *d_A, *d_B, *d_result;
    size_t size = N * sizeof(float);

    // turns out cudaMalloc expects null double pointer 
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_result, sizeof(float));

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMemset(d_result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // cap in case blocksPerGrid exceed the maximum number of blocks per grid (x-dimension)
    // https://forums.developer.nvidia.com/t/maximum-block-per-grid/246841/5
    if (blocksPerGrid > 1024) {
        blocksPerGrid = 1024;
    }

    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    dotProductKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_result, N);
    
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}
