#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void block_reduce(float *input, float *block_sums, int N) {
    __shared__ float block_sum[BLOCK_SIZE];

    int id = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory (zero out extra threads)
    if (tid < N) {
        block_sum[id] = input[tid];
    } else {
        block_sum[id] = 0.0f;
    }
    __syncthreads();

    // Block-level reduction
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * id;
        if (index < blockDim.x) {
            block_sum[index] += block_sum[index + i];
        }
        __syncthreads();
    }

    // Store block result in global memory
    if (id == 0) {
        block_sums[blockIdx.x] = block_sum[0];
    }
}

__global__ void grid_reduce(float *block_sums, float *result, int num_blocks) {
    __shared__ float sum[BLOCK_SIZE];

    int id = threadIdx.x;

    // Load into shared memory
    if (id < num_blocks) {
        sum[id] = block_sums[id];
    } else {
        sum[id] = 0.0f;
    }
    __syncthreads();

    // Reduction within the final block
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * id;
        if (index < blockDim.x) {
            sum[index] += sum[index + i];
        }
        __syncthreads();
    }

    // Store final result
    if (id == 0) {
        *result = sum[0];
    }
}

void solve(const float* input, float* output, int N) {  
    int threads_per_block = 256;  
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;  

    // Device memory allocation
    float *d_input, *d_block_sums, *d_result;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Block-Level Reduction
    block_reduce<<<num_blocks, threads_per_block>>>(d_input, d_block_sums, N);
    
    // Step 2: Grid-Level Reduction (if needed)
    if (num_blocks > 1) {
        grid_reduce<<<1, num_blocks>>>(d_block_sums, d_result, num_blocks);
        // Copy final result back to host
        cudaMemcpy(output, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        // If only one block, result is already in d_block_sums[0]
        cudaMemcpy(output, d_block_sums, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_block_sums);
    cudaFree(d_result);
}

