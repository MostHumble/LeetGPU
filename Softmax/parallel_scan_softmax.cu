#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>

// max reduction kernel
__global__ void find_max_kernel(const float* input, float* block_max, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements handled by this thread
    sdata[tid] = (i < N) ? input[i] : -FLT_MAX;
    __syncthreads();

    // Reduction in shared memory - proper tree-based reduction
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block
    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

// sum reduction kernel
__global__ void compute_sum_kernel(const float* exp_values, float* block_sums, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements handled by this thread
    sdata[tid] = (i < N) ? exp_values[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory - proper tree-based reduction
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Compute exp(x_i - max) kernel
__global__ void compute_exp_kernel(const float* input, float* output, float global_max, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = expf(input[tid] - global_max);
    }
}

// reduction kernel for finding the maximum across all blocks
__global__ void reduce_max_kernel(float* block_max, float* result, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    float maxVal = -FLT_MAX;
    unsigned int i = tid;
    
    // Grid-stride loop to handle arbitrary sized arrays
    while (i < N) {
        maxVal = fmaxf(maxVal, block_max[i]);
        i += gridSize;
    }
    
    sdata[tid] = maxVal;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// reduction kernel for summing across all blocks
__global__ void reduce_sum_kernel(float* block_sums, float* result, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    unsigned int i = tid;
    
    // Grid-stride loop to handle arbitrary sized arrays
    while (i < N) {
        sum += block_sums[i];
        i += gridSize;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// Normalization kernel
__global__ void normalize_kernel(float* output, float total_sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] /= total_sum;
    }
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output, *d_block_max, *d_block_sums, *d_result;
    float h_global_max, h_total_sum;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for block maximums
    cudaMalloc(&d_block_max, blocksPerGrid * sizeof(float));

    // First pass: find the maximum value across all elements
    find_max_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_block_max, N);

    // Reduce to find the global maximum
    reduce_max_kernel<<<1, 256, 256 * sizeof(float)>>>(
        d_block_max, d_result, blocksPerGrid);

    // Copy the global maximum back to the host
    cudaMemcpy(&h_global_max, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Second pass: compute exp(x_i - max) for each element
    compute_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, h_global_max, N);

    // Allocate memory for block sums
    cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));

    // Third pass: compute sum of exp values
    compute_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_output, d_block_sums, N);

    // Reduce to find the total sum
    reduce_sum_kernel<<<1, 256, 256 * sizeof(float)>>>(
        d_block_sums, d_result, blocksPerGrid);
    
    // Copy the total sum back to the host
    cudaMemcpy(&h_total_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Fourth pass: normalize by dividing by the total sum
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_output, h_total_sum, N);

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_max);
    cudaFree(d_block_sums);
    cudaFree(d_result);
}
