#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load block_max values into shared memory
    sdata[tid] = (i < N) ? block_max[i] : -FLT_MAX;

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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load block_sums values into shared memory
    sdata[tid] = (i < N) ? block_sums[i] : 0.0f;
    
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

// Scale matrix by a constant factor kernel
__global__ void scale_matrix_kernel(float* matrix, float scale_factor, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        matrix[tid] *= scale_factor;
    }
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && column < cols) {
        output[column * rows + row] = input[row * cols + column];
    }
}

// Function to apply softmax to each row of a matrix
void apply_softmax_rows(float* d_matrix, float* d_output, int rows, int cols) {
    float *d_block_max, *d_block_sums, *d_result;
    float h_global_max, h_total_sum;
    
    // Calculate thread block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate device memory
    cudaMalloc(&d_block_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));
    
    // Process each row
    for (int row = 0; row < rows; row++) {
        float* d_row = d_matrix + row * cols;
        float* d_out_row = d_output + row * cols;
        
        // Find maximum in this row
        find_max_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_row, d_block_max, cols);
        
        // Reduce to find the global maximum for this row
        reduce_max_kernel<<<1, 256, 256 * sizeof(float)>>>(
            d_block_max, d_result, blocksPerGrid);
        
        // Copy the global maximum back to the host
        cudaMemcpy(&h_global_max, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute exp(x_i - max) for each element in this row
        compute_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_row, d_out_row, h_global_max, cols);
        
        // Compute sum of exp values for this row
        compute_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_out_row, d_block_sums, cols);
        
        // Reduce to find the total sum
        reduce_sum_kernel<<<1, 256, 256 * sizeof(float)>>>(
            d_block_sums, d_result, blocksPerGrid);
        
        // Copy the total sum back to the host
        cudaMemcpy(&h_total_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Normalize by dividing by the total sum
        normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_out_row, h_total_sum, cols);
    }
    
    // Free temporary device memory
    cudaFree(d_block_max);
    cudaFree(d_block_sums);
    cudaFree(d_result);
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // Step 1: Transpose K
    float *d_KT;
    cudaMalloc(&d_KT, N * d * sizeof(float));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (d + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(K, d_KT, N, d);
    cudaDeviceSynchronize();
    
    // Step 2: Compute Q * K^T (matrix multiplication)
    float *d_QK_T;
    cudaMalloc(&d_QK_T, M * N * sizeof(float));
    
    blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (M + threadsPerBlock.y - 1) / threadsPerBlock.y;
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(Q, d_KT, d_QK_T, M, d, N);
    cudaDeviceSynchronize();
    
    // Free K transpose as it's no longer needed
    cudaFree(d_KT);
    
    // Step 3: Scale QK^T by 1/sqrt(d)
    float scale_factor = 1.0f / sqrtf((float)d);
    int total_elements = M * N;
    int scale_threads = 256;
    int scale_blocks = (total_elements + scale_threads - 1) / scale_threads;
    
    scale_matrix_kernel<<<scale_blocks, scale_threads>>>(d_QK_T, scale_factor, total_elements);
    cudaDeviceSynchronize();
    
    // Step 4: Apply softmax row-wise to the scaled QK^T matrix
    float *d_softmax;
    cudaMalloc(&d_softmax, M * N * sizeof(float));
    
    apply_softmax_rows(d_QK_T, d_softmax, M, N);
    cudaDeviceSynchronize();
    
    // Free QK^T matrix as it's no longer needed
    cudaFree(d_QK_T);
    
    // Step 5: Compute softmax(QK^T/sqrt(d)) * V
    blocksPerGrid.x = (d + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (M + threadsPerBlock.y - 1) / threadsPerBlock.y;
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_softmax, V, output, M, N, d);
    cudaDeviceSynchronize();
    
    // Free softmax result
    cudaFree(d_softmax);
}
