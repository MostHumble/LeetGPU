#include "solve.h" 
#include <cuda_runtime.h> 
#include <stdio.h>

// Kernel to reverse elements within each block using shared memory
__global__ void reverse_block(float* input, int N) { 
    extern __shared__ float block_reverse[];

    unsigned int tid = threadIdx.x; 
    unsigned int id = blockIdx.x * blockDim.x + tid;

    // Load into shared memory with bounds check
    block_reverse[tid] = (id < N) ? input[id] : 0.0f; 

    __syncthreads();

    unsigned int block_size = min(blockDim.x, N - blockIdx.x * blockDim.x);

    // Only swap valid threads within the actual block size
    if (tid < block_size / 2) {
        int opposite = block_size - 1 - tid;
        float temp = block_reverse[tid]; 
        block_reverse[tid] = block_reverse[opposite]; 
        block_reverse[opposite] = temp; 
    } 

    __syncthreads();

    // Write back to global memory
    if (id < N) {
        input[id] = block_reverse[tid]; 
    } 
} 

// Kernel to reverse block order (after reversing inside each block)
__global__ void merge_blocks(float* input, float* output, int N, int blockSize) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        int block_num = id / blockSize;
        int element_offset = id % blockSize;

        int num_blocks = (N + blockSize - 1) / blockSize;
        int lastBlockSize = N - (num_blocks - 1) * blockSize;

        int output_index;

        if (block_num == num_blocks - 1) {
            // Element is in the last original block
            output_index = element_offset;
        } else {
            int reversed_block_num = num_blocks - 1 - block_num;
            output_index = lastBlockSize + (reversed_block_num - 1) * blockSize + element_offset;
        }

        // Bounds check
        if (output_index < N) {
            output[output_index] = input[id];
        }
    }
}

void solve(float* input, int N) { 
    int threadsPerBlock = (N < 256) ? N : 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
  
    // Reverse elements inside each block
    reverse_block<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, N);
    cudaDeviceSynchronize();

    float* temp_output;
    cudaMalloc(&temp_output, N * sizeof(float));
    
    // Reverse block order
    merge_blocks<<<blocksPerGrid, threadsPerBlock>>>(input, temp_output, N, threadsPerBlock);
    cudaDeviceSynchronize();

    cudaMemcpy(input, temp_output, N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaFree(temp_output);
    
    cudaDeviceSynchronize(); 
}
