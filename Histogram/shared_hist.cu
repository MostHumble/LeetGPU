#include <cuda_runtime.h>
#include <stdio.h>

__global__ void histogram_shared_kernel(const int* input, int* output, int num_elements, int num_bins) {
    extern __shared__ int shared_hist[];

    int tid = threadIdx.x;
    
    for (int i = tid; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }

    __syncthreads();

    int id = blockIdx.x * blockDim.x + tid;

    if (id < num_elements) {
        int bin = input[id];

        if (bin >= 0 && bin < num_bins) { 
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&output[i], shared_hist[i]);
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {

    int ThreadsPerBlock = 256;
    int numBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    int* d_input;
    int* d_output;

    float size_input = N * sizeof(int);
    float size_output = num_bins * sizeof(int);

    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);

    cudaMemset(d_output, 0, size_output);

    size_t sharedMemSize = num_bins * sizeof(int);

    histogram_shared_kernel<<<numBlocks, ThreadsPerBlock, sharedMemSize>>>(d_input, d_output, N, num_bins);

    cudaMemcpy(histogram, d_output, size_output, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_input);
    
}
