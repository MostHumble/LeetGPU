#include "solve.h"
#include <cuda_runtime.h>

__global__ void atomic_histogram(int* input, int* output, const int num_bins, const int N){

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N ){
        int bin = input[tid];
        if (bin >= 0 && bin < num_bins){
            atomicAdd(&output[bin], 1);
    }
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

    atomic_histogram<<<numBlocks, ThreadsPerBlock>>>(d_input, d_output, num_bins, N);

    cudaMemcpy(histogram, d_output, size_output, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_input);
    
}
