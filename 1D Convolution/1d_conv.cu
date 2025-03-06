#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input_size + kernel_size){
        for (int j = 0; j < kernel_size; j += 1){
            output[tid] += input[tid+j] * kernel[j];
            }
    }
}

void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    float *d_input, *d_kernel, *d_output;
    int output_size = input_size - kernel_size + 1;

    // Allocate device memory
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
