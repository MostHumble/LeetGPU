#include "solve.h"
#include <cuda_runtime.h>

__global__ void conv2d(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols, 
                       int output_rows, int output_cols) {

    unsigned int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.0f;

    if (out_row < output_rows && out_col < output_cols) {

        for (int m = 0; m < kernel_rows; ++m) {

            for (int n = 0; n < kernel_cols; ++n) {

                int input_row = out_row + m;
                int input_col = out_col + n;

                float input_value = input[input_row * input_cols + input_col];
                float kernel_value = kernel[m * kernel_cols + n];

                sum += input_value * kernel_value;
            }
         }

        output[out_row * output_cols + out_col] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
	
    float* d_input;
    float* d_kernel;
    float* d_output;

    int output_cols = input_cols - (kernel_cols - 1);
    int output_rows = input_rows - (kernel_rows - 1);

    unsigned int size_input = input_cols * input_rows;
    unsigned int size_kernel = kernel_cols * kernel_rows;
    unsigned int size_output = output_cols * output_rows;

     
    cudaMalloc(&d_input, size_input * sizeof(float));
    cudaMalloc(&d_kernel, size_kernel * sizeof(float));
    cudaMalloc(&d_output, size_output * sizeof(float));
 
    cudaMemcpy(d_input, input, size_input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, size_kernel * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);


     conv2d<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output,
                                           input_rows, input_cols,
                                           kernel_rows, kernel_cols,
                                           output_rows, output_cols);

    cudaMemcpy(output, d_output, size_output * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
