#include "solve.h"
#include <cuda_runtime.h>

__global__ void conv3d_kernel_shared(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* output,
    int input_depth, int input_rows, int input_cols,
    int kernel_depth, int kernel_rows, int kernel_cols,
    int output_depth, int output_rows, int output_cols
) {
    extern __shared__ float shared_input[];

    int od = blockIdx.z * blockDim.z + threadIdx.z;
    int orow = blockIdx.y * blockDim.y + threadIdx.y;
    int ocol = blockIdx.x * blockDim.x + threadIdx.x;

    int block_output_depth = blockDim.z;
    int block_output_rows = blockDim.y;
    int block_output_cols = blockDim.x;

    int shared_depth = block_output_depth + kernel_depth - 1;
    int shared_rows  = block_output_rows  + kernel_rows  - 1;
    int shared_cols  = block_output_cols  + kernel_cols  - 1;

    int tid = threadIdx.z * (shared_rows * shared_cols) + threadIdx.y * shared_cols + threadIdx.x;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    int shared_size = shared_depth * shared_rows * shared_cols;

    int input_start_d = blockIdx.z * blockDim.z;
    int input_start_r = blockIdx.y * blockDim.y;
    int input_start_c = blockIdx.x * blockDim.x;

    // load input tile into shared memory
    for (int idx = tid; idx < shared_size; idx += num_threads) {
        int sd = idx / (shared_rows * shared_cols);
        int rem = idx % (shared_rows * shared_cols);
        int sr = rem / shared_cols;
        int sc = rem % shared_cols;

        // calculate input indices relative to the start of the input tile
        int id = input_start_d + sd;
        int ir = input_start_r + sr;
        int ic = input_start_c + sc;

        float val = 0.0f;
        if (id < input_depth && ir < input_rows && ic < input_cols) {
            int input_idx = id * (input_rows * input_cols) + ir * input_cols + ic;
            val = input[input_idx];
        }

        shared_input[idx] = val;
    }

    __syncthreads();

    if (od >= output_depth || orow >= output_rows || ocol >= output_cols)
        return;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {

                int sd = threadIdx.z + kd;
                int sr = threadIdx.y + kr;
                int sc = threadIdx.x + kc;

                int shared_idx =
                    sd * (shared_rows * shared_cols) +
                    sr * shared_cols +
                    sc;

                int kernel_idx =
                    kd * (kernel_rows * kernel_cols) +
                    kr * kernel_cols +
                    kc;

                sum += shared_input[shared_idx] * kernel[kernel_idx];
            }
        }
    }

    int output_idx =
        od * (output_rows * output_cols) +
        orow * output_cols +
        ocol;

    output[output_idx] = sum;
}


// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols) {

    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t input_size = input_depth * input_rows * input_cols * sizeof(float);
    size_t kernel_size = kernel_depth * kernel_rows * kernel_cols * sizeof(float);
    size_t output_size = output_depth * output_rows * output_cols * sizeof(float);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_kernel, kernel_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
        (output_cols + blockDim.x - 1) / blockDim.x,
        (output_rows + blockDim.y - 1) / blockDim.y,
        (output_depth + blockDim.z - 1) / blockDim.z
    );

    int shared_depth = blockDim.z + kernel_depth - 1;
    int shared_rows  = blockDim.y + kernel_rows - 1;
    int shared_cols  = blockDim.x + kernel_cols - 1;

    size_t shared_mem_size = shared_depth * shared_rows * shared_cols * sizeof(float);

    conv3d_kernel_shared<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_kernel, d_output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols
    );

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
