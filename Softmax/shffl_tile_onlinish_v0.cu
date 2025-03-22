// inspired by:
// https://github.com/Maharshi-Pandya/cudacodes/blob/master/softmax/kernels/blocktiling_5.cu

#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 32/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void max_kernel(const float* input, int N, float* max_partials) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    float block_max = blockReduceMax(thread_max);

    if (threadIdx.x == 0) {
        max_partials[blockIdx.x] = block_max;
    }
}

__global__ void reduce_max_kernel(float* partials, int num_partials, float* result) {
    unsigned int tid = threadIdx.x;
    float thread_max = -FLT_MAX;

    for (int i = tid; i < num_partials; i += blockDim.x) {
        thread_max = fmaxf(thread_max, partials[i]);
    }

    float block_max = blockReduceMax(thread_max);

    if (threadIdx.x == 0) {
        *result = block_max;
    }
}

__global__ void sum_kernel(const float* input, float* output, int N, const float* d_max, float* sum_partials) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = *d_max;
    float thread_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        thread_sum += exp_val;
    }

    thread_sum = blockReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        sum_partials[blockIdx.x] = thread_sum;
    }
}

__global__ void reduce_sum_kernel(float* partials, int num_partials, float* result) {
    unsigned int tid = threadIdx.x;
    float thread_sum = 0.0f;

    for (int i = tid; i < num_partials; i += blockDim.x) {
        thread_sum += partials[i];
    }

    thread_sum = blockReduceSum(thread_sum);

    if (threadIdx.x == 0) {
        *result = thread_sum;
    }
}

__global__ void divide_kernel(float* output, int N, const float* d_sum) {
    float sum_val = *d_sum;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        output[i] /= sum_val;
    }
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;
    float *d_max_partials, *d_max;
    float *d_sum_partials, *d_sum;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_max_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N, d_max_partials);
    reduce_max_kernel<<<1, threadsPerBlock>>>(d_max_partials, blocksPerGrid, d_max);

    cudaMalloc((void**)&d_sum_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_max, d_sum_partials);
    reduce_sum_kernel<<<1, threadsPerBlock>>>(d_sum_partials, blocksPerGrid, d_sum);

    divide_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, N, d_sum);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max_partials);
    cudaFree(d_max);
    cudaFree(d_sum_partials);
    cudaFree(d_sum);
}
