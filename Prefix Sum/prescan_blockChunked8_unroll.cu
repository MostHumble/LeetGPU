#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_THREADS       128
#define ELEMENTS_PER_THREAD 8
#define ELEMS_PER_BLOCK     (BLOCK_THREADS * ELEMENTS_PER_THREAD)

__global__ void prescan_kernel(float *g_out, const float *g_in, float *g_blockSums, int n) {

    extern __shared__ float shared_mem[];

    float *buffer = shared_mem;

    int tid = threadIdx.x;
    int block_base = blockIdx.x * ELEMS_PER_BLOCK;

    float thread_data[ELEMENTS_PER_THREAD];
    
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_base + tid * ELEMENTS_PER_THREAD + i;
        thread_data[i] = (idx < n) ? g_in[idx] : 0.0f;
    }

#pragma unroll
    for (int i = 1; i < ELEMENTS_PER_THREAD; i++) {
        thread_data[i] += thread_data[i - 1];
    }

    buffer[tid] = thread_data[ELEMENTS_PER_THREAD - 1];

    __syncthreads();

    int offset = 1;
    for (int d = BLOCK_THREADS >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            buffer[bi] += buffer[ai];
        }
        __syncthreads();
        offset *= 2;
    }
    if (tid == 0) {
        g_blockSums[blockIdx.x] = buffer[BLOCK_THREADS - 1];
    }

    if (tid == 0) {
        buffer[BLOCK_THREADS - 1] = 0.0f;
    }
    __syncthreads();

    for (int d = 1; d < BLOCK_THREADS; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = buffer[ai];
            buffer[ai] = buffer[bi];
            buffer[bi] += t;
        }
        __syncthreads();
    }

    __syncthreads();

    float thread_offset = buffer[tid];

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_base + tid * ELEMENTS_PER_THREAD + i;
        if (idx < n) {
            g_out[idx] = thread_data[i] + thread_offset;
        }
    }
}

__global__ void add_block_sums(float *g_data, const float *g_blockSums, int n) {

    int block_base = blockIdx.x * ELEMS_PER_BLOCK;
    int tid = threadIdx.x;

    if (blockIdx.x == 0) return;

    float add_value = g_blockSums[blockIdx.x - 1];

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_base + tid * ELEMENTS_PER_THREAD + i;
        if (idx < n)
            g_data[idx] += add_value;
    }
}

// Ideas have mainly being picked from here:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
void solve(const float* input, float* output, int N) {

    float *d_input, *d_output, *d_blockSums;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    int elemsPerBlock = ELEMS_PER_BLOCK;
    int numBlocks = (N + elemsPerBlock - 1) / elemsPerBlock;

    cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_mem_size = BLOCK_THREADS * sizeof(float);

    prescan_kernel<<<numBlocks, BLOCK_THREADS, shared_mem_size>>>(d_output, d_input, d_blockSums, N);

    if (numBlocks > 1) {
        float *h_blockSums = (float*)malloc(numBlocks * sizeof(float));
        cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 1; i < numBlocks; i++) {
            h_blockSums[i] += h_blockSums[i - 1];
        }

        cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(float), cudaMemcpyHostToDevice);
        free(h_blockSums);

        add_block_sums<<<numBlocks, BLOCK_THREADS>>>(d_output, d_blockSums, N);
    }

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
}
