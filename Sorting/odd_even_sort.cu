#include "solve.h"
#include <cuda_runtime.h>

__global__ void oddEvenTranspositionStep(float* data, int N, int phase) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int idx = i * 2 + phase;
    
    if (idx + 1 < N) {
        float a = data[idx];
        float b = data[idx + 1];
        
        if (a > b) {
            data[idx] = b;
            data[idx + 1] = a;
        }
    }
}

void solve(float* data, int N) {
    float *d_data;

    cudaMalloc((void**)&d_data, N * sizeof(float));
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N/2 + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < N; iter++) {
        // Odd phase (phase = 0): compare indices (0,1), (2,3), ...
        oddEvenTranspositionStep<<<blocks, threadsPerBlock>>>(d_data, N, 0);
        cudaDeviceSynchronize();

        // Even phase (phase = 1): compare indices (1,2), (3,4), ...
        oddEvenTranspositionStep<<<blocks, threadsPerBlock>>>(d_data, N, 1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
