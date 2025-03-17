#include "solve.h"
#include <cuda_runtime.h>

__global__ void spmv_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;

        // Loop through each column in the row
        for (int col = 0; col < N; ++col) {
            float val = A[row * N + col];

            // Skip zeros (because the matrix is sparse)
            if (val != 0.0f) {
                sum += val * x[col];
            }
        }

        y[row] = sum;
    }
}

// A, x, y are device pointers
void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {

    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    // Launch the kernel
    spmv_kernel<<<gridSize, blockSize>>>(A, x, y, M, N);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}
