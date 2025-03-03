#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row index

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x]; // Transpose operation
    }
}
