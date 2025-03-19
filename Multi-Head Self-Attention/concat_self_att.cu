#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float val = 0.0f;
        for (int i = 0; i < N; ++i) {
            val += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = val;
    }
}

__global__ void scale_kernel(float* matrix, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale;
    }
}

__global__ void row_softmax_kernel(float* input, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < cols; ++i) {
        float val = input[row * cols + i];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(input[row * cols + i] - max_val);
        input[row * cols + i] = e;
        sum += e;
    }

    for (int i = 0; i < cols; ++i) {
        input[row * cols + i] /= sum;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int d_head = d_model / h;
    dim3 threadsPerBlock(16, 16);

    // Allocate space for the final concatenated output
    float* d_concat;
    cudaMalloc((void**)&d_concat, N * d_model * sizeof(float));

    // Process each head independently
    for (int head = 0; head < h; ++head) {
        const float* Q_head = Q + head * d_head;
        const float* K_head = K + head * d_head;
        const float* V_head = V + head * d_head;

        float *d_Q, *d_K, *d_V 

        cudaMalloc((void**)&d_Q, N * d_head * sizeof(float));
        cudaMalloc((void**)&d_K, N * d_head * sizeof(float));
        cudaMalloc((void**)&d_V, N * d_head * sizeof(float));

        // Copy head slices into local Q, K, V
        for (int i = 0; i < N; ++i) {
            cudaMemcpy(d_Q + i * d_head, Q_head + i * d_model, d_head * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_K + i * d_head, K_head + i * d_model, d_head * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_V + i * d_head, V_head + i * d_model, d_head * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // Transpose K -> K^T
        float* d_KT;
        cudaMalloc((void**)&d_KT, d_head * N * sizeof(float));

        dim3 blocks_KT((d_head + 15) / 16, (N + 15) / 16);
        transpose_kernel<<<blocks_KT, threadsPerBlock>>>(d_K, d_KT, N, d_head);

        // Q * K^T
        float* d_scores;
        cudaMalloc((void**)&d_scores, N * N * sizeof(float));

        dim3 blocks_scores((N + 15) / 16, (N + 15) / 16);
        matmul_kernel<<<blocks_scores, threadsPerBlock>>>(d_Q, d_KT, d_scores, N, d_head, N);

        // Scale scores by 1 / sqrt(d_head)
        float scale = 1.0f / sqrtf((float)d_head);
        int total_scores = N * N;
        int blocks_scale = (total_scores + 255) / 256;
        scale_kernel<<<blocks_scale, 256>>>(d_scores, scale, total_scores);

        // Softmax along rows of d_scores
        int blocks_softmax = (N + 255) / 256;
        row_softmax_kernel<<<blocks_softmax, 256>>>(d_scores, N, N);

        // Multiply softmax(scores) * V
        float* d_output_head;
        cudaMalloc((void**)&d_output_head, N * d_head * sizeof(float));

        dim3 blocks_out((d_head + 15) / 16, (N + 15) / 16);
        matmul_kernel<<<blocks_out, threadsPerBlock>>>(d_scores, d_V, d_output_head, N, N, d_head);

        // Write the head's output into the correct section of d_concat
        for (int i = 0; i < N; ++i) {
            cudaMemcpy(d_concat + i * d_model + head * d_head,
                       d_output_head + i * d_head,
                       d_head * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }

        // Clean up head allocations
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_KT);
        cudaFree(d_scores);
        cudaFree(d_output_head);
    }

    // Copy final concatenated output to result
    cudaMemcpy(output, d_concat, N * d_model * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_concat);
}
