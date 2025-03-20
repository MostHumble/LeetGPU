#include "solve.h"
#include <cuda_runtime.h>

__global__ void flock_kernel(const float* agents, float* agents_next, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x_i = agents[4 * idx + 0];
    float y_i = agents[4 * idx + 1];
    float vx_i = agents[4 * idx + 2];
    float vy_i = agents[4 * idx + 3];

    float vx_avg = 0.0f;
    float vy_avg = 0.0f;
    int neighbor_count = 0;

    // find neighbors within radius
    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float x_j = agents[4 * j + 0];
        float y_j = agents[4 * j + 1];

        // squared distance:
        float dx = x_i - x_j;
        float dy = y_i - y_j;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < 25.0f) {  // r_squared = 25.0 (r = 5.0)
            float vx_j = agents[4 * j + 2];
            float vy_j = agents[4 * j + 3];

            vx_avg += vx_j;
            vy_avg += vy_j;
            neighbor_count++;
        }
    }

    float vx_new = vx_i;
    float vy_new = vy_i;
  
    // update 
    if (neighbor_count > 0) {
        vx_avg /= neighbor_count;
        vy_avg /= neighbor_count;

        vx_new = vx_i + 0.05f * (vx_avg - vx_i);
        vy_new = vy_i + 0.05f * (vy_avg - vy_i);
    }

    float x_new = x_i + vx_new;
    float y_new = y_i + vy_new;

    agents_next[4 * idx + 0] = x_new;
    agents_next[4 * idx + 1] = y_new;
    agents_next[4 * idx + 2] = vx_new;
    agents_next[4 * idx + 3] = vy_new;
}

void solve(const float* agents, float* agents_next, int N) {
    unsigned int size_agents = N * 4 * sizeof(float);

    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    float *d_agents, *d_agents_next;

    cudaMalloc((void**)&d_agents, size_agents);
    cudaMalloc((void**)&d_agents_next, size_agents);

    cudaMemcpy(d_agents, agents, size_agents, cudaMemcpyHostToDevice);

    flock_kernel<<<num_blocks, threads_per_block>>>(d_agents, d_agents_next, N);

    cudaDeviceSynchronize();

    cudaMemcpy(agents_next, d_agents_next, size_agents, cudaMemcpyDeviceToHost);

    cudaFree(d_agents);
    cudaFree(d_agents_next);
}
