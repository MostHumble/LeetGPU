#include "solve.h" 
#include <cuda_runtime.h> 

// direct array reversal
__global__ void reverse_array(float* input, int N) { 
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only threads with valid indices participate
    if (id < N / 2) {
        // Swap elements from opposite ends of the array
        float temp = input[id];
        input[id] = input[N - 1 - id];
        input[N - 1 - id] = temp;
    }
} 
 
// input is device pointer 
void solve(float* input, int N) { 
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    // Direct reversal with one kernel
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
  
    cudaDeviceSynchronize(); 
}
