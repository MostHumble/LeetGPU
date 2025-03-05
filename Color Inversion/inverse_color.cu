#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCount = width * height;

    if (tid < pixelCount) {
        int pixelIndex = tid * 4;

        // Invert R, G, B components, leave A unchanged
        for (int i = 0; i < 3; ++i) {
            image[pixelIndex + i] = 255 - image[pixelIndex + i];
        }
    }
}

void solve(unsigned char* image, int width, int height) {
    unsigned char* d_image;
    int image_size = width * height * 4;

    // Allocate device memory
    cudaMalloc(&d_image, image_size * sizeof(unsigned char));

    // Copy input data from host to device
    cudaMemcpy(d_image, image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
}
