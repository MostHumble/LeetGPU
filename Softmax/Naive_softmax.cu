__global__ void softmax(float* input, float* output, int N) {
    // Shared memory for block-wise reduction
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    int tid = threadIdx.x;
    
    // local max
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    // global max
    atomicMax((int*)&max_val, __float_as_int(local_max)); 
    __syncthreads();

    //  Compute exponentials and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    // denominator of softmax (global softmax sum)
    atomicAdd(&sum_exp, local_sum); 
    __syncthreads();

    // stable softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}