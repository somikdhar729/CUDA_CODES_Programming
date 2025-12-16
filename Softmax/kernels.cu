#include "kernels.cuh"
#include <float.h>

__global__ void softmax_naive(const float *input, float *output, int N){
    
    extern __shared__ float sdata[];
    float* max_vals = sdata;
    float* sum_vals = &sdata[blockDim.x];

    int tid = threadIdx.x;
    
    // Find max value in row
    float thread_max = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x){
        thread_max = fmaxf(thread_max, input[i]);
    }
    max_vals[tid] = thread_max;
    __syncthreads();

    // Parallel reduction to find overall max
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    float max_val = max_vals[0];
    __syncthreads();

    // Compute sum of exponentials
    float thread_sum = 0.0f;
    for(int i = tid; i < N; i += blockDim.x){
        thread_sum += expf(input[i] - max_val);
    }
    sum_vals[tid] = thread_sum;
    __syncthreads();
    // Parallel reduction to find overall sum
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sum_vals[0];
    __syncthreads();
    // Compute softmax output
    for(int i = tid; i < N; i += blockDim.x){
        output[i] = expf(input[i] - max_val) / sum_val;
    }

}


// Kernel 2
__global__ void max_kernel(const float* input, float* output, int N){
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim. x * blockIdx.x;
    extern __shared__ float sOut[];
    if(gid < N){
        sOut[tid] = input[gid];
    }
    else{
        sOut[tid] = -FLT_MAX;
    }

    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(tid < i){
            sOut[tid] = max(sOut[tid], sOut[tid + i]);
        }
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void sum_kernel(const float* input, float* output, int N){
    int tid = threadIdx. x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float sOut[];
    if(gid < N){
        sOut[tid] = input[gid];
    }
    else{
        sOut[tid] = 0.0f;
    }
    __syncthreads();
    
    for(int i = blockDim.x / 2; i > 32; i >>= 1){
        if(tid < i){
            sOut[tid] += sOut[tid + i];
        }
        __syncthreads();
    }
    if(tid < 32){
        warpReduce(sOut,tid);
    }
    if(tid == 0){
        output[blockIdx. x] = sOut[0];
    }
}

__global__ void exp_kernel(const float* input, float* output, int N, const float* max_) {
    int gid = threadIdx.x + blockDim.x * blockIdx. x;
    if(gid < N){
        output[gid] = __expf(input[gid] - max_[0]);
    }
}

__global__ void normalize_kernel(const float* input, float* output, int N, const float* exp_sum) {
    int gid = threadIdx.x + blockDim.x * blockIdx. x;
    if(gid < N){
        output[gid] = input[gid] / exp_sum[0];
    }
}

void softmax_multi_stage(
    const float* d_input,
    float* d_output,
    int size,
    int threadsPerBlock
) {
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float *d_in_temp, *d_out_temp;
    float *d_max, *d_sum, *d_exp;

    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_exp, size * sizeof(float));
    cudaMalloc(&d_in_temp, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_out_temp, blocksPerGrid * sizeof(float));

    // ---- Max reduction ----
    int currN = size;
    int numBlocks = blocksPerGrid;
    float* d_src = d_in_temp;
    float* d_dst = d_out_temp;

    max_kernel<<<numBlocks, threadsPerBlock,
                 threadsPerBlock * sizeof(float)>>>(d_input, d_src, currN);

    currN = numBlocks;
    while (currN > 1) {
        numBlocks = (currN + threadsPerBlock - 1) / threadsPerBlock;
        max_kernel<<<numBlocks, threadsPerBlock,
                     threadsPerBlock * sizeof(float)>>>(d_src, d_dst, currN);
        std::swap(d_src, d_dst);
        currN = numBlocks;
    }

    cudaMemcpy(d_max, d_src, sizeof(float), cudaMemcpyDeviceToDevice);

    // ---- Exp ----
    exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_exp, size, d_max
    );

    // ---- Sum reduction ----
    currN = size;
    numBlocks = blocksPerGrid;
    d_src = d_in_temp;
    d_dst = d_out_temp;

    sum_kernel<<<numBlocks, threadsPerBlock,
                 threadsPerBlock * sizeof(float)>>>(d_exp, d_src, currN);

    currN = numBlocks;
    while (currN > 1) {
        numBlocks = (currN + threadsPerBlock - 1) / threadsPerBlock;
        sum_kernel<<<numBlocks, threadsPerBlock,
                     threadsPerBlock * sizeof(float)>>>(d_src, d_dst, currN);
        std::swap(d_src, d_dst);
        currN = numBlocks;
    }

    cudaMemcpy(d_sum, d_src, sizeof(float), cudaMemcpyDeviceToDevice);

    // ---- Normalize ----
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_exp, d_output, size, d_sum
    );

    // cudaDeviceSynchronize();

    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_exp);
    cudaFree(d_in_temp);
    cudaFree(d_out_temp);
}
