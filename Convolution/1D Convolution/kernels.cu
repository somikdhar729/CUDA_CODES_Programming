#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < input_size - kernel_size + 1){
        float temp = 0.0f;
        #pragma unroll
        for(int i = 0;i < kernel_size;i++){
            temp += input[tid + i] * kernel[i];
        }
        output[tid] = temp;
    }    
}

__constant__ float c_kernel[2048]; // Increased size for larger kernels
void upload_kernel_to_constant(const float* h_kernel, int K) {
    cudaMemcpyToSymbol(c_kernel, h_kernel, K * sizeof(float));
}
__global__ void convolution_1d_kernel_constant(const float* input, float* output,
                                           int input_size, int kernel_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < input_size - kernel_size + 1) {
        float temp = 0.0f;
        #pragma unroll
        for (int j = 0; j < kernel_size; ++j) {
            temp += input[tid + j] * c_kernel[j];
        }
        output[tid] = temp;
    }
}

__global__ void convolution_1d_kernel_shared_mem(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    
    extern __shared__ float sI[];   
    int tid = threadIdx.x;
    int base = blockDim.x * blockIdx.x;
    
    for(int i = tid; i < blockDim.x + kernel_size - 1; i+=blockDim.x){
        sI[i] = (base + i < input_size) ? input[base + i] : 0.0f;
        
    }
    __syncthreads();    

    int gid = base + tid;
    if(gid < input_size - kernel_size + 1){
        float temp = 0.0f;
        #pragma unroll
        for(int i = 0;i < kernel_size;i++){
            temp += kernel[i] * sI[tid + i];
        }
        output[gid] = temp;
    }

}