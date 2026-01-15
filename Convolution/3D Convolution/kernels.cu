#include "kernels.cuh"
#include <cuda_runtime.h>

__constant__ float d_kernel[125];
void upload_kernel_to_constant_3d(const float* h_kernel, int KD, int KR, int KC) {
    cudaMemcpyToSymbol(d_kernel, h_kernel, KD * KR * KC * sizeof(float));
}

__global__ void convolution_3d_shared_mem(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols){
    
    int depth = threadIdx.z + blockDim.z * blockIdx.z;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    
    extern __shared__ float sI[];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;
    int s_cols = blockDim.x + kernel_cols - 1;
    int s_rows = blockDim.y + kernel_rows - 1;
    for(int i = tidz; i < blockDim.z + kernel_depth - 1; i+=blockDim.z){
        for(int j = tidy; j < blockDim.y + kernel_rows - 1; j +=blockDim.y){
            for(int k = tidx; k < blockDim.x + kernel_cols - 1; k+= blockDim.x){
                int d = blockDim.z * blockIdx.z + i;
                int r = blockDim.y * blockIdx.y + j;
                int c = blockDim.x * blockIdx.x + k;
                if(d < input_depth && r < input_rows && c < input_cols){
                    sI[i * s_rows * s_cols + j * s_cols + k] = input[d * input_rows * input_cols + r * input_cols + c];

                }
                else{
                    sI[i * s_rows * s_cols + j * s_cols + k] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    if(row < (input_rows - kernel_rows + 1) && col < (input_cols - kernel_cols + 1)
        && depth < (input_depth - kernel_depth + 1)){
        float temp = 0.0f;
        #pragma unroll
        for(int i = 0;i < kernel_depth;i++){
            #pragma unroll
            for(int j = 0;j < kernel_rows;j++){
                #pragma unroll
                for(int k = 0;k < kernel_cols;k++){
                    temp += sI[(tidz + i) * s_cols * s_rows + (tidy + j) * s_cols + tidx + k] * kernel[i * kernel_rows * kernel_cols + j * kernel_cols + k];
                }

            }
        }
        output[depth * (input_cols - kernel_cols + 1) * (input_rows - kernel_rows + 1) + row * (input_cols - kernel_cols + 1) + col] = temp;
    }
}

__global__ void convolution_3d_naive(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols){
    
    int depth = threadIdx.z + blockDim.z * blockIdx.z;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < (input_rows - kernel_rows + 1) && col < (input_cols - kernel_cols + 1)
        && depth < (input_depth - kernel_depth + 1)){
        float temp = 0.0f;
        for(int i = 0;i < kernel_depth;i++){
            for(int j = 0;j < kernel_rows;j++){
                for(int k = 0;k < kernel_cols;k++){
                    temp += input[(depth + i) * input_cols * input_rows + (row + j) * input_cols + col + k] 
                            * kernel[ i * kernel_rows * kernel_cols + j * kernel_cols + k];
                }

            }
        }
        output[depth * (input_cols - kernel_cols + 1) * (input_rows - kernel_rows + 1) + row * (input_cols - kernel_cols + 1) + col] = temp;
    }
}