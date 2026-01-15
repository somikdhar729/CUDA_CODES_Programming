#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void convolution_2d_naive(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    int col_i = blockDim.x * blockIdx.x + threadIdx.x;
    int row_i = blockDim.y * blockIdx.y + threadIdx.y;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if(row_i < output_rows && col_i < output_cols)
    {
        float temp = 0.0f;
        for(int i = 0; i < kernel_rows;i++)
        {
            for(int j = 0;j < kernel_cols;j++)
            {
                temp += input[(row_i + i)* input_cols + col_i + j] * kernel[i * kernel_cols + j];
            }
        }
        output[row_i*output_cols + col_i] = temp;
    }
}

__constant__ float c_kernel[1024];
void upload_kernel_to_constant_2d(const float* h_kernel, int R, int S){
    size_t size = R * S * sizeof(float);
    cudaMemcpyToSymbol(c_kernel, h_kernel, size);
}

__global__ void convolution_2d_kernel_shared_mem(const float* input, const float* kernel,  float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols){

    extern __shared__ float sI[];
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int s_cols = (blockDim.x + kernel_cols - 1);
    
    for(int i = tidy; i < blockDim.y + kernel_rows - 1; i+=blockDim.y){
        for(int j = tidx; j < blockDim.x + kernel_cols - 1; j+=blockDim.x){
            int r = blockDim.y * blockIdx.y + i;
            int c = blockDim.x * blockIdx.x + j;
            if(r < input_rows && c < input_cols){
                sI[i * s_cols + j] = input[r * input_cols + c];
            }
            else{
                sI[i * s_cols + j] = 0.0f;
            }
        }

    }
    __syncthreads();

    if(row < input_rows - kernel_rows + 1 && col < input_cols - kernel_cols + 1){
        float temp = 0.0f;
        #pragma unroll
        for(int i = 0; i < kernel_rows; i++){
            #pragma unroll
            for(int j = 0; j < kernel_cols; j++){
                temp += sI[(tidy + i) * s_cols + tidx + j]* kernel[i * kernel_cols + j];
            }
        }
        output[row * (input_cols - kernel_cols + 1) + col] = temp;
    }
}