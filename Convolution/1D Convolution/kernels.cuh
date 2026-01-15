#pragma once
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size);

// extern __device__ __constant__ float c_kernel[2048];
void upload_kernel_to_constant(const float* h_kernel, int K);
__global__ void convolution_1d_kernel_constant(const float* input, float* output, int input_size, int kernel_size);

__global__ void convolution_1d_kernel_shared_mem(const float* input, const float* kernel, float* output, int input_size, int kernel_size);
