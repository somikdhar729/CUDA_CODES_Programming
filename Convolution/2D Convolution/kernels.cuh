#pragma once

__global__ void convolution_2d_kernel_shared_mem(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols);
__global__ void convolution_2d_naive(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols);
void upload_kernel_to_constant_2d(const float* h_kernel, int R, int S);
