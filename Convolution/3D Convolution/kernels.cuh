#pragma once

void upload_kernel_to_constant_3d(const float* h_kernel, int KD, int KR, int KC);

__global__ void convolution_3d_shared_mem(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols);

__global__ void convolution_3d_naive(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols);
