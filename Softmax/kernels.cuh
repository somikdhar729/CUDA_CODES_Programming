#pragma once
__global__ void softmax_naive(const float* input, float* output, int N);
__global__ void max_kernel(const float* input, float* output, int N);
__global__ void sum_kernel(const float* input, float* output, int N);
__global__ void exp_kernel(const float* input, float* output, int N, const float* max_);
__global__ void normalize_kernel(const float* input, float* output, int N, const float* exp_sum);
void softmax_multi_stage(const float* d_input, float* d_output, int size, int threadsPerBlock);

void softmax_online(const float* d_input, float* d_output, int size, int threadsPerBlock);