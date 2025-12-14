#pragma once
__global__ void softmax_naive(const float* input, float* output, int N, int D);
