#include "kernels.cuh"
#include <float.h>

__global__ void softmax_naive(const float *input, float output, int N){
    extern __shared__ float shared_data[];

}