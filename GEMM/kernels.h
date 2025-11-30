#ifndef KERNELS_H
#define KERNELS_H

__global__ void GEMM_naive(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);

__global__ void GEMM_global(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);


__global__ void GEMM_shared_memory(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);

__global__ void GEMM_1D_thread_tiled(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);

__global__ void GEMM_2D_thread_tiled(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);

#endif // KERNELS_H