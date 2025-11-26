#include "kernels.h"
#define TILE_SIZE 64 // Change tile size as needed
#define THREAD_TILE 4 // For 1D thread tiled kernel
#define THREAD_TILE_M 8 // For 2D thread tiled kernel
#define THREAD_TILE_N 8 // For 2D thread tiled kernel

__global__ void GEMM_naive(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < M && col < N){
        float temp = 0.0f;
        for(int i = 0; i < K; i++){
            temp += A[row*K + i] * B[col + i*N];
        }
        C[row*N + col] = alpha * temp + beta * C[row*N + col];
    }

}

__global__ void GEMM_shared_memory(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float temp = 0.0f;
    for(int i = 0;i < (K + TILE_SIZE - 1)/TILE_SIZE; i++)
    {
        // Load from tile A
        if(row < M && threadIdx.x + TILE_SIZE * i < K)
        {
            sA[threadIdx.y][threadIdx.x] = (A[row * K + threadIdx.x + TILE_SIZE*i]);
        }
        else{
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col < N && threadIdx.y + TILE_SIZE * i < K)
        {
            sB[threadIdx.y][threadIdx.x] = (B[(threadIdx.y + TILE_SIZE * i)*N + col]);
        }
        else{
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int j = 0;j < TILE_SIZE; j++)
        {
            temp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();

    } 
    if(row < M && col < N){
        C[row*N + col] = alpha * temp + beta * C[row*N + col];
    }
}

__global__ void GEMM_1D_thread_tiled(const float* A, const float* B, float*C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.y + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x * THREAD_TILE + blockIdx.x * TILE_SIZE;

    __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];

    float temp[THREAD_TILE];
    for (int i = 0; i < THREAD_TILE; i++)
        temp[i] = 0.0f;

    for(int i = 0; i < (K + TILE_SIZE - 1)/TILE_SIZE; i++){
        
        // LOAD MATRIX A
        for(int j = 0; j < THREAD_TILE; j++){
            int gCol = i * TILE_SIZE + threadIdx.x * THREAD_TILE + j;
            if(row < M && gCol < K){
                sA[threadIdx.y][threadIdx.x * THREAD_TILE + j] = A[row * K + gCol];
            }
            else{
                sA[threadIdx.y][threadIdx.x * THREAD_TILE + j] = 0.0f;
            }
        } 

        // LOAD MATRIX B
        for(int j = 0; j < THREAD_TILE; j++)
        {
            int gCol = threadIdx.x * THREAD_TILE + j + blockIdx.x * TILE_SIZE;
            int gRow = i * TILE_SIZE + threadIdx.y; 
            if(gRow < K && gCol < N){
                sB[threadIdx.y][threadIdx.x * THREAD_TILE + j] = B[gRow * N + gCol];
            }
            else{
                sB[threadIdx.y][threadIdx.x * THREAD_TILE + j] = 0.0f;
            }
        }

        __syncthreads();

        for(int j = 0; j < TILE_SIZE;j++){
            float a_val = sA[threadIdx.y][j];
            for(int k = 0; k < THREAD_TILE; k++){
                float b_val = sB[j][threadIdx.x * THREAD_TILE + k];
                temp[k] += a_val * b_val;
            }
        }
        __syncthreads();
    }

    for(int j = 0; j < THREAD_TILE;j++){
        if(row < M && col + j< N){
            C[row * N + col + j] = alpha * temp[j] + beta * C[row * N + col + j];
        }
    }
}

__global__ void GEMM_2D_thread_tiled(const float* A, const float* B, float*C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.y * THREAD_TILE_M + blockIdx.y * TILE_SIZE;
    int col = threadIdx.x * THREAD_TILE_N + blockIdx.x * TILE_SIZE;

    __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];

    float temp[THREAD_TILE_M][THREAD_TILE_N];
    for (int i = 0; i < THREAD_TILE_M; i++)
    {
        for(int j = 0;j < THREAD_TILE_N;j++)
        {
            temp[i][j] = 0.0f;
        }
    }
        

    for(int i = 0; i < (K + TILE_SIZE - 1)/TILE_SIZE; i++){
        
        // LOAD MATRIX A
        for(int j = 0; j < THREAD_TILE_N; j++){
            for(int k = 0; k < THREAD_TILE_M; k++){
                int sRow = threadIdx.y * THREAD_TILE_M + k;
                int sCol = threadIdx.x * THREAD_TILE_N + j;
                int gRow = blockIdx.y * TILE_SIZE + sRow;
                int gCol = i  * TILE_SIZE + sCol;
                if(gRow < M && gCol < K){
                    sA[sRow][sCol] = A[gRow * K + gCol];
                }
                else{
                    sA[sRow][sCol] = (0.0f);
                }
            }
        } 

        // LOAD MATRIX B
        for(int j = 0; j < THREAD_TILE_N; j++)
        {
            for(int k = 0;k < THREAD_TILE_M; k++){
                int sRow = threadIdx.y * THREAD_TILE_M + k;
                int sCol = threadIdx.x * THREAD_TILE_N + j;
                int gRow = i * TILE_SIZE + sRow;
                int gCol = blockIdx.x * TILE_SIZE + sCol;
                if(gRow < K && gCol < N){
                    sB[sRow][sCol] = B[gRow * N + gCol];
                }
                else{
                    sB[sRow][sCol] =(0.0f);
                }
            }
        }

        __syncthreads();

        for(int m = 0;m < TILE_SIZE;m++){
            for(int k = 0; k < THREAD_TILE_M;k++){
                for(int l = 0;l < THREAD_TILE_N;l++){
                    float a_val = (sA[threadIdx.y * THREAD_TILE_N + k][m]);
                    float b_val = (sB[m][threadIdx.x* THREAD_TILE_M + l]);
                    temp[k][l] += a_val * b_val;
                }
            }
        }
            
        
        __syncthreads();
    }

    for(int i = 0; i < THREAD_TILE_M; i++){
        for(int j = 0; j < THREAD_TILE_N; j++){
            if(row + i < M && col + j < N)
                C[(row + i)*N + col + j] = alpha * temp[i][j] + beta * C[(row + i) * N + col + j];
    }
}
}