#include "kernels.cuh"

__global__ void reduction_1(const float* input, float* output, int N){
    extern __shared__ float sOut[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    sOut[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    for(int i = 1;i < blockDim.x;i<<=1){
        if(tid % (2*i) == 0){
            sOut[tid] += sOut[tid + i];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}

__global__ void reduction_2(const float* input, float* output, int N){
    extern __shared__ float sOut[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    sOut[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    for(int i = 1;i < blockDim.x;i <<= 1){
        int index = 2 * i * tid;
        if(index < blockDim.x){
            sOut[index] += sOut[index + i];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}

__global__ void reduction_3(const float* input, float* output, int N){
    extern __shared__ float sout[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // Load data into shared memory
    if(gid < N){
        sout[tid] = input[gid];
    }
    else{
        sout[tid] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction within block
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sout[tid] += sout[tid + i];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if(tid == 0){
        output[blockIdx.x] = sout[0];
    }
}


__global__ void reduction_4(const float* input, float* output, int N){
    extern __shared__ float sOut[];
    int tid = threadIdx.x;
    int gid = 2 * blockDim.x * blockIdx.x + threadIdx.x;


    // Load two elements per thread if within bounds
    sOut[tid] = (gid < N ? input[gid] : 0.0f) 
        + (gid + blockDim.x < N ? input[gid + blockDim.x] : 0.0f);

    __syncthreads();

    for(int i = blockDim.x / 2;i > 0;i >>=1){
        if(tid < i){
            sOut[tid] += sOut[tid + i];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}

__device__ void warpReduce(volatile float* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}


__global__ void reduction_5(const float* input, float* output, int N){
    extern __shared__ float sOut[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    sOut[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    for(int i = blockDim.x / 2;i > 32;i >>=1){
        if(tid < i){
            sOut[tid] += sOut[tid + i];
        }
        __syncthreads();
    }
    if(tid < 32){
        warpReduce(sOut,tid);
    }

    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}



__global__ void reduction_6(const float* input, float* output, int N){
    extern __shared__ float sOut[];
    int tid = threadIdx.x;
    int gid = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    int gridSize = 2 * blockDim.x * gridDim.x;
    sOut[tid] = 0;

    while(gid < N){
        sOut[tid] += (gid < N ? input[gid] : 0.0f) + (gid + blockDim.x < N ? input[gid + blockDim.x] : 0.0f);
        gid += gridSize;
    }
    __syncthreads();

    for(int i = blockDim.x / 2;i > 0;i >>=1){
        if(tid < i){
            sOut[tid] += sOut[tid + i];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = sOut[0];
    }
}


