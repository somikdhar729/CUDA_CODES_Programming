#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include <random>
#include <fstream>


#define CUDA_CHECK(err) \
    do{ \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1);  \
        } \
    } while(0)

int main(){
    // Get GPU device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    double peak_memory_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // in GB/s
    std::cout << "Peak Memory Bandwidth: " << peak_memory_bandwidth << " GB/s" << std::endl;

    // Random number generation
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    const int input_sizes[] = {1500000};
    const int kernel_sizes[] = {31, 63, 127, 255, 1023, 2047};
    // std::ofstream result_file("benchmark_results_naive.csv");
    // std::ofstream result_file("benchmark_results_constant.csv");
    std::ofstream result_file("benchmark_results_shared_mem.csv");
    result_file << "Input Size,Kernel Size,Time (ms)" << std::endl;
    // result_file_shared << "Input Size,Kernel Size,Time (ms)" << std::endl;
    // result_file_shared_mem << "Input Size,Kernel Size,Time (ms)" << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for(int N: input_sizes){
        // Host input allocation and initialization
        std::vector<float> h_input(N);
        for(int i = 0; i < N; ++i){
            h_input[i] = dis(gen);
        }
        // Device input allocation
        float *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        for(int K: kernel_sizes){
            if(K >= N) continue; // Kernel size must be less than input size
            int output_size = N - K + 1;
            // Host output allocation
            std::vector<float> h_output(output_size, 0.0f);
            // Device output allocation
            float *d_output;
            CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
            // Host kernel allocation and initialization
            std::vector<float> h_kernel(K);
            for(int i = 0; i < K; ++i){
                h_kernel[i] = dis(gen);
            }
            // // Copy kernel to constant memory
            
            // Device kernel allocation
            float *d_kernel;
            CUDA_CHECK(cudaMalloc(&d_kernel, K * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            // CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel.data(), K * sizeof(float)));

            // upload_kernel_to_constant(h_kernel.data(), K);
            // Launch parameters
            int blockSize = 256;
            int numBlocks = (output_size + blockSize - 1) / blockSize;
            int sharedMemSize = (blockSize + K - 1) * sizeof(float);
            // Warm-up run
            for(int i = 0; i < 10; ++i){
                // convolution_1d_kernel<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output, N, K);
                // convolution_1d_kernel_constant<<<numBlocks, blockSize>>>(d_input, d_output, N, K);
                convolution_1d_kernel_shared_mem<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_kernel, d_output, N, K);

            }
            CUDA_CHECK(cudaDeviceSynchronize());
            // Timing run
            float min_ms = std::numeric_limits<float>::max();
            for(int iter = 0; iter < 100; ++iter){
                CUDA_CHECK(cudaEventRecord(start));
                // convolution_1d_kernel<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output, N, K);
                // convolution_1d_kernel_constant<<<numBlocks, blockSize>>>(d_input, d_output, N, K);
                convolution_1d_kernel_shared_mem<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_kernel, d_output, N, K);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                if(ms < min_ms) min_ms = ms;
            }

            result_file << N << "," << K << "," << min_ms << std::endl;
            // std::cout << "Shared Memory - Input Size: " << N << ", Kernel Size: " << K << ", Time: " << min_ms << " ms" << std::endl;

            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaFree(d_kernel));

        }
        CUDA_CHECK(cudaFree(d_input));
    }
    result_file.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}