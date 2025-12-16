#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels.cuh"
#include <random>
#include <numeric>

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0
#define BLOCK_SIZE 256

#define CUDA_CHECK(err) \
    do{ \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);  \
        } \
    } while(0)

int main(){
    // Get GPU device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // in GB/s
    std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s" << std::endl;


    std::string file_name = "softmax_benchmark_results_naive_kernel.csv";
    std::ofstream outfile(file_name);
    if(!outfile.is_open()){
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return EXIT_FAILURE;
    }
    outfile << "Array Size, Time(msec), Throughput(GB/s), Efficiency(%)\n";

    std::vector<int> array_sizes = {64, 256, 1024, 2048, 4096, 8192, 1<<13, 1<<16, 1<<20, 1<<22, 1<<24};
    // Creating Random Input Data
    std::mt19937 rng(std::random_device{}()); // random number generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // uniform distribution between 0 and 1

    
    int threadsPerBlock = BLOCK_SIZE;

    // Create cudnn handle
    cudaEvent_t start, stop;
    float *d_input, *d_output;
    size_t max_size = *std::max_element(array_sizes.begin(), array_sizes.end());
    cudaMalloc(&d_input, max_size * sizeof(float));
    cudaMalloc(&d_output, max_size * sizeof(float));

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(const auto& size: array_sizes){
        std::vector<float> h_input(size);
        for(auto& val : h_input){
            val = dist(rng);
        }
        // int blocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;
        // Allocate device memory
        // cudaMalloc(&d_input, size * sizeof(float));
        // cudaMalloc(&d_output, size * sizeof(float));
        // Copy input data to device
        cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        
        
        // Warm-up run
        for(int i = 0; i < 3; ++i){
            // std::cout << "Warm-up run " << i+1 << " for size: " << size << std::endl;
            // softmax_naive<<<blocksPerGrid, threadsPerBlock, 2* threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            softmax_multi_stage(
                d_input,
                d_output,
                size,
                threadsPerBlock
            );
            
            cudaDeviceSynchronize();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "Completed warm-up runs for size: " << size << std::endl;
        // Timing runs
        const int iterations = 5;
        float min_ms = std::numeric_limits<float>::max();
        for(int iter = 0; iter < iterations; ++iter){
            cudaEventRecord(start);
            // softmax_naive<<<blocksPerGrid, threadsPerBlock, 2* threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            softmax_multi_stage(
                d_input,
                d_output,
                size,
                threadsPerBlock
            );
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            if(ms < min_ms){
                min_ms = ms;
            }
        }

        // size_t bytes_processed = 2 * size * sizeof(float); // input + output
        // float throughput = bytes_processed / (min_ms / 1e3) / 1e9; // in GB/s
        // float efficiency = (throughput / peak_bandwidth) * 100.0f;

        std::cout << "Array Size: " << size 
                  << ", Time: " << min_ms << " ms"
                //   << ", Throughput: " << throughput << " GB/s"
                //   << ", Efficiency: " << efficiency << " %"
                  << std::endl;
        
    }
    
    cudaFree(d_output);
    cudaFree(d_input);
    
    outfile.close();
    return EXIT_SUCCESS;
}
