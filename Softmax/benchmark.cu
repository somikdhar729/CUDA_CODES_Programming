#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels.cuh"
#include <random>
#include <numeric>
#include <cudnn.h>


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

    std::vector<int> array_sizes = {64, 256, 1024, 2048, 4096, 8192, 1<<6, 1<<10, 1<<13, 1<<16, 1<<20, 1<<24};
    // Creating Random Input Data
    std::mt19937 rng(std::random_device{}()); // random number generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // uniform distribution between 0 and 1

    int threadsPerBlock = BLOCK_SIZE;

    for(const auto& size: array_sizes){
        std::vector<float> h_input(size);
        for(auto& val : h_input){
            val = dist(rng);
        }

        

        
    }

    



    return EXIT_SUCCESS;
}
