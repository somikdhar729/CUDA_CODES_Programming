#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels.cuh"
#include <random>
#include <algorithm>
#include <numeric>

#define EXIT_FAILURE 1
#define BLOCK_SIZE 256

#define CUDA_CHECK(err) \
    if(err != cudaSuccess){ \
        std::cerr<< "CUDA Error: "<<cudaGetErrorString(err)<<" at line "<<__LINE__<<std::endl; \
        exit(EXIT_FAILURE); \
    }

// CPU reference implementation
float cpu_reduce(const std::vector<float>& data) {
    return std::accumulate(data. begin(), data. end(), 0.0f);
}

int main(){
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Running on GPU: " << prop. name << std::endl;
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // in GB/s
    std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    // std::ofstream file("reduction_benchmark_results_kernel_1.csv");
    // std::ofstream file("reduction_benchmark_results_kernel_2.csv");
    // std::ofstream file("reduction_benchmark_results_kernel_3. csv");
    // std::ofstream file("reduction_benchmark_results_kernel_4.csv");
    // std::ofstream file("reduction_benchmark_results_kernel_5.csv");
    std::ofstream file("reduction_benchmark_results_kernel_6.csv");
    file << "Array Size,Time (ms),Throughput (GB/s),Efficiency (%),Verified" << std::endl;

    // Array Sizes
    std::vector<size_t> sizes = {64, 128, 256, 1024, 2048, 4096, 8192, 1<<6, 1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    
    // Creating random data
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    int threadsPerBlock = BLOCK_SIZE;

    // Creating the array
    for (const auto& size : sizes) {
        std::vector<float> h_data(size);
        for (auto& val : h_data) {
            val = dist(rng);
        }

        // CPU reference result
        float cpu_result = cpu_reduce(h_data);

        int blocksPerGrid = (size + threadsPerBlock - 1) / (threadsPerBlock);
        int sharedMemSize = threadsPerBlock * sizeof(float);

        // Allocate device memory
        float* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        const float* input = d_input;
        float* output;
        CUDA_CHECK(cudaMalloc(&output, sizeof(float)));

        float* d_in_temp;
        float* d_out_temp;
        
        CUDA_CHECK(cudaMalloc(&d_in_temp, blocksPerGrid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_temp, blocksPerGrid * sizeof(float)));

        // Timing variables
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for(int i = 0; i < 3; i++){
            const float* d_src = input;
            float* d_dst = d_in_temp;
            
            int curr_N = size;
            while(curr_N > 1){
                blocksPerGrid = (curr_N + threadsPerBlock - 1) / (threadsPerBlock);
                // reduction_1<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_2<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_3<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_4<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_5<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                reduction_6<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                curr_N = blocksPerGrid;
                d_src = d_dst;
                d_dst = (d_dst == d_in_temp) ? d_out_temp : d_in_temp;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Launch kernel and measure time
        std::vector<float> times;
        const float* final_result_ptr = nullptr;
        
        for(int i = 0; i < 10; i++){
            CUDA_CHECK(cudaEventRecord(start));
            
            const float* d_src = input;
            float* d_dst = d_in_temp;
            
            int curr_N = size;
            while(curr_N > 1){
                blocksPerGrid = (curr_N + threadsPerBlock - 1) / (threadsPerBlock);
                // reduction_1<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_2<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_3<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_4<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                // reduction_5<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                reduction_6<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_src, d_dst, curr_N);
                curr_N = blocksPerGrid;
                d_src = d_dst;
                d_dst = (d_dst == d_in_temp) ? d_out_temp : d_in_temp;
            }
            
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float run_time_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&run_time_ms, start, stop));
            times.push_back(run_time_ms);
            
            // Save the final result pointer from the last iteration
            if(i == 9) {
                final_result_ptr = d_src;
            }
        }
        
        // Copy final result to output and then to host
        CUDA_CHECK(cudaMemcpy(output, final_result_ptr, sizeof(float), cudaMemcpyDeviceToDevice));
        
        float gpu_result = 0.0f;
        CUDA_CHECK(cudaMemcpy(&gpu_result, output, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Calculate error
        float error = std::abs(gpu_result - cpu_result);
        float relative_error = error / std::abs(cpu_result);
        
        // Sort times and get minimum
        std::sort(times. begin(), times.end());
        float time_ms = times[0]; // Minimum time 
        
        // Calculate total bytes transferred (accurate method)
        size_t total_elements = 0;
        int curr_N = size;
        int temp_blocks = 0;
        
        while(curr_N > 1){
            total_elements += curr_N; // Read curr_N elements
            temp_blocks = (curr_N + threadsPerBlock - 1) / threadsPerBlock;
            if(temp_blocks > 1) {
                total_elements += temp_blocks; // Write temp_blocks elements
            }
            curr_N = temp_blocks;
        }
        
        float bytes_transferred = total_elements * sizeof(float);
        float bandwidth = bytes_transferred / (time_ms * 1e-3) / 1e9; // in GB/s
        float efficiency = (bandwidth / peak_bandwidth) * 100.0f; // Efficiency as percentage

        // Print verification results
        std::cout << "Size: " << size 
                  << " | Time: " << time_ms << " ms"
                  << " | BW: " << bandwidth << " GB/s"
                  << " | Eff: " << efficiency << "%"
                  << " | Verified:" << (relative_error < 1e-4 ? "PASS" : "FAIL")
                //   << " | GPU: " << gpu_result 
                //   << " | CPU: " << cpu_result 
                  << " | Rel Error: " << (relative_error * 100.0f) << "%"<< std::endl;
        
        bool pass = relative_error < 1e-4;
        // if(relative_error < 1e-4) {
        //     std::cout << " ✓ PASS" << std::endl;
        // } else {
        //     std::cout << " ✗ FAIL" << std::endl;
        // }

        // Write results to file
        file << size << "," << time_ms << "," << bandwidth << "," 
             << efficiency << "," << (pass ? "PASS" : "FAIL") << std::endl;

        // Cleanup
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_in_temp));
        CUDA_CHECK(cudaFree(d_out_temp));
        CUDA_CHECK(cudaFree(output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    file.close();
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark complete!  Results saved to CSV file." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}