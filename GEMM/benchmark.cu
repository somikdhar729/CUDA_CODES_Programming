#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels.h"
#include "utils.h"
#include <random>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

struct MatrixSize {
    int M, K, N;
};


int main() {
    
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    double peak_memory_bw = (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    std::cout << "Peak Memory BW: " << peak_memory_bw << " GB/s" << std::endl << std::endl;
    
    const float alpha = 1.0f;
    const float beta = 0.5f;
    const int TILE_SIZE = 64;
    const int THREAD_TILE =4;
    const int THREAD_TILE_M = 8;
    const int THREAD_TILE_N = 8;

    
    // Test matrix sizes
    std::vector<MatrixSize> sizes = {
        // {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {8192, 4096, 8192},
        {4096, 8192, 4096}
    };
    
    // Create CSV file
    
    std::ofstream file("benchmark_results_1D_thread_tiled_64_8.csv"); // Change filename as needed
    file << "M,K,N,Time_ms,GFLOPS,MemoryBW_GBps,Efficiency_Percent\n";
    
    for (auto& size : sizes) {
        int M = size.M;
        int K = size.K;
        int N = size.N;
        
        std::cout << "\n=== Testing: A[" << M << "x" << K << "] * B[" << K << "x" << N << "] ===\n";
        
        // Memory check
        size_t total_memory = (size_t)(M*K + K*N + M*N) * sizeof(float);
        if (total_memory > 6000000000ULL) {
            std::cout << "Skipping - requires " << total_memory/1000000 << " MB\n";
            continue;
        }
        
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);
        
        // Create random matrices
        std::cout << "Creating matrices..." << std::endl;
        std::vector<float> h_A = createRandomMatrix(M, K);
        std::vector<float> h_B = createRandomMatrix(K, N);
        std::vector<float> h_C = createRandomMatrix(M, N);
        // std::vector<float> h_C_initial = createRandomMatrix(M, N);

        // std::vector<float> h_C_cpu(M * N);
        // std::vector<float> h_C_gpu(M * N);
        // bool do_verification = (M * N * K <= 128*1024*1024);
        // if (do_verification) {
        //     std::cout << "Computing CPU reference..." << std::endl;
        //     cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N, alpha, beta, h_C.data());
        // }
        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));
        
        // Setup kernel launch config

        // For naive and shared memory kernels
        // dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        // dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        //                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // For 1D thread tiled kernel
        dim3 threadsPerBlock(TILE_SIZE/ THREAD_TILE, TILE_SIZE);
        dim3 blocksPerGrid((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);
        
        // For 2D thread tiled kernel
        // dim3 threadsPerBlock(TILE_SIZE / THREAD_TILE_N, TILE_SIZE / THREAD_TILE_M);
        // dim3 blocksPerGrid((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);
        
        // Create timing events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));
            // GEMM_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            // GEMM_shared_memory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            GEMM_1D_thread_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            // GEMM_2D_thread_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const int num_runs = 10;  // Or 100 for smaller matrices
        std::vector<float> times;

        for (int run = 0; run < num_runs; run++){
            CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());  // Ensure memcpy completes
            
            CUDA_CHECK(cudaEventRecord(start));
            // GEMM_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            // GEMM_shared_memory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            GEMM_1D_thread_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            // GEMM_2D_thread_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float run_time_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&run_time_ms, start, stop));
            times.push_back(run_time_ms);
        }

        std::sort(times.begin(), times.end());
        float time_ms = times[0]; // Minimum time 

        // 1. GFLOPS - Standard GEMM formula
        double total_flops = 2.0 * M * N * K;  // This is the standard
        double gflops = (total_flops / (time_ms / 1000.0)) / 1e9;
        
        // 2. Memory Bandwidth - Minimum theoretical data movement
        // This is what cuBLAS reports and what everyone uses for comparison
        double total_bytes = 4.0 * (M*K + K*N + 2.0*M*N);  // A + B + read/write C
        double memory_bw = total_bytes / (time_ms / 1000.0) / 1e9;
        
        // 3. Efficiency (compared to peak hardware)
        double efficiency = (memory_bw / peak_memory_bw) * 100.0;
        
        // Print results
        std::cout << "Time:        " << time_ms << " ms" << std::endl;
        std::cout << "GFLOPS:      " << gflops << " GFLOPS/s" << std::endl;
        std::cout << "Memory BW:   " << memory_bw << " GB/s" << std::endl;
        std::cout << "Efficiency:  " << efficiency << "% of peak" << std::endl;
        
        // bool verified = true;
        // if (do_verification) {
        //     verify_results(h_C_cpu, h_C_gpu, M, N, "cuBLAS");
        // } else {
        //     std::cout << "\n  Skipping verification (matrix too large)\n";
        // }
        
        // Save to CSV
        file << M << "," << K << "," << N << "," 
             << time_ms << "," << gflops << "," 
             << memory_bw << "," << efficiency << "\n";
        file.flush();
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    file.close();    
    return 0;
}