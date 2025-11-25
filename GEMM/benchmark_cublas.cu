#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

struct MatrixSize {
    int M, K, N;
};

std::vector<float> createRandomMatrix(int rows, int cols,
                                      float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    std::vector<float> matrix(rows * cols);
    for (auto& x : matrix) x = dis(gen);
    return matrix;
}

// CPU reference GEMM (row-major): C = A * B
void cpu_gemm(const float* A, const float* B, float* C,
              int M, int K, int N, float alpha, float beta, const float* C_init)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m*K + k] * B[k*N + n];
            C[m*N + n] = alpha * sum + beta * C_init[m*N + n];
        }
    }
}

// Compare CPU and GPU results
void verify_results(const std::vector<float>& C_cpu,
                    const std::vector<float>& C_gpu,
                    int M, int N, const std::string& label = "")
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    int errors = 0;

    for (int i = 0; i < M * N; i++) {
        double a = C_cpu[i];
        double b = C_gpu[i];
        double abs_err = fabs(a - b);
        double rel_err = abs_err / (fabs(a) + 1e-7);

        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        
        if (rel_err > 1e-3) errors++;
    }

    std::cout << "Verification " << label << ":\n";
    std::cout << "  Max Abs Error = " << max_abs << "\n";
    std::cout << "  Max Rel Error = " << max_rel << "\n";

    if (errors == 0)
        std::cout << "  ✔ PASS\n";
    else
        std::cout << "  ✘ FAIL (" << errors << "/" << (M*N) << " elements)\n";
}

int main() {

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    double peak_memory_bw = (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    std::cout << "Peak Memory BW: " << peak_memory_bw << " GB/s\n\n";

    const float alpha = 1.0f;
    const float beta  = 0.5f;

    std::vector<MatrixSize> sizes = {
        {4, 4, 4},
        {16, 16, 16},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8191},
        {8192, 4096, 8192},
        {4096, 8192, 4096},
    };

    std::ofstream file("results_gemm_cublas.csv");
    file << "M,K,N,Time_ms,GFLOPS,MemoryBW_GBps,Efficiency_Percent\n";

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (auto& size : sizes) {
        int M = size.M;
        int K = size.K;
        int N = size.N;

        std::cout << "\n=== Testing: A[" << M << "x" << K << "] * B[" << K << "x" << N << "] ===\n";

        size_t size_A = (size_t)M * K * sizeof(float);
        size_t size_B = (size_t)K * N * sizeof(float);
        size_t size_C = (size_t)M * N * sizeof(float);

        size_t total_memory = size_A + size_B + size_C;
        if (total_memory > 6000000000ULL) {
            std::cout << "Skipping - requires " << total_memory/1000000 << " MB\n";
            continue;
        }

        // Host matrices
        std::cout << "Creating matrices..." << std::endl;
        std::vector<float> h_A = createRandomMatrix(M, K);
        std::vector<float> h_B = createRandomMatrix(K, N);
        std::vector<float> h_C_initial = createRandomMatrix(M, N);
        // std::vector<float> h_C_cpu(M * N);
        std::vector<float> h_C_gpu(M * N);

        // CPU reference computation (only for smaller matrices)
        // bool do_verification = (M * N * K <= 128*1024*1024);
        // if (do_verification) {
        //     std::cout << "Computing CPU reference..." << std::endl;
        //     cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N, alpha, beta, h_C_initial.data());
        // }

        // Device matrices
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C_initial.data(), size_C, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warm-up runs
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(d_C, h_C_initial.data(), size_C, cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N
            ));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset C to initial values before timing
        CUDA_CHECK(cudaMemcpy(d_C, h_C_initial.data(), size_C, cudaMemcpyHostToDevice));

        // Benchmark - single iteration
        std::cout << "Timing single iteration..." << std::endl;
        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        
        // GFLOPS
        double total_flops = 2.0 * M * N * K;
        double gflops = (total_flops / (time_ms / 1000.0)) / 1e9;
        
        // Memory Bandwidth - Minimum theoretical data movement
        double total_bytes = 4.0 * (M*K + K*N + 2.0*M*N);
        double memory_bw = total_bytes / (time_ms / 1000.0) / 1e9;
        
        //  Efficiency
        double efficiency = (memory_bw / peak_memory_bw) * 100.0;

        // Print results
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Time:        " << time_ms << " ms" << std::endl;
        std::cout << "GFLOPS:      " << gflops << " GFLOPS/s" << std::endl;
        std::cout << "Memory BW:   " << memory_bw << " GB/s" << std::endl;
        std::cout << "Efficiency:  " << efficiency << "% of peak" << std::endl;

        // Verify correctness
        // bool verified = true;
        // if (do_verification) {
        //     verify_results(h_C_cpu, h_C_gpu, M, N, "cuBLAS");
        // } else {
        //     std::cout << "\n⚠️  Skipping verification (matrix too large)\n";
        // }

        // Save to CSV
        file << M << "," << K << "," << N << ","
             << time_ms << "," << gflops << ","
             << memory_bw << "," << efficiency << "," <<"\n";
            //  << (verified ? "PASS" : "SKIP") << "\n";
        file.flush();

        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    file.close();

    std::cout << "\n Results saved to results_gemm_cublas.csv\n";
    return 0;
}