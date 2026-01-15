#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <limits>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define CUDA_CHECK(err) \
    do { \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {

    /* ---------------- GPU Info ---------------- */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;

    /* ---------------- Random Init ---------------- */
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    /* ---------------- 2D Sizes ---------------- */
    const int input_rows[]  = {256, 512, 1048, 2048, 3072};
    const int input_cols[]  = {256, 512, 1048, 2048, 3072};
    const int kernel_rows[] = {3, 5, 7, 11, 15, 21, 31};
    const int kernel_cols[] = {3, 5, 7, 11, 15, 21, 31};

    // std::ofstream result_file("benchmark_results_2d_shared_mem.csv");
    std::ofstream result_file("benchmark_results_2d_naive.csv");
    result_file << "H,W,R,S,Time_ms\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float *d_input, *d_output, *d_kernel;

    /* ---------------- Benchmark Loop ---------------- */
    for (int H : input_rows) {
        for (int W : input_cols) {

            size_t input_elems = (size_t)H * W;
            std::vector<float> h_input(input_elems);
            for (auto &v : h_input) v = dis(gen);

            // float* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, input_elems * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(
                d_input,
                h_input.data(),
                input_elems * sizeof(float),
                cudaMemcpyHostToDevice));

            for (int R : kernel_rows) {
                for (int S : kernel_cols) {

                    if (R > H || S > W) continue;

                    int outH = H - R + 1;
                    int outW = W - S + 1;
                    size_t output_elems = (size_t)outH * outW;

                    // float* d_output;
                    CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(float)));

                    /* Kernel */
                    std::vector<float> h_kernel(R * S);
                    for (auto &v : h_kernel) v = dis(gen);

                    // upload_kernel_to_constant_2d(h_kernel.data(), R, S);
                    // float *d_kernel;
                    CUDA_CHECK(cudaMalloc(&d_kernel, R * S * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), R * S * sizeof(float), cudaMemcpyHostToDevice));
                    /* Launch config */
                    dim3 threadsPerBlock(16, 16);
                    dim3 BlocksPerGrid((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,(outH + threadsPerBlock.y - 1) / threadsPerBlock.y
                    );

                    size_t sharedMemSize = (threadsPerBlock.x + S - 1) * (threadsPerBlock.y + R - 1) * sizeof(float);

                    /* Warm-up */
                    for (int i = 0; i < 5; i++) {
                        convolution_2d_naive<<<BlocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, H, W, R, S);
                        // convolution_2d_kernel_shared_mem<<<BlocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_kernel, d_output, H, W, R, S);
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    /* Timing */
                    float best_ms = std::numeric_limits<float>::max();
                    for (int i = 0; i < 10; i++) {
                        CUDA_CHECK(cudaEventRecord(start));
                        convolution_2d_naive<<<BlocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, H, W, R, S);
                        // convolution_2d_kernel_shared_mem<<<BlocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_kernel, d_output, H, W, R, S);
                        CUDA_CHECK(cudaEventRecord(stop));
                        CUDA_CHECK(cudaEventSynchronize(stop));

                        float ms;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                        best_ms = std::min(best_ms, ms);
                    }

                    result_file << H << "," << W << ","
                                << R << "," << S << ","
                                << best_ms << "\n";

                    // std::cout
                    //     << "H=" << H
                    //     << " W=" << W
                    //     << " R=" << R
                    //     << " S=" << S
                    //     << " : " << best_ms << " ms\n";

                    CUDA_CHECK(cudaFree(d_output));
                    CUDA_CHECK(cudaFree(d_kernel));
                }
            }
            CUDA_CHECK(cudaFree(d_input));
        }
    }

    result_file.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
