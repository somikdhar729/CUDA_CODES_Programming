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

    /* ---------------- RNG ---------------- */
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    /* ---------------- Sizes ---------------- */
    const int input_sizes[]  = {256};
    const int kernel_sizes[] = {3, 5, 7};

    // std::ofstream file("benchmark_results_3d_shared_mem.csv");
    std::ofstream file("benchmark_results_3d_naive.csv");
    file << "D,H,W,KD,KR,KC,Time_ms,Throughput\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float *d_input, *d_output, *d_kernel;

    /* ---------------- Benchmark Loop ---------------- */
    for (int D : input_sizes) {
        for (int H : input_sizes) {
            for (int W : input_sizes) {

                size_t input_elems = (size_t)D * H * W;
                std::vector<float> h_input(input_elems);
                for (auto &v : h_input) v = dis(gen);

                // float* d_input;
                CUDA_CHECK(cudaMalloc(&d_input,
                    input_elems * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(
                    d_input, h_input.data(),
                    input_elems * sizeof(float),
                    cudaMemcpyHostToDevice));

                for (int KD : kernel_sizes) {
                    for (int KR : kernel_sizes) {
                        for (int KC : kernel_sizes) {

                            if (KD > D || KR > H || KC > W) continue;

                            int outD = D - KD + 1;
                            int outH = H - KR + 1;
                            int outW = W - KC + 1;

                            size_t output_elems =
                                (size_t)outD * outH * outW;

                            // float* d_output;
                            CUDA_CHECK(cudaMalloc(
                                &d_output,
                                output_elems * sizeof(float)));

                            /* Kernel init */
                            std::vector<float> h_kernel(KD * KR * KC);
                            for (auto &v : h_kernel) v = dis(gen);

                            // upload_kernel_to_constant_3d(
                            //     h_kernel.data(), KD, KR, KC);
                            // float *d_kernel;
                            CUDA_CHECK(cudaMalloc(&d_kernel, KD * KR * KC * sizeof(float)));
                            CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), KD * KR * KC * sizeof(float), cudaMemcpyHostToDevice));
                            /* Launch config */
                            dim3 threadsPerBlock(16, 8, 2);
                            dim3 blocksPerGrid((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                (outH + threadsPerBlock.y - 1) / threadsPerBlock.y,
                                (outD + threadsPerBlock.z - 1) / threadsPerBlock.z
                            );

                            size_t shmem = (threadsPerBlock.x + KC - 1) * (threadsPerBlock.y + KR - 1) * (threadsPerBlock.z + KD - 1) * sizeof(float);

                            /* Warm-up */
                            for (int i = 0; i < 5; i++) {
                                convolution_3d_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, D, H, W, KD, KR, KC);
                                // convolution_3d_shared_mem<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_input, d_kernel, d_output, D, H, W, KD, KR, KC);
                            }
                            CUDA_CHECK(cudaDeviceSynchronize());
                            // std::cout << "Warm-up done for D=" << D << " H=" << H << " W=" << W << " KD=" << KD << " KR=" << KR << " KC=" << KC << std::endl;
                            /* Timing */
                            float best_ms = std::numeric_limits<float>::max();

                            for (int i = 0; i < 10; i++) {
                                CUDA_CHECK(cudaEventRecord(start));
                                convolution_3d_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, D, H, W, KD, KR, KC);
                                // convolution_3d_shared_mem<<<blocksPerGrid, threadsPerBlock, shmem>>>(
                                        // d_input, d_kernel, d_output,
                                        // D, H, W,
                                        // KD, KR, KC);
                                CUDA_CHECK(cudaEventRecord(stop));
                                CUDA_CHECK(cudaEventSynchronize(stop));

                                float ms;
                                CUDA_CHECK(cudaEventElapsedTime(
                                    &ms, start, stop));
                                best_ms = std::min(best_ms, ms);
                            }
                            double FLOPs = 2.0 * 1 * 1 * outD * outH * outW * KD * KR * KC;
                            double TFLOPs = FLOPs / (best_ms * 1e6);
                            file << D << "," << H << "," << W << ","
                                 << KD << "," << KR << "," << KC << ","
                                 << best_ms << "," << TFLOPs << "\n";

                            // std::cout << "D=" << D << " H=" << H << " W=" << W << " KD=" << KD << " KR=" << KR << " KC=" << KC
                            //           << " Time=" << best_ms << " ms"
                            //           << " Throughput=" << TFLOPs << " TFLOPs" << std::endl;

                            CUDA_CHECK(cudaFree(d_output));
                        }
                    }
                }
                CUDA_CHECK(cudaFree(d_input));
            }
        }
    }

    file.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
