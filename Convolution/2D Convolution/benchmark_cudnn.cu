#include <cudnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <limits>

#define CUDNN_CHECK(status) do{ \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK(status) do{ \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {

    /* GPU Info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    /* Input & kernel sizes */
    const int input_sizes[]  = {256, 512, 1048, 2048, 3072};
    const int kernel_sizes[] = {3, 5, 7, 11, 15, 21, 31};

    /* cuDNN setup */
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::ofstream file("cudnn_2d_convolution_benchmark.csv");
    file << "H,W,R,S,Time_ms\n";
    float *d_input, *d_output, *d_kernel;

    for (int H : input_sizes) {
        for (int W : input_sizes) {

            size_t in_elems = H * W;
            std::vector<float> h_input(in_elems);
            for (auto &v : h_input) v = dis(gen);

            // float *d_input;
            CUDA_CHECK(cudaMalloc(&d_input, in_elems * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice));

            for (int R : kernel_sizes) {
                for (int S : kernel_sizes) {

                    if (R > H || S > W) continue;

                    int outH = H - R + 1;
                    int outW = W - S + 1;

                    size_t out_elems = outH * outW;
                    // float *d_output;
                    CUDA_CHECK(cudaMalloc(&d_output, out_elems * sizeof(float)));

                    std::vector<float> h_kernel(R * S);
                    for (auto &v : h_kernel) v = dis(gen);

                    // float *d_kernel;
                    CUDA_CHECK(cudaMalloc(&d_kernel, R * S * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), R * S * sizeof(float), cudaMemcpyHostToDevice));

                    /* Descriptors */
                    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, H, W));
                    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, outH, outW));
                    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, R, S));

                    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                        0, 0,        // padding
                        1, 1,        // stride
                        1, 1,        // dilation
                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

                    /* Algorithm selection */
                    cudnnConvolutionFwdAlgoPerf_t perf;
                    int count;
                    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, xDesc, wDesc, convDesc, yDesc, 1, &count, &perf));

                    size_t workspaceSize = perf.memory;
                    void* workspace = nullptr;
                    if (workspaceSize > 0)
                        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

                    float alpha = 1.0f, beta = 0.0f;

                    /* Warm-up */
                    for (int i = 0; i < 5; i++) {
                        CUDNN_CHECK(cudnnConvolutionForward(
                            cudnn,
                            &alpha,
                            xDesc, d_input,
                            wDesc, d_kernel,
                            convDesc,
                            perf.algo,
                            workspace, workspaceSize,
                            &beta,
                            yDesc, d_output));
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    /* Timing */
                    float best_ms = std::numeric_limits<float>::max();
                    for (int i = 0; i < 10; i++) {
                        CUDA_CHECK(cudaEventRecord(start));
                        CUDNN_CHECK(cudnnConvolutionForward(
                            cudnn,
                            &alpha,
                            xDesc, d_input,
                            wDesc, d_kernel,
                            convDesc,
                            perf.algo,
                            workspace, workspaceSize,
                            &beta,
                            yDesc, d_output));
                        CUDA_CHECK(cudaEventRecord(stop));
                        CUDA_CHECK(cudaEventSynchronize(stop));

                        float ms;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                        best_ms = std::min(best_ms, ms);
                    }

                    file << H << "," << W << ","
                         << R << "," << S << ","
                         << best_ms << "\n";

                    // std::cout << "H=" << H << " W=" << W
                    //           << " R=" << R << " S=" << S
                    //           << " : " << best_ms << " ms\n";

                    if (workspace) CUDA_CHECK(cudaFree(workspace));
                    CUDA_CHECK(cudaFree(d_kernel));
                    CUDA_CHECK(cudaFree(d_output));
                }
            }
            CUDA_CHECK(cudaFree(d_input));
        }
    }

    file.close();

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
