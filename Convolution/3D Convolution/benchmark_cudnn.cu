#include <cudnn.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <limits>

#define CUDNN_CHECK(status) do { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK(status) do { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {

    /* ---------------- GPU Info ---------------- */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    /* ---------------- Constraints ---------------- */
    const int input_sizes[]  = {256};   
    const int kernel_sizes[] = {3, 5, 7};            

    /* ---------------- cuDNN Setup ---------------- */
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

    float *d_input, *d_output, *d_kernel;

    std::ofstream file("cudnn_3d_convolution_benchmark.csv");
    file << "D,H,W,KD,KR,KC,Time_ms,Throughput\n";

    /* ---------------- Benchmark ---------------- */
    for (int D : input_sizes) {
        for (int H : input_sizes) {
            for (int W : input_sizes) {

                size_t in_elems = (size_t)D * H * W;
                std::vector<float> h_input(in_elems);
                for (auto &v : h_input) v = dis(gen);

                CUDA_CHECK(cudaMalloc(&d_input, in_elems * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice));

                for (int KD : kernel_sizes) {
                    for (int KR : kernel_sizes) {
                        for (int KC : kernel_sizes) {

                            if (KD > D || KR > H || KC > W) continue;

                            int outD = D - KD + 1;
                            int outH = H - KR + 1;
                            int outW = W - KC + 1;

                            size_t out_elems = (size_t)outD * outH * outW;

                            CUDA_CHECK(cudaMalloc(&d_output, out_elems * sizeof(float)));

                            std::vector<float> h_kernel(KD * KR * KC);
                            for (auto &v : h_kernel) v = dis(gen);

                            CUDA_CHECK(cudaMalloc(&d_kernel, KD * KR * KC * sizeof(float)));
                            CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), KD * KR * KC * sizeof(float), cudaMemcpyHostToDevice));

                            /* -------- Tensor Descriptors (NCDHW) -------- */
                            int x_dims[5]    = {1, 1, D, H, W};
                            int x_strides[5]= {D * H * W, H * W, H * W, W, 1};

                            int y_dims[5]    = {1, 1, outD, outH, outW};
                            int y_strides[5]= {outD * outH * outW, outH * outW, outH * outW, outW, 1};

                            CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 5, x_dims, x_strides));
                            CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 5, y_dims, y_strides));

                            int w_dims[5] = {1, 1, KD, KR, KC};
                            CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dims));

                            /* -------- Convolution Descriptor -------- */
                            int pad[3]      = {0, 0, 0};
                            int stride[3]   = {1, 1, 1};
                            int dilation[3] = {1, 1, 1};

                            CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(convDesc, 3, pad, stride, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

                            /* -------- Algorithm Selection -------- */
                            cudnnConvolutionFwdAlgoPerf_t perf;
                            int count;
                            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, xDesc, wDesc, convDesc, yDesc, 1, &count, &perf));

                            void* workspace = nullptr;
                            if (perf.memory > 0)
                                CUDA_CHECK(cudaMalloc(&workspace, perf.memory));

                            float alpha = 1.0f, beta = 0.0f;

                            /* -------- Warm-up -------- */
                            for (int i = 0; i < 5; i++) {
                                CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, xDesc, d_input, wDesc, d_kernel, convDesc, perf.algo, workspace, perf.memory, &beta, yDesc, d_output));
                            }
                            CUDA_CHECK(cudaDeviceSynchronize());
                            // std::cout<<"Warm-up done for D="<<D<<" H="<<H<<" W="<<W<<" KD="<<KD<<" KR="<<KR<<" KC="<<KC<<std::endl;
                            /* -------- Timing -------- */
                            float best_ms =
                                std::numeric_limits<float>::max();

                            for (int i = 0; i < 10; i++) {
                                CUDA_CHECK(cudaEventRecord(start));
                                CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, xDesc, d_input, wDesc, d_kernel, convDesc, perf.algo, workspace, perf.memory, &beta, yDesc, d_output));
                                CUDA_CHECK(cudaEventRecord(stop));
                                CUDA_CHECK(cudaEventSynchronize(stop));

                                float ms;
                                CUDA_CHECK(cudaEventElapsedTime(
                                    &ms, start, stop));
                                best_ms = std::min(best_ms, ms);
                            }
                            
                            double FLOPs = 2.0 * 1 * 1 * outD * outH * outW * KD * KR * KC;
                            double TFLOPs = FLOPs / (best_ms * 1e6);
                            file << D << "," << H << "," << W << "," << KD << "," << KR << "," << KC << "," << best_ms << "," << TFLOPs << "\n";
                            // std::cout << "D=" << D << " H=" << H << " W=" << W << " KD=" << KD << " KR=" << KR << " KC=" << KC
                            //           << " Time=" << best_ms << " ms"
                            //           << " Throughput=" << TFLOPs << " TFLOPs" << std::endl;
                            if (workspace)
                                CUDA_CHECK(cudaFree(workspace));
                            CUDA_CHECK(cudaFree(d_kernel));
                            CUDA_CHECK(cudaFree(d_output));
                        }
                    }
                }
                CUDA_CHECK(cudaFree(d_input));
            }
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
