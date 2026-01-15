#include <cudnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cuda_runtime.h>

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

int main(){
    /* GPU Info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU Device Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    double peak_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // in GB/s
    std::cout << "Peak Memory Bandwidth: " << peak_bw << " GB/s" << std::endl;

    // Random init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    const int input_sizes[] = {1500000};
    const int kernel_sizes[] = {31, 63, 127, 255, 1023, 2047};

    /* cuDNN setup */
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::ofstream result_file("cudnn_1d_convolution_benchmark_1.csv");
    result_file << "Input Size,Kernel Size,Time (ms)" << std::endl;

    for(int N: input_sizes){
        // Host input allocation and initialization
        std::vector<float> h_input(N);
        for(int i = 0; i < N; ++i){
            h_input[i] = dis(gen);
        }

        // Device input  
        float *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        for(int K : kernel_sizes){
            if( K >= N ) continue; // kernel size must be less than input size
            int outSize = N - K + 1;
            // Host output allocation
            std::vector<float> h_output(outSize, 0.0f);
            // Device output allocation
            float *d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outSize * sizeof(float)));
            // Device filter allocation and initialization
            std::vector<float> h_filter(K);
            for(int i = 0; i < K; ++i){
                h_filter[i] = dis(gen);
            }
            float *d_filter;
            CUDA_CHECK(cudaMalloc(&d_filter, K * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            // Set tensor and filter descriptors
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, outSize));
            CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, K));
            // Set convolution descriptor
            CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                       0, 0, // pad height, width
                                                       1, 1, // vertical, horizontal stride
                                                       1, 1, // dilation height, width
                                                       CUDNN_CROSS_CORRELATION,
                                                       CUDNN_DATA_FLOAT));
            // Find convolution algorithm
            cudnnConvolutionFwdAlgoPerf_t perf;
            int returnedAlgoCount = 0;

            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn,
                xDesc,
                filterDesc,
                convDesc,
                yDesc,
                1, // max number of algorithms to return
                &returnedAlgoCount,
                &perf
            ));
            cudnnConvolutionFwdAlgo_t algo = perf.algo;
            // std::cout << "Selected algo enum = " << perf.algo << std::endl;
            // Workspace 
            size_t workspaceSize = perf.memory;
            void* d_workspace = nullptr;
            if(workspaceSize > 0){
                CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
            }
            float alpha = 1.0f, beta = 0.0f;
            // Warm-up
            for(int i = 0; i < 5; i++){
                CUDNN_CHECK(cudnnConvolutionForward(
                    cudnn,
                    &alpha,
                    xDesc,
                    d_input,
                    filterDesc,
                    d_filter,
                    convDesc,
                    algo,
                    d_workspace,
                    workspaceSize,
                    &beta,
                    yDesc,
                    d_output
                ));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            // Timing
            float min_ms = std::numeric_limits<float>::max();
            for(int i = 0; i < 10; i++){
                CUDA_CHECK(cudaEventRecord(start));
                CUDNN_CHECK(cudnnConvolutionForward(
                    cudnn,
                    &alpha,
                    xDesc,
                    d_input,
                    filterDesc,
                    d_filter,
                    convDesc,
                    algo,
                    d_workspace,
                    workspaceSize,
                    &beta,
                    yDesc,
                    d_output
                ));
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                min_ms = std::min(min_ms, ms);
            }
            
            result_file << N << "," << K << "," << min_ms << std::endl;
            // std::cout << "Input Size: " << N << ", Kernel Size: " << K << ", Time: " << min_ms << " ms" << std::endl;
            // Clean up
            if(workspaceSize > 0){
                CUDA_CHECK(cudaFree(d_workspace));
            }
            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaFree(d_filter));
        }
        CUDA_CHECK(cudaFree(d_input));
    }
    result_file.close();
    /* CleanUp */
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroy(cudnn);
    
    return 0;
}


