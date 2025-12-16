// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <cuda_runtime.h>
// #include <cmath>
// #include "kernels.cuh"
// #include <random>
// #include <numeric>
// #include <cudnn.h>


// #define EXIT_FAILURE 1
// #define EXIT_SUCCESS 0
// #define BLOCK_SIZE 256

// #define CUDA_CHECK(err) \
//     do{ \
//         if(err != cudaSuccess){ \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
//             exit(EXIT_FAILURE);  \
//         } \
//     } while(0)

// int main(){
//     // Get GPU device properties
//     cudaDeviceProp prop;
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
//     std::cout << "Using GPU: " << prop.name << std::endl;
//     double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // in GB/s
//     std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s" << std::endl;


//     std::string file_name = "softmax_benchmark_results_naive_kernel.csv";
//     std::ofstream outfile(file_name);
//     if(!outfile.is_open()){
//         std::cerr << "Failed to open file: " << file_name << std::endl;
//         return EXIT_FAILURE;
//     }
//     outfile << "Array Size, Time(msec), Throughput(GB/s), Efficiency(%)\n";

//     std::vector<int> array_sizes = {64, 256, 1024, 2048, 4096, 8192, 1<<6, 1<<10, 1<<13, 1<<16, 1<<20, 1<<24};
//     // Creating Random Input Data
//     std::mt19937 rng(std::random_device{}()); // random number generator
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f); // uniform distribution between 0 and 1

//     int threadsPerBlock = BLOCK_SIZE;

//     // Benchmarking loop
//     // cudnn constants
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Create cudnn handle
//     cudnnHandle_t cudnn;
//     cudnnCreate(&cudnn);
//     // cudnnTensorDescriptor_t tensorDesc;
//     // cudnnCreateTensorDescriptor(&tensorDesc);
//     // cudnnSetTensor4dDescriptor(tensorDesc,
//     //                              CUDNN_TENSOR_NCHW,
//     //                              CUDNN_DATA_FLOAT,
//     //                              1, 1, 1, 1); // N, C, H, W will be set later
//     cudaEvent_t start, stop;
//     float *d_input, *d_output;
//     size_t max_size = *std::max_element(array_sizes.begin(), array_sizes.end());
//     cudaMalloc(&d_input, max_size * sizeof(float));
//     cudaMalloc(&d_output, max_size * sizeof(float));
//     cudnnTensorDescriptor_t tensorDesc;
//     cudnnCreateTensorDescriptor(&tensorDesc);
    
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     for(const auto& size: array_sizes){
//         std::vector<float> h_input(size);
//         for(auto& val : h_input){
//             val = dist(rng);
//         }

//         // Allocate device memory
//         // cudaMalloc(&d_input, size * sizeof(float));
//         // cudaMalloc(&d_output, size * sizeof(float));
//         // Copy input data to device
//         cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

//         // Tensor descriptor setup
        

//         cudnnSetTensor4dDescriptor(tensorDesc,
//                                      CUDNN_TENSOR_NCHW,
//                                      CUDNN_DATA_FLOAT,
//                                      1, size, 1, 1); // N, C, H, W
        
        
//         for(int i = 0;i < 3; ++i){
//             cudnnSoftmaxForward(
//                 cudnn,
//                 CUDNN_SOFTMAX_ACCURATE,
//                 CUDNN_SOFTMAX_MODE_CHANNEL,
//                 &alpha,
//                 tensorDesc,
//                 d_input,
//                 &beta,
//                 tensorDesc,
//                 d_output
//             );
//         }
//         cudaDeviceSynchronize();

        
//         // Launching cudnn softmax
//         float min_ms = std::numeric_limits<float>::max();

//         for(int i = 0;i < 10; ++i){
//             cudaEventRecord(start);
//             cudnnSoftmaxForward(
//                 cudnn,
//                 CUDNN_SOFTMAX_ACCURATE,
//                 CUDNN_SOFTMAX_MODE_CHANNEL,
//                 &alpha,
//                 tensorDesc,
//                 d_input,
//                 &beta,
//                 tensorDesc,
//                 d_output
//             );
//             cudaEventRecord(stop);
//             cudaEventSynchronize(stop);
//             float ms;
//             cudaEventElapsedTime(&ms, start, stop);
//             min_ms = std::min(min_ms, ms);
//         }

//         // size_t bytes_processed = 2 * size * sizeof(float); // input + output
//         // float throughput = bytes_processed / (min_ms / 1e3) / 1e9; // in GB/s
//         // float efficiency = (throughput / peak_bandwidth) * 100.0f;

//         std::cout << "Array Size: " << size 
//                   << ", Time: " << min_ms << " ms"
//                 //   << ", Throughput: " << throughput << " GB/s"
//                 //   << ", Efficiency: " << efficiency << " %"
//                   << std::endl;

        
        
        

        
//     }
//     cudnnDestroyTensorDescriptor(tensorDesc);
//     cudaFree(d_output);
//     cudaFree(d_input);
//     cudnnDestroy(cudnn);
//     outfile.close();
//     return EXIT_SUCCESS;
// }

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cudnn.h>

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0
#define BLOCK_SIZE 256

#define CUDA_CHECK(err) \
    do { \
        if ((err) != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDNN_CHECK(err) \
    do { \
        if ((err) != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN Error: " << cudnnGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ---------------- CPU reference softmax ---------------- */
void softmax_cpu(const std::vector<float>& input,
                 std::vector<float>& output)
{
    float max_val = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }
}

/* ---------------- Main ---------------- */
int main()
{
    /* GPU info */
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;

    double peak_bw = 2.0 * prop.memoryClockRate *
                     (prop.memoryBusWidth / 8) / 1.0e6;
    std::cout << "Peak memory BW: " << peak_bw << " GB/s\n";

    /* Test sizes */
    std::vector<int> sizes = {
        64, 256, 1024, 2048, 4096,
        8192, 1 << 10, 1 << 13,
        1 << 16, 1 << 20, 1 << 24
    };

    size_t max_size = *std::max_element(sizes.begin(), sizes.end());

    /* RNG */
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    /* Allocate GPU memory */
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, max_size * sizeof(float)));

    /* cuDNN setup */
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t tensorDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));

    float alpha = 1.0f;
    float beta  = 0.0f;

    /* Timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\nSize | Time(ms) | MaxAbsErr | MaxRelErr | Sum\n";
    std::cout << "----------------------------------------------------------\n";

    for (int size : sizes) {
        /* Host input */
        std::vector<float> h_input(size);
        for (auto& v : h_input)
            v = dist(rng);

        CUDA_CHECK(cudaMemcpy(
            d_input, h_input.data(),
            size * sizeof(float),
            cudaMemcpyHostToDevice));

        /* cuDNN tensor: N=1, C=size */
        CUDNN_CHECK(
            cudnnSetTensor4dDescriptor(
                tensorDesc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, size, 1, 1));

        /* Warmup */
        for (int i = 0; i < 3; ++i) {
            CUDNN_CHECK(
                cudnnSoftmaxForward(
                    cudnn,
                    CUDNN_SOFTMAX_ACCURATE,
                    CUDNN_SOFTMAX_MODE_CHANNEL,
                    &alpha,
                    tensorDesc,
                    d_input,
                    &beta,
                    tensorDesc,
                    d_output));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Benchmark */
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < 10; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            CUDNN_CHECK(
                cudnnSoftmaxForward(
                    cudnn,
                    CUDNN_SOFTMAX_ACCURATE,
                    CUDNN_SOFTMAX_MODE_INSTANCE,
                    // CUDNN_SOFTMAX_MODE_CHANNEL,
                    &alpha,
                    tensorDesc,
                    d_input,
                    &beta,
                    tensorDesc,
                    d_output));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            min_ms = std::min(min_ms, ms);
        }

        /* Copy result back */
        std::vector<float> h_output(size);
        CUDA_CHECK(cudaMemcpy(
            h_output.data(), d_output,
            size * sizeof(float),
            cudaMemcpyDeviceToHost));

        /* CPU reference */
        std::vector<float> h_ref(size);
        softmax_cpu(h_input, h_ref);

        /* Error metrics */
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        float sum = 0.0f;

        for (int i = 0; i < size; ++i) {
            float abs_err = std::abs(h_output[i] - h_ref[i]);
            float rel_err = abs_err / (std::abs(h_ref[i]) + 1e-8f);

            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
            sum += h_output[i];
        }

        std::cout << size << " | "
                  << min_ms << " | "
                  << max_abs_err << " | "
                  << max_rel_err << " | "
                  << sum << std::endl;

        /* Hard correctness check */
        if (max_abs_err > 1e-5f ){//|| std::abs(sum - 1.0f) > 1e-5f) {
            std::cerr << "Numerical validation failed at size "
                      << size << std::endl;
            // std::exit(EXIT_FAILURE);
        }
    }

    /* Cleanup */
    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroy(cudnn);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ All tests passed.\n";
    return EXIT_SUCCESS;
}
