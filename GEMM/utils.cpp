#include <vector>
#include <random>
#include <array>
#include <chrono>
#include <iostream>

#include "utils.h"

// std::vector<float> createRandomMatrix(int rows, int cols, float min_val, float max_val){
//     // static thread_local std::mt19937 generator = []{
//     //     std::random_device rd;
//     //     using seed_type = std::random_device::result_type;
//     //     std::array<seed_type, 3> seed_data{rd(), rd(), rd()};
//     //     std::seed_seq seq(seed_data.begin(), seed_data.end());
//     //     return std::mt19937(seq);
//     // }();
//     static thread_local std::mt19937 generator(1234);

//     std::uniform_real_distribution<float> distribution(min_val, max_val);
//     std::vector<float> matrix(rows * cols);
//     for(float &v : matrix){
//         v = distribution(generator);
//     }
//     return matrix;
// }
std::vector<float> createRandomMatrix(int rows, int cols,
                                      float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    std::vector<float> matrix(rows * cols);
    for (auto& x : matrix) x = dis(gen);
    return matrix;
}

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

void verify_results(const std::vector<float>& C_cpu,
                    const std::vector<float>& C_gpu,
                    int M, int N, const std::string& label)
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