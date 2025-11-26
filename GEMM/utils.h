#pragma once
#include <vector>
#include <string>

// Default arguments go here
std::vector<float> createRandomMatrix(int rows, int cols, float min_val = -1.0f, float max_val = 1.0f);

void cpu_gemm(const float* A, const float* B, float* C,
              int M, int K, int N, float alpha, float beta, const float* C_init);

// Default argument only in header
void verify_results(const std::vector<float>& C_cpu,
                    const std::vector<float>& C_gpu,
                    int M, int N, const std::string& label = "");
