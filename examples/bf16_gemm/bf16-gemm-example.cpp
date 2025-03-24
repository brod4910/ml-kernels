//
// Created by Brian Rodriguez on 8/26/23.
//
//
// Created by Brian Rodriguez on 6/17/23.
//
#include "bf16_gemm_cpu.h"
#include "bf16_gemm_cuda.h"

#include <iostream>
#include <tuple>
#include <vector>

int main() {
  // clang-format off
  std::vector<std::tuple<int, int, int>> matrix_sizes = {
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048}
  };
  // clang-format on

  float alpha = 1.0;
  float beta = 0.0;
  for (const auto &[M, N, K] : matrix_sizes) {
    std::cout << "CPU" << std::endl;
    bf16_gemm_cpu(M, N, K, alpha, beta, 1);

    std::cout << "CUDA" << std::endl;
    bf16_gemm_cuda(M, N, K, alpha, beta);
  }
}
