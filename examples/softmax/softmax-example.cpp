// #ifdef __AVX2__
// #include "sgemm_avx.h"
// #endif

// #ifdef __CUDA__
// #include "sgemm_cuda.h"
// #endif

#include "softmax_cpu.h"
#include <tuple>
#include <vector>

int main() {
  // clang-format off
  std::vector<std::tuple<int, int>> matrix_sizes = {
    {256, 256},
    {512, 512},
    {1024, 1024},
    {2048, 2048}
  };
  // clang-format on

  float alpha = 1.0;
  float beta = 0.0;
  for (const auto &[M, N] : matrix_sizes) {
    std::cout << "CPU" << std::endl;
    softmax_cpu(M, N);
    // #ifdef __AVX2__
    //     std::cout << "AVX2" << std::endl;
    //     sgemm_avx(M, N, K, alpha, beta);
    // #endif
    // #ifdef __CUDA__
    //     std::cout << "CUDA" << std::endl;
    //     softmax_cuda(M, N, K, alpha, beta);
    // #endif
  }
}
