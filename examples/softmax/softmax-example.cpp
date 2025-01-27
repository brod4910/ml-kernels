#ifdef __CUDA__
#include "softmax_cuda.h"
#endif

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
  for (const auto &[M, N] : matrix_sizes) {
    std::cout << "CPU" << std::endl;
    softmax_cpu(M, N);
#ifdef __CUDA__
    std::cout << "CUDA" << std::endl;
    softmax_cuda(M, N);
#endif
  }
}
