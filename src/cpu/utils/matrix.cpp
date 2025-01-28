#include <iomanip>
#include <iostream>
#include <random>

#include "mlkl/cpu/utils/matrix.h"

namespace mlkl::cpu::utils {
void set_random_matrix(float *matrix, int M, int N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < M * N; ++i) {
    matrix[i] = dist(gen);
  }
}

void fill_matrix(float *matrix, int M, int N, float value) {
  for (int i = 0; i < M * N; ++i) {
    matrix[i] = value;
  }
}

void initialize_matrix_from_0_to_N(float *matrix, size_t M, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      matrix[i * N + j] = static_cast<float>(j % 128);
    }
  }
}

bool assert_correctness(float *matrix, float *ref_matrix, size_t M, size_t N, float epsilon) {
  double diff = 0.;

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int linear = m * N + n;
      diff = fabs((double) matrix[linear] - (double) ref_matrix[linear]);
      if (diff > 1e-2) {
        printf("Error: Output Matrix: %5.2f, Ref Matrix: %5.2f, (M, N): (%lu, %lu) \n", matrix[linear], ref_matrix[linear], m, n);
        return false;
      }
    }
  }
  return true;
}

}// namespace mlkl::cpu::utils