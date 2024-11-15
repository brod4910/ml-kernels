//
// Created by Brian Rodriguez on 8/26/23.
//
#include <mlkl/cpu/operators/gemm.h>

namespace ml::operators::cpu {
void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta,
           float *c, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t k = 0; k < K; ++k) {
        c[m * N + n] += a[m * K + k] * b[k * N + n];
      }
    }
  }
}
}// namespace ml::operators::cpu