//
// Created by Brian Rodriguez on 8/26/23.
//
#include <mlkl/cpu/operators/gemm.h>

namespace ml::operators::cpu {
/*
  Cache-aware implementation of sgemm
*/
void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta,
           float *c, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t n = 0; n < N; ++n) {
        c[m * N + n] += a[m * N + k] * b[k * N + n];
      }
      // c[n * M + m] = alpha * dot_product + beta * c_v;
    }
  }
}
}// namespace ml::operators::cpu