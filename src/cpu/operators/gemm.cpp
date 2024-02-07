//
// Created by Brian Rodriguez on 8/26/23.
//
#include "luna_cpu/operators/gemm.h"

namespace ml::operators::cpu {
void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta,
           float *c, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float dot_product = 0.0;
      for (size_t k = 0; k < K; ++k) {
        dot_product += a[m * K + k] * b[n * K + k];
      }
      float c_v = c[n * M + m];
      c[n * M + m] = alpha * dot_product + beta * c_v;
    }
  }
}
}// namespace ml::operators::cpu