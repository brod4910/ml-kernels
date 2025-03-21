//
// Created by Brian Rodriguez on 8/26/23.
//
#include <mlkl/operators/cpu/gemm.h>

namespace mlkl::operators::cpu {
namespace {
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
}// namespace

void sgemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta) {
  sgemm(a->fp32_(), alpha, b->fp32_(), beta, c->fp32_(), c->shape[0], c->shape[1], a->shape[1]);
}
}// namespace mlkl::operators::cpu