//
// Created by Brian Rodriguez on 10/20/23.
//
#include <immintrin.h>
#include "luna_avx/operators/gemm.h"

namespace luna::operators::avx {
void inline sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta, float *c,
           size_t M, size_t N, size_t K) {

  constexpr size_t kBlockSize = 16;

  for (size_t m = 0; m < M; m += kBlockSize) {
    for (size_t n = 0; n < N; n += kBlockSize) {
      for (size_t k = 0; k < K; k += kBlockSize) {
        __m256 dot_product = _mm256_set1_ps(0.0);
        for (size_t ms = 0; ms < kBlockSize; ms += 8) {
          for (size_t ns = 0; ns < kBlockSize; ns += 8) {
            for (size_t ks = 0; ks < kBlockSize; ks += 8) {
              __m256 av = _mm256_load_ps(&a[(m + ms) * K + (k + ks)]);
              __m256 bv = _mm256_load_ps(&b[(n + ns) * K + (k + ks)]);
              _mm256_add_ps(dot_product, _mm256_mul_ps(av, bv));
            }
          }
        }
      }
    }
  }
}
}// namespace luna::operators::avx