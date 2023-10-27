//
// Created by Brian Rodriguez on 10/20/23.
//

#include "luna_avx/operators/gemm.h"

namespace luna::operators::avx {
void sgemm(const __m256 *__restrict__ a, const float alpha,
           const __m256 *__restrict__ b, const float beta, __m256 *c,
           size_t M, size_t N, size_t K) {
  constexpr size_t kSize = 8;
  size_t Ms = M / kSize;
  size_t Ns = N / kSize;
  size_t Ks = K / kSize;

  __m128 alpha_128 = _mm_set1_ps(alpha);

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      __m256 dot_product_v = _mm256_setzero_ps();
      for (size_t k = 0; k < Ks; ++k) {
        __m256 a_v = a[m * Ks + k];
        __m256 b_v = b[n * Ks + k];
        __m256 x = _mm256_mul_ps(a_v, b_v);
        dot_product_v = _mm256_add_ps(dot_product_v, x);
      }
      __m128 hv = _mm256_extractf128_ps(dot_product_v, 1);
      __m128 res = _mm_add_ps(hv, _mm256_castps256_ps128(dot_product_v));
      res = _mm_movehl_ps(res, res);
      res = _mm_add_ss(res, _mm_movehdup_ps(res));
      res = _mm_mul_ss(res, alpha_128);
      c[n * M + m] = _mm256_insertf128_ps(dot_product_v, res, 0);
    }
  }
}
}// namespace luna::operators::avx