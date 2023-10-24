//
// Created by Brian Rodriguez on 10/20/23.
//

#include "luna_avx/operators/gemm.h"

#include <immintrin.h>

namespace luna::operators::avx {
void sgemm(const __m256 *__restrict__ a, const float alpha,
           const __m256 *__restrict__ b, const float beta, __m256 *c, size_t M,
           size_t N, size_t K) {
  constexpr size_t kSize = 8;
  size_t Ms = M / kSize;
  size_t Ns = N / kSize;
  size_t Ks = K / kSize;

  __m256 alpha_256 = _mm256_set1_ps(alpha);
  __m256 beta_256 = _mm256_set1_ps(beta);

  for (size_t m = 0; m < Ms; ++m) {
    for (size_t n = 0; n < Ns; ++n) {
      __m256 dot_product_v = _mm256_setzero_ps();
      for (size_t k = 0; k < Ks; ++k) {
        __m256 a_v = a[m * K + k];
        __m256 b_v = b[n * K + k];
        __m256 c_v = c[n * M + m];
        __m256 x = _mm256_mul_ps(a_v, b_v);
        __m256 y = _mm256_mul_ps(c_v, beta_256);
        __m256 z = _mm256_fmadd_ps(x, alpha_256, y);
        dot_product_v = _mm256_add_ps(dot_product_v, z);
      }
      c[n * M + m] = dot_product_v;
    }
  }
}
} // namespace luna::operators::avx