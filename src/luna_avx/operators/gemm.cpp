//
// Created by Brian Rodriguez on 10/20/23.
//

#include "luna_avx/operators/gemm.h"

namespace luna::operators::avx {
void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta, float *c,
           size_t M, size_t N, size_t K) {

  constexpr size_t kSize = 8;
  size_t Ms = M / kSize;
  size_t Ns = N / kSize;
  size_t Ks = K / kSize;

  for (size_t m = 0; m < M; ++m) {
    for(size_t k = 0; k < K; ++k) {
      __m256 mv = _mm256_set1_ps(a[m * K + k]);
      for(size_t n = 0; n < N; n += kSize) {
        __m256 nv = _mm256_load_ps(&b[k * N + n]);
        __m256 res = _mm256_mul_ps(mv, nv);
//        c[m * N + n] = _mm256_add_ps(res, c[m * N + n]);
      }
    }
  }

  for (size_t m = 0; m < M; ++m) {
    for (size_t ms = 0; ms < kSize; ++ms) {
      __m256 mv = _mm256_set1_ps(a[m * K + ms]);
      for (size_t n = 0; n < N; n+=kSize) {
        for (size_t ns = 0; ns < kSize; ++ns) {
          for (size_t k = 0; k < K; ++k) {
            __m256 bv = _mm256_load_ps(&b[n * K + k]);
            __m256 cv = _mm256_set1_ps(c[n * K + k]);
            __m256 x = _mm256_mul_ps(mv, bv);
            __m256 y = _mm256_add_ps(x, cv);
            _mm256_storeu_ps(c + n * K + k, x);
          }
        }
      }
    }
  }

//  __m128 alpha_128 = _mm_set1_ps(alpha);
//
//  for (size_t m = 0; m < M; ++m) {
//    for (size_t n = 0; n < N; ++n) {
//      __m256 dot_product_v = _mm256_setzero_ps();
//      for (size_t k = 0; k < Ks; ++k) {
//        __m256 a_v = a[m * K + k];
//        __m256 b_v = b[n * K + k];
//        __m256 x = _mm256_mul_ps(a_v, b_v);
//        dot_product_v = _mm256_add_ps(dot_product_v, x);
//      }
//      // extract the high 128 bits and add them to the low 128 bits
//      __m128 hv = _mm256_extractf128_ps(dot_product_v, 1);
//      __m128 res = _mm_add_ps(hv, _mm256_castps256_ps128(dot_product_v));
//      // extract the high 64 bits and add them to the low 64 bits
//      res = _mm_movehl_ps(res, res);
//      res = _mm_add_ss(res, _mm_movehdup_ps(res));
//      res = _mm_mul_ss(res, alpha_128);
//      c[m * N + n] = _mm256_insertf128_ps(dot_product_v, res, 0);
//    }
//  }
}
}// namespace luna::operators::avx