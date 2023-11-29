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
        for (size_t ms = 0; ms < kBlockSize; ms += 8) {
          for (size_t ns = 0; ns < kBlockSize; ns += 8) {

            __m256 m_v = _mm256_load_ps(&a[(m + ms) * K + k]);
            __m256 n_v = _mm256_load_ps(&b[(n + ns) * K + k]);
            __m256 c_v = _mm256_add_ps(_mm256_set1_ps(0.0), _mm256_mul_ps(m_v, n_v));
            // Probably don't need this loop since we can just manually unroll it
            // if our block size is small.
            for (size_t ks = 8; ks < kBlockSize; ks += 8) {
              m_v = _mm256_load_ps(&a[(m + ms) * K + (k + ks)]);
              n_v = _mm256_load_ps(&b[(n + ns) * K + (k + ks)]);
              c_v = _mm256_add_ps(c_v, _mm256_mul_ps(m_v, n_v));
            }
            __m128 lo = _mm256_castps256_ps128(c_v);
            __m128 hi = _mm256_extractf128_ps(c_v, 1);
            lo = _mm_add_ps(lo, hi);
            __m128 shuffle = _mm_movehdup_ps(lo);
            __m128 sum = _mm_add_ps(lo, shuffle);
            shuffle = _mm_movehl_ps(shuffle, sum);
            sum = _mm_add_ss(sum, shuffle);
            c[(m + ms) * N + (n + ns)] = _mm_cvtss_f32(sum);
          }
        }
      }
    }
  }
}
}// namespace luna::operators::avx