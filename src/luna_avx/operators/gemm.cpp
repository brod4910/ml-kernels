//
// Created by Brian Rodriguez on 10/20/23.
//
#include <immintrin.h>
#include "luna_avx/operators/gemm.h"

namespace luna::operators::avx {
float inline reduce_gemm(__m256 &vec) {
  __m128 lo = _mm256_castps256_ps128(vec);
  __m128 hi = _mm256_extractf128_ps(vec, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuffle = _mm_movehdup_ps(lo);
  __m128 sum = _mm_add_ps(lo, shuffle);
  shuffle = _mm_movehl_ps(shuffle, sum);
  sum = _mm_add_ss(sum, shuffle);
  return _mm_cvtss_f32(sum);
}

void inline sgemm_4x8(const float *__restrict__ a, const float alpha,
               const float *__restrict__ b, const float beta, float *__restrict__ c,
               size_t M, size_t N, size_t K) {
  __m256 b0 = _mm256_load_ps(b);
  __m256 b1 = _mm256_load_ps(&b[K * 2]);
  __m256 b2 = _mm256_load_ps(&b[K * 3]);
  __m256 b3 = _mm256_load_ps(&b[K * 4]);


  for (int i = 0; i < 4; ++i) {
    __m256 av = _mm256_load_ps(&a[M * i]);
    __m256 c0 = _mm256_set1_ps(0.0);
    __m256 c1 = _mm256_set1_ps(0.0);
    __m256 c2 = _mm256_set1_ps(0.0);
    __m256 c3 = _mm256_set1_ps(0.0);

    c0 = _mm256_add_ps(c0, _mm256_mul_ps(av, b0));
    c1 = _mm256_add_ps(c1, _mm256_mul_ps(av, b1));
    c2 = _mm256_add_ps(c2, _mm256_mul_ps(av, b2));
    c3 = _mm256_add_ps(c3, _mm256_mul_ps(av, b3));

    c[(M * i) + 0] = reduce_gemm(c0);
    c[(M * i) + 1] = reduce_gemm(c1);
    c[(M * i) + 2] = reduce_gemm(c2);
    c[(M * i) + 3] = reduce_gemm(c3);
  }
}

void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta, float *__restrict__ c,
           size_t M, size_t N, size_t K) {

  constexpr size_t kBlockSize = 16;

  for (size_t m = 0; m < M; m += kBlockSize) {
    for (size_t n = 0; n < N; n += kBlockSize) {
      for (size_t k = 0; k < K; k += kBlockSize) {
        for (size_t ms = 0; ms < kBlockSize; ms += 8) {
          for (size_t ns = 0; ns < kBlockSize; ns += 8) {
            for (size_t ks = 0; ks < kBlockSize; ks += 4) {
              sgemm_4x8(&a[(m + ms) * K + k],
                        alpha,
                        &b[(n + ns) * K + k],
                        beta,
                        &c[(m + ms) * N + (n + ns)],
                        M, N, K);
            }
          }
        }
      }
    }
  }
}
}// namespace luna::operators::avx