//
// Created by Brian Rodriguez on 10/20/23.
//
#include <immintrin.h>
#include <mlkl/avx/operators/gemm.h>

namespace ml::operators::avx {
void inline sgemm_8x8(const float *__restrict__ a,
                      const float alpha,
                      const float *__restrict__ b,
                      const float beta,
                      float *__restrict__ c,
                      size_t M, size_t N, size_t K) {
  __m256 alpha_v = _mm256_set1_ps(alpha);
  __m256 beta_v = _mm256_set1_ps(beta);

  __m256 b0 = _mm256_load_ps(b);
  __m256 b1 = _mm256_load_ps(&b[K]);
  __m256 b2 = _mm256_load_ps(&b[K * 2]);
  __m256 b3 = _mm256_load_ps(&b[K * 3]);
  __m256 b4 = _mm256_load_ps(&b[K * 4]);
  __m256 b5 = _mm256_load_ps(&b[K * 5]);
  __m256 b6 = _mm256_load_ps(&b[K * 6]);
  __m256 b7 = _mm256_load_ps(&b[K * 7]);

  for (size_t i = 0; i < 8; ++i) {
    __m256 av = _mm256_load_ps(&a[M * i]);
    __m256 cv = _mm256_load_ps(&c[M * i]);

    auto v = _mm256_add_ps(cv, _mm256_mul_ps(av, b0));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b1));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b2));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b3));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b4));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b5));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b6));
    v = _mm256_add_ps(v, _mm256_mul_ps(av, b7));

    v = _mm256_add_ps(_mm256_mul_ps(alpha_v, v), _mm256_mul_ps(cv, beta_v));
    _mm256_store_ps(&c[M * i], v);
  }
}

void sgemm(const float *__restrict__ a, const float alpha,
           const float *__restrict__ b, const float beta, float *__restrict__ c,
           size_t M, size_t N, size_t K) {

  constexpr size_t kBlockSize = 16;

  for (size_t m = 0; m < M; m += kBlockSize) {
    for (size_t bm = 0; bm < kBlockSize; bm += 8) {
      for (size_t k = 0; k < K; k += kBlockSize) {
        for (size_t bk = 0; bk < kBlockSize; bk += 8) {
          for (size_t n = 0; n < N; n += kBlockSize) {
            for (size_t bn = 0; bn < kBlockSize; bn += 8) {
              sgemm_8x8(
                &a[(m + bm) * K + (k + bk)],
                alpha,
                &b[(k + bk) * N + (n + bn)],
                beta,
                &c[(m + bm) * N + (n + bn)],
                M, N, K);
            }
          }
        }
      }
    }
  }
}
}// namespace ml::operators::avx