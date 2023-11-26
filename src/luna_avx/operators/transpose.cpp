//
// Created by Brian Rodriguez on 10/29/23.
//

#include <immintrin.h>
#include <luna_avx/operators/transpose.h>

namespace luna::operators::avx {
void inline transpose_8x8(const float *a, float *b, size_t M, size_t N) {
  __m256 row_0 = _mm256_load_ps(a);
  __m256 row_1 = _mm256_load_ps(&a[M]);
  __m256 row_2 = _mm256_load_ps(&a[M * 2]);
  __m256 row_3 = _mm256_load_ps(&a[M * 3]);
  __m256 row_4 = _mm256_load_ps(&a[M * 4]);
  __m256 row_5 = _mm256_load_ps(&a[M * 5]);
  __m256 row_6 = _mm256_load_ps(&a[M * 6]);
  __m256 row_7 = _mm256_load_ps(&a[M * 7]);

  __m256 r0_r1_lo = _mm256_unpacklo_ps(row_0, row_1);
  __m256 r0_r1_hi = _mm256_unpackhi_ps(row_0, row_1);
  __m256 r2_r3_lo = _mm256_unpacklo_ps(row_2, row_3);
  __m256 r2_r3_hi = _mm256_unpackhi_ps(row_2, row_3);
  __m256 r4_r5_lo = _mm256_unpacklo_ps(row_4, row_5);
  __m256 r4_r5_hi = _mm256_unpackhi_ps(row_4, row_5);
  __m256 r6_r7_lo = _mm256_unpacklo_ps(row_6, row_7);
  __m256 r6_r7_hi = _mm256_unpackhi_ps(row_6, row_7);

  auto shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b01000100);
  auto shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b01000100);

  auto t0 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
  auto t4 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);

  _mm256_store_ps(b, t0);
  _mm256_store_ps(&b[N * 4], t4);

  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b11101110);
  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b11101110);

  auto t1 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
  auto t5 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);

  _mm256_store_ps(&b[N], t1);
  _mm256_store_ps(&b[N * 5], t5);

  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_hi, r2_r3_hi, 0b01000100);
  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b01000100);

  auto t2 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
  auto t6 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);

  _mm256_store_ps(&b[N * 2], t2);
  _mm256_store_ps(&b[N * 6], t6);

  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_hi, r2_r3_hi, 0b11101110);
  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b11101110);

  auto t3 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
  auto t7 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);

  _mm256_store_ps(&b[N * 3], t3);
  _mm256_store_ps(&b[N * 7], t7);
}

void transpose(const float *__restrict__ a, float *__restrict__ b, size_t M, size_t N) {
  constexpr size_t kBlockSize = 16;

  for (size_t m = 0; m < M; m += kBlockSize) {
    for (size_t n = 0; n < N; n += kBlockSize) {
      for (size_t bs = 0; bs < kBlockSize; bs += 8) {
        size_t mm = m * N + n + bs;
        size_t nn = (n + bs) * M + m;
        transpose_8x8(&a[m * N + n + bs], &b[(n + bs) * M + m], M, N);
      }
    }
  }
}
}
