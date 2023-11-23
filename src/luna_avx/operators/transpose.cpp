//
// Created by Brian Rodriguez on 10/29/23.
//

#include <luna_avx/operators/transpose.h>
#include <immintrin.h>
/*
 * Transpose AVX Notes:
 *
 * Transposed 8x8 matrix
 *
 * [ 0  8 16 24 32 40 48 56]
 * [ 1  9 17 25 33 41 49 57]
 * [ 2 10 18 26 34 42 50 58]
 * [ 3 11 19 27 35 43 51 59]
 * [ 4 12 20 28 36 44 52 60]
 * [ 5 13 21 29 37 45 53 61]
 * [ 6 14 22 30 38 46 54 62]
 * [ 7 15 23 31 39 47 55 63]
 *
 * We start to by unpacking the lo and hi values from r0 & r1, r2 & r3, etc.
 *
 *                      Unpack hi-lo
 *
 * [ 0  1  2  3  4  5  6  7]    [ 0  8  1  9  4  12 5  13 ] r0 & r1 lo
 * [ 8  9 10 11 12 13 14 15]    [ 2  10 3  11 6  14 7  15 ] r0 & r1 hi
 * [16 17 18 19 20 21 22 23]    [ 16 24 17 25 20 28 21 29 ] r2 & r3 lo
 * [24 25 26 27 28 29 30 31]    [ 18 26 19 27 22 30 23 31 ] r2 & r3 hi
 * [32 33 34 35 36 37 38 39] -> [ 32 40 33 41 36 44 32 45 ] r4 & r5 lo
 * [40 41 42 43 44 45 46 47]    [ 34 42 35 43 38 46 39 47 ] r4 & r5 hi
 * [48 49 50 51 52 53 54 55]    [ 48 56 49 57 52 60 53 61 ] r6 & r7 lo
 * [56 57 58 59 60 61 62 63]    [ 50 58 51 59 54 62 55 63 ] r6 & r7 hi
 *
 * SHUFFLE: r0_r1_lo and r2_r3_lo (01 00 11 10)
 *
 * [ 0  8  1  9  4  12 5  13 ]      10 11 00 01 | 10 11 00 01
 *                              -> [ 1  9 16 24 |  5 13 20 28 ]
 * [ 16 24 17 25 20 28 21 29 ]
 *
 * BLEND: r0-r1-lo and shuffle-r[0-3]-lo (11 00 11 00)
 *
 *
 * [ 0 8  1  9  4 12  5 13 ]      0  0  1  1   0   0  1  1
 *                          ->  [ 0 8 16 24 | 4  12 20 28 ]
 * [ 1 9 16 24  5 13 20 28 ]
 *
 * SHUFFLE: r4_r5_lo and r6_r7_lo  (01 00 11 10)
 *
 * [ 32 40 33 41 36 44 32 45 ]       10 11 00 01   10 11 00 01
 *                              -> [ 33 41 48 56 | 32 45 52 60]
 * [ 48 56 49 57 52 60 53 61 ]
 *
 * BLEND: r4-r5-lo and shf r[4-7]-lo (11 00 11 00)
 *
 * [ 32 40 33 41 36 44 32 45 ]         0  0  1  1    0  0  1  1
 *                              ->  [ 32 40 48 56 | 36 44 52 60 ]
 * [ 33 41 48 56 32 45 52 60 ]
 *
 * PERMUTE: blend(r0-r1-lo & shf-r[0-3]-lo) and blend(r4-r5-lo & shf r[4-7]-lo) (00100000)
 *
 *  [ 0  8  16 24 4  12 20 28 ]        00 00         10 00
 *                              -> [ 0 8 16 24 | 32 40 48 56 ]
 *  [ 32 40 48 56 36 44 52 60 ]
 *
 * Transposed 8x8 matrix
 * [ 0  8 16 24 32 40 48 56] t0
 * [ 1  9 17 25 33 41 49 57] t1
 * [ 2 10 18 26 34 42 50 58] t2
 * [ 3 11 19 27 35 43 51 59] t3
 * [ 4 12 20 28 36 44 52 60] t4
 * [ 5 13 21 29 37 45 53 61] t5
 * [ 6 14 22 30 38 46 54 62] t6
 * [ 7 15 23 31 39 47 55 63] t7
 */


void transpose_8x8(const float* a, size_t i, float* b, size_t j) {
  __m256 row_0 = _mm256_load_ps(a);
  __m256 row_1 = _mm256_load_ps(a + 8);
  __m256 row_2 = _mm256_load_ps(a + 16);
  __m256 row_3 = _mm256_load_ps(a + 24);
  __m256 row_4 = _mm256_load_ps(a + 32);
  __m256 row_5 = _mm256_load_ps(a + 40);
  __m256 row_6 = _mm256_load_ps(a + 48);
  __m256 row_7 = _mm256_load_ps(a + 56);

  __m256 r0_r1_lo = _mm256_unpacklo_ps(row_0, row_1);
  __m256 r0_r1_hi = _mm256_unpackhi_ps(row_0, row_1);
  __m256 r2_r3_lo = _mm256_unpacklo_ps(row_2, row_3);
  __m256 r2_r3_hi = _mm256_unpackhi_ps(row_2, row_3);
  __m256 r4_r5_lo = _mm256_unpacklo_ps(row_4, row_5);
  __m256 r4_r5_hi = _mm256_unpackhi_ps(row_4, row_5);
  __m256 r6_r7_lo = _mm256_unpacklo_ps(row_6, row_7);
  __m256 r6_r7_hi = _mm256_unpackhi_ps(row_6, row_7);

  auto shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b01001110);
  auto blend_r0_r3 = _mm256_blend_ps(r0_r1_lo, shf_r0_r3, 0b11001100);
  auto shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b01001110);
  auto blend_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, shf_r4_r7, 0b11001100);
  auto t0 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00100000);
  auto t4 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00110001);
  _mm256_store_ps(b, t0);
  _mm256_store_ps(b + 32, t4);

  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_hi, r2_r3_hi, 0b01001110);
  blend_r0_r3 = _mm256_blend_ps(r0_r1_hi, shf_r0_r3, 0b11001100);
  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b01001110);
  blend_r4_r7 = _mm256_blend_ps(r4_r5_hi, shf_r4_r7, 0b11001100);
  auto t1 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00100000);
  auto t5 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00110001);

  shf_r0_r3 = _mm256_shuffle_ps(r2_r3_lo, r, 0b01001110);
  blend_r0_r3 = _mm256_blend_ps(r0_r1_hi, shf_r0_r3, 0b11001100);
  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b01001110);
  blend_r4_r7 = _mm256_blend_ps(r4_r5_hi, shf_r4_r7, 0b11001100);
  auto t1 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00100000);
  auto t5 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00110001);

}

void transpose(const float *__restrict__ a, float * __restrict__ b, size_t M, size_t N) {
  constexpr size_t kBlockSize = 16;

  for(size_t i = 0; i < M; i += kBlockSize) {
    for(size_t j = 0; j < N; j += kBlockSize) {

    }
  }
}