//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <iostream>
#include <luna_avx/operators/gemm.h>

void initialize_matrix(float *matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

void arange_matrix(float *matrix, size_t size) {
  for (size_t i = 0; i < size; i += 1) {
    matrix[i] = float(i);
  }
}

void print_matrix(const float *matrix, size_t M, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

void sgemm_avx(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto *a = static_cast<float *>(_mm_malloc(8 * 8 * sizeof(float), 32));
  auto *b = static_cast<float *>(_mm_malloc(8 * 8 * sizeof(float), 32));
  arange_matrix(a, 64);
  initialize_matrix(b, 64, 0);

  __m256 row_0 = _mm256_load_ps(a);
  __m256 row_1 = _mm256_load_ps(a + 8);
  __m256 row_2 = _mm256_load_ps(a + 16);
  __m256 row_3 = _mm256_load_ps(a + 24);
  __m256 row_4 = _mm256_load_ps(a + 32);
  __m256 row_5 = _mm256_load_ps(a + 40);
  __m256 row_6 = _mm256_load_ps(a + 48);
  __m256 row_7 = _mm256_load_ps(a + 56);

  __m256 r0_r1_lo = _mm256_unpacklo_ps(row_0, row_1);
  __m256 r2_r3_lo = _mm256_unpacklo_ps(row_2, row_3);
  auto shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b01001101);
  auto blend_r0_r3 = _mm256_blend_ps(r0_r1_lo, shf_r0_r3, 0b11001100);
  __m256 r4_r5_lo = _mm256_unpacklo_ps(row_4, row_5);
  __m256 r6_r7_lo = _mm256_unpacklo_ps(row_6, row_7);
  auto shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b01001101);
  auto blend_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, shf_r4_r7, 0b11001100);
  auto t1 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00100000);
  auto t4 = _mm256_permute2f128_ps(blend_r0_r3, blend_r4_r7, 0b00110001);

  print_matrix(reinterpret_cast<float*>(&t1), 1, 8);
  print_matrix(reinterpret_cast<float*>(&t4), 1, 8);

  __m256 r0_r1_hi = _mm256_unpackhi_ps(row_0, row_1);
  __m256 r2_r3_hi = _mm256_unpackhi_ps(row_2, row_3);
  __m256 r4_r5_hi = _mm256_unpackhi_ps(row_4, row_5);
  __m256 r6_r7_hi = _mm256_unpackhi_ps(row_6, row_7);

//  auto r0 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0x4E);
//  std::cout << "Shuffle r0-lo & r2-lo: ";
//  print_matrix(reinterpret_cast<float*>(&r0), 1, 8);
//  r0 = _mm256_blend_ps(r0_r1_lo, r0, 0xCC);
//  std::cout << "Blend shf-0-lo & r0-lo: ";
//  print_matrix(reinterpret_cast<float*>(&r0), 1, 8);
//  auto r1 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0x4E);
//  std::cout << "Shuffle r4-lo & r6-lo: ";
//  print_matrix(reinterpret_cast<float*>(&r1), 1, 8);
//  r1 = _mm256_blend_ps(r4_r5_lo, r1, 0xCC);
//  std::cout << "Blend shf-4-lo & r1-lo: ";
//  print_matrix(reinterpret_cast<float*>(&r1), 1, 8);
//  auto z0 = _mm256_permute2f128_ps(r0, r1, 0x20);
//  std::cout << "Permute: ";
//  print_matrix(reinterpret_cast<float*>(&z0), 1, 8);
  //  auto *a = static_cast<float *>(_mm_malloc(M * K * sizeof(float), 32));
//  auto *b = static_cast<float *>(_mm_malloc(N * K * sizeof(float), 32));
//  auto *c = static_cast<float *>(_mm_malloc(M * N * sizeof(float), 32));
//  initialize_matrix(a, M * K, 1);
//  initialize_matrix(b, N * K, 2);
//  initialize_matrix(c, M * N, 0);
//
//  luna::operators::avx::sgemm(a, alpha, b, beta, c, M, N, K);
//  print_matrix(reinterpret_cast<float *>(c), M, N);
//
//  _mm_free(a);
//  _mm_free(b);
//  _mm_free(c);
}
