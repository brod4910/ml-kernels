//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <iostream>
#include <immintrin.h>

#include <luna_avx/operators/transpose.h>

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
  auto *a = static_cast<float *>(_mm_malloc(M * N * sizeof(float), 32));
  auto *b = static_cast<float *>(_mm_malloc(N * M * sizeof(float), 32));
  arange_matrix(a, M * N);
  initialize_matrix(b, N * M, 0);
  luna::operators::avx::transpose(a, b, M, N);
  print_matrix(b, M, N);
//  __m256 row_0 = _mm256_load_ps(a);
//  __m256 row_1 = _mm256_load_ps(a + 8);
//  __m256 row_2 = _mm256_load_ps(a + 16);
//  __m256 row_3 = _mm256_load_ps(a + 24);
//  __m256 row_4 = _mm256_load_ps(a + 32);
//  __m256 row_5 = _mm256_load_ps(a + 40);
//  __m256 row_6 = _mm256_load_ps(a + 48);
//  __m256 row_7 = _mm256_load_ps(a + 56);
//
//  __m256 r0_r1_lo = _mm256_unpacklo_ps(row_0, row_1);
//  __m256 r2_r3_lo = _mm256_unpacklo_ps(row_2, row_3);
//  __m256 r4_r5_lo = _mm256_unpacklo_ps(row_4, row_5);
//  __m256 r6_r7_lo = _mm256_unpacklo_ps(row_6, row_7);
//  __m256 r0_r1_hi = _mm256_unpackhi_ps(row_0, row_1);
//  __m256 r2_r3_hi = _mm256_unpackhi_ps(row_2, row_3);
//  __m256 r4_r5_hi = _mm256_unpackhi_ps(row_4, row_5);
//  __m256 r6_r7_hi = _mm256_unpackhi_ps(row_6, row_7);
//
//  auto shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b01000100);
//  auto shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b01000100);
//
//  auto t0 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
//  auto t4 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);
//
//  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_lo, r2_r3_lo, 0b11101110);
//  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_lo, r6_r7_lo, 0b11101110);
//
//  auto t1 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
//  auto t5 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);
//
//  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_hi, r2_r3_hi, 0b01000100);
//  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b01000100);
//
//  auto t2 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
//  auto t6 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);
//
//  shf_r0_r3 = _mm256_shuffle_ps(r0_r1_hi, r2_r3_hi, 0b11101110);
//  shf_r4_r7 = _mm256_shuffle_ps(r4_r5_hi, r6_r7_hi, 0b11101110);
//
//  auto t3 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00100000);
//  auto t7 = _mm256_permute2f128_ps(shf_r0_r3, shf_r4_r7, 0b00110001);
//
//  print_matrix(reinterpret_cast<float*>(&t0), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t1), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t2), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t3), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t4), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t5), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t6), 1, 8);
//  print_matrix(reinterpret_cast<float*>(&t7), 1, 8);


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
