//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <iostream>
#include <immintrin.h>

#include <luna_avx/operators.h>

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
  auto *a = static_cast<float *>(_mm_malloc(M * K * sizeof(float), 32));
  auto *b = static_cast<float *>(_mm_malloc(N * K * sizeof(float), 32));
  auto *b_T = static_cast<float *>(_mm_malloc(N * K * sizeof(float), 32));
  auto *c = static_cast<float *>(_mm_malloc(M * N * sizeof(float), 32));

  initialize_matrix(a, M * K, 1);
  initialize_matrix(b, N * K, 2);
  luna::operators::avx::transpose(b, b_T, M, N);
  luna::operators::avx::sgemm(a, alpha, b_T, beta, c, M, N, K);
  print_matrix(c, M, N);
  _mm_free(a);
  _mm_free(b);
  _mm_free(b_T);
  _mm_free(c);
}
