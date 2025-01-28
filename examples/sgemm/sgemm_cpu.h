//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <mlkl/operators/operators.h>

#include <chrono>
#include <iostream>
#include <memory>

void initialize_matrix_cpu(float *matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

void print_matrix_cpu(const float *matrix, size_t M, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

void sgemm_cpu(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto *a = new float[M * K];
  auto *b = new float[N * K];
  auto *b_T = new float[K * N];
  auto *c = new float[M * N];

  const int num_runs = 100;
  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    initialize_matrix_cpu(a, M * K, 1, 1);
    initialize_matrix_cpu(b, N * K, 2, 1);
    initialize_matrix_cpu(b_T, K * N, 2, 1);
    initialize_matrix_cpu(c, M * N, 0, 1);

    auto start = std::chrono::high_resolution_clock::now();
    // ml::operators::cpu::transpose(b, b_T, M, N);
    mlkl::operators::cpu::sgemm(a, alpha, b_T, beta, c, M, N, K);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    total_duration += duration.count();
  }
  long long average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CPU GEMM: " << average_duration << " milliseconds" << std::endl;
  // print_matrix_cpu(c, M, N);

  delete[] a, delete[] b, delete[] c;
}