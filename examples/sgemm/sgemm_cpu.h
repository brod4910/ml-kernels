//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <chrono>
#include <iostream>

#include <mlkl/mlkl.h>

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

void sgemm_cpu(int M, int N, int K, float alpha, float beta) {
  auto cpu_allocator = mlkl::TensorAllocator(mlkl::Device::CPU);

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto a = cpu_allocator.empty(s1);
  auto b = cpu_allocator.empty(s2);
  auto c = cpu_allocator.empty(s3);

  const int num_runs = 100;
  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    mlkl::randn(a, mlkl::Device::CPU);
    mlkl::randn(b, mlkl::Device::CPU);
    mlkl::fill(c, 0, mlkl::Device::CPU);

    auto start = std::chrono::high_resolution_clock::now();
    // ml::operators::cpu::transpose(b, b_T, M, N);
    mlkl::sgemm(a, b, c, alpha, beta, mlkl::Device::CPU);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    total_duration += duration.count();
  }
  long long average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CPU GEMM: " << average_duration << " milliseconds" << std::endl;
}