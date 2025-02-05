//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <chrono>
#include <iostream>

#include <mlkl/mlkl.h>

void sgemm_cpu(int M, int N, int K, float alpha, float beta) {
  auto allocator = mlkl::TensorAllocator();

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto a = allocator.empty(s1, mlkl::Device::CPU);
  auto b = allocator.empty(s2, mlkl::Device::CPU);
  auto c = allocator.empty(s3, mlkl::Device::CPU);

  const int num_runs = 100;
  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    mlkl::randn(a);
    mlkl::randn(b);
    mlkl::fill(c, 0);

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