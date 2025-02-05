//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <mlkl/mlkl.h>

#include <chrono>
#include <iostream>

void softmax_cpu(int M, int N) {
  std::vector<int> shape{M};
  auto allocator = mlkl::TensorAllocator();

  auto input = allocator.randn(shape, mlkl::Device::CPU);
  auto output = allocator.empty(shape, mlkl::Device::CPU);

  const int num_runs = 100;
  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    mlkl::randn(input);
    mlkl::fill(output, 0);

    auto start = std::chrono::high_resolution_clock::now();
    mlkl::softmax(input, output, 0, mlkl::Device::CPU);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    total_duration += duration.count();
  }

  long long average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CPU Softmax: " << average_duration << " nanoseconds" << std::endl;

  // if (!assert_correctness(output, M)) {
  //   std::cerr << "Reference CPU [kernel produced incorrect results." << std::endl;
  // }

  //   print_matrix_cpu(output, 1, M);
}