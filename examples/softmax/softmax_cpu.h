//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include "mlkl/cpu/operators.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

void initialize_matrix_cpu(float *matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

void set_random_matrix(float *matrix, int M, int N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < M * N; ++i) {
    matrix[i] = dist(gen);
  }
}

void print_matrix_cpu(const float *matrix, size_t M, size_t N) {
  const int width = 6;
  for (size_t i = 0; i < M; ++i) {
    std::cout << "[ ";
    for (size_t j = 0; j < N; ++j) {
      std::cout << std::setw(width) << matrix[i * N + j];
      if (j < N - 1) std::cout << " ";
    }
    std::cout << " ]" << std::endl;
  }
}

bool assert_correctness(float *output, int M) {
  float sum = 0.;

  for (int i = 0; i < M; ++i) {
    sum += output[i];
  }

  std::cout << "SUM: " << std::setw(6) << sum << std::endl;

  return sum == 1.f;
}

void softmax_cpu(int M, int N) {
  auto *input = new float[M];
  auto *output = new float[M];

  std::vector<int> shape{M};

  const int num_runs = 100;
  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    set_random_matrix(input, M, 1);
    initialize_matrix_cpu(output, M, 0, 1);

    auto start = std::chrono::high_resolution_clock::now();
    ml::operators::cpu::softmax(input, output, 0, shape);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    total_duration += duration.count();
  }

  long long average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CPU Softmax: " << average_duration << " nanoseconds" << std::endl;

  if (!assert_correctness(output, M)) {
    std::cerr << "Reference CPU kernel produced incorrect results." << std::endl;
  }

  //   print_matrix_cpu(output, 1, M);

  delete[] input, delete[] output;
}