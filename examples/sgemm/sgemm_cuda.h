//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include "cuda_runtime_api.h"

#include <mlkl/cuda/operators.h>
#include <chrono>
#include <iostream>
#include <memory>

void set_cpu_matrix(float *matrix, size_t size, float value, int skip = 1) {
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

float* initialize_cuda_matrix(float* matrix, size_t size) {
    float* d_matrix;

    cudaMalloc(&d_matrix, size * sizeof(float));

    return d_matrix;
}

void set_cuda_matrix(float* matrix, float* d_matrix, size_t size) {
    cudaMemcpy(d_matrix, matrix, size * sizeof(float), cudaMemcpyHostToDevice);
}

void sgemm_cuda(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto *a = new float[M * K];
  auto *b = new float[N * K];
  auto *b_T = new float[K * N];
  auto *c = new float[M * N];

  auto* a_d = initialize_cuda_matrix(a, M * K);
  auto* b_d = initialize_cuda_matrix(b, N * K);
  auto* b_T_d = initialize_cuda_matrix(b_T, N * K);
  auto* c_d = initialize_cuda_matrix(c, M * N);

  const int num_runs = 100;
  long long total_duration = 0;
  int blk_size = 64;

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  set_cpu_matrix(a, M * K, 1, 1);
  set_cpu_matrix(b, N * K, 2, 1);
  set_cpu_matrix(b_T, K * N, 2, 1);
  set_cpu_matrix(c, M * N, 0, 1);
  
  for (int i = 0; i < num_runs; ++i) {
    set_cuda_matrix(a, a_d, M * K);
    set_cuda_matrix(b, b_d, N * K);
    set_cuda_matrix(b_T, b_T_d, N * K);
    set_cuda_matrix(c, c_d, M * N);

    cudaEventRecord(start, 0);
    
    // ml::operators::cuda::transpose(b, b_T, M, N);
    ml::operators::cuda::sgemm_v1(a, alpha, b_T, beta, c, M, N, K, blk_size);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    total_duration += time_elapsed;
  }
  float average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CUDA GEMM: " << average_duration << " milliseconds" << std::endl;

  delete[] a, delete[] b, delete[] c;
}