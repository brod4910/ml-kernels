//
// Created by Brian Rodriguez on 10/24/23.
//
#include "cuda_runtime_api.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <mlkl/cuda/operators.h>

void check_cuda_error(const char *file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(error) << " \""
              << cudaGetErrorString(error) << "\"" << std::endl;
    exit(-1);
  }
}

void set_cpu_matrix(float *matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

void print_matrix_cuda(float *d_matrix, size_t M, size_t N) {
  float *matrix = new float[M * N];
  cudaMemcpy(matrix, d_matrix, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  delete[] matrix;
}

float *initialize_cuda_matrix(float *matrix, size_t size) {
  float *d_matrix;

  cudaMalloc(&d_matrix, size * sizeof(float));

  return d_matrix;
}

void set_cuda_matrix(float *matrix, float *d_matrix, size_t size) {
  cudaMemcpy(d_matrix, matrix, size * sizeof(float), cudaMemcpyHostToDevice);
}

void sgemm_cuda(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto *a = new float[M * K];
  auto *b = new float[N * K];
  auto *b_T = new float[K * N];
  auto *c = new float[M * N];

  auto *a_d = initialize_cuda_matrix(a, M * K);
  check_cuda_error(__FILE__, __LINE__);
  auto *b_d = initialize_cuda_matrix(b, N * K);
  check_cuda_error(__FILE__, __LINE__);
  auto *b_T_d = initialize_cuda_matrix(b_T, N * K);
  check_cuda_error(__FILE__, __LINE__);
  auto *c_d = initialize_cuda_matrix(c, M * N);
  check_cuda_error(__FILE__, __LINE__);

  const int num_runs = 100;
  float total_duration = 0;
  int blk_size = 32;

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
    check_cuda_error(__FILE__, __LINE__);
    set_cuda_matrix(b, b_d, N * K);
    check_cuda_error(__FILE__, __LINE__);
    set_cuda_matrix(b_T, b_T_d, N * K);
    check_cuda_error(__FILE__, __LINE__);
    set_cuda_matrix(c, c_d, M * N);
    check_cuda_error(__FILE__, __LINE__);

    cudaEventRecord(start);

    // ml::operators::cuda::transpose(b, b_T, M, N);
    ml::operators::cuda::launch_sgemm_v1(a_d, alpha, b_T_d, beta, c_d, M, N, K, blk_size);
    check_cuda_error(__FILE__, __LINE__);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    total_duration += time_elapsed;
  }

  float average_duration = total_duration / num_runs;
  std::cout << "Average time taken by function CUDA GEMM: " << average_duration << " milliseconds" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(b_T_d);
  cudaFree(c_d);

  delete[] a;
  delete[] b;
  delete[] b_T;
  delete[] c;
}