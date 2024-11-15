//
// Created by Brian Rodriguez on 10/24/23.
//
#include "mlkl/cpu/operators/gemm.h"
#include "mlkl/cuda/operators.h"
#include "mlkl/cuda/operators/gemm.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <vector>

void check_cuda_error(const char *file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(error) << " \""
              << cudaGetErrorString(error) << "\"" << std::endl;
    exit(-1);
  }
}

void set_random_matrix(float *matrix, int M, int N) {
  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);// Random float in the range [0, 1]

  for (int i = 0; i < M * N; ++i) {
    matrix[i] = dist(gen);
  }
}

void fill_matrix(float *matrix, int M, int N, int value) {
  for (int i = 0; i < M * N; ++i) {
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

bool assert_correctness(float *matrix, float *ref_matrix, size_t M, size_t N, float epsilon = 1e-6) {
  double diff = 0.;

  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      int linear = m * N + n;
      diff = fabs((double) matrix[linear] - (double) ref_matrix[linear]);
      if (diff > 1e-2) {
        printf("Error: %5.2f,%5.2f, (%lu, %lu) \n", matrix[linear], ref_matrix[linear], m, n);
        return false;
      }
    }
  }
  return true;
}

void sgemm_cuda(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto *a = new float[M * K];
  auto *b = new float[N * K];
  auto *b_T = new float[K * N];
  auto *c = new float[M * N];
  auto *ref_matrix = new float[M * N];

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
  constexpr int BLOCK_SIZE_X = 16;
  constexpr int BLOCK_SIZE_Y = 16;

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  set_random_matrix(a, M, K);
  set_random_matrix(b, N, K);
  set_random_matrix(b_T, K, N);
  fill_matrix(c, M, N, 0);

  fill_matrix(ref_matrix, M, N, 0);
  ml::operators::cpu::sgemm(a, alpha, b_T, beta, ref_matrix, M, N, K);

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

    // ml::operators::cuda::launch_sgemm_v1(a_d, alpha, b_T_d, beta, c_d, M, N, K);
    // ml::operators::cuda::launch_sgemm_v2(a_d, alpha, b_T_d, beta, c_d, M, N, K, BLOCK_SIZE_X);
    ml::operators::cuda::launch_sgemm_v3(a_d, alpha, b_T_d, beta, c_d, M, N, K);
    check_cuda_error(__FILE__, __LINE__);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    total_duration += time_elapsed;
  }

  cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  if (!assert_correctness(c, ref_matrix, M, N)) {
    std::cout << "Result for CUDA GEMM not correct..." << std::endl;
  } else {
    float average_duration = total_duration / num_runs;
    std::cout << "Average time taken by function CUDA GEMM: " << average_duration << " milliseconds" << std::endl;
  }

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