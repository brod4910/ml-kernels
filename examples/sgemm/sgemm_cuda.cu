#include "mlkl/cpu/operators/gemm.h"
#include "mlkl/cuda/operators/gemm.h"

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

#define CHECK_CUDA_ERROR() check_cuda_error(__FILE__, __LINE__)
void check_cuda_error(const char *file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(error) << " \""
              << cudaGetErrorString(error) << "\"" << std::endl;
    exit(-1);
  }
}

#define CHECK_CUBLAS_STATUS(val) checkCuBLASStatus((val), #val, __FILE__, __LINE__)
void checkCuBLASStatus(cublasStatus_t status, const char *const func, const char *const file, const int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS Error at : " << file << ":" << line << '\n';
    std::cerr << cublasGetStatusString(status) << " " << func << '\n';
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

void initialize_matrix_from_0_to_N(float *matrix, size_t M, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      matrix[i * N + j] = static_cast<float>(i * N + j);
    }
  }
}

void fill_matrix(float *matrix, int M, int N, float value) {
  for (int i = 0; i < M * N; ++i) {
    matrix[i] = value;
  }
}

float *initialize_cuda_matrix(float *matrix, size_t size) {
  float *d_matrix;
  cudaMalloc(&d_matrix, size * sizeof(float));
  if (matrix != nullptr) {
    cudaMemcpy(d_matrix, matrix, size * sizeof(float), cudaMemcpyHostToDevice);
  }
  return d_matrix;
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

void print_matrix(const float *matrix, size_t M, size_t N) {
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

template<typename Kernel>
void test_kernel(const char *kernel_name,
                 Kernel kernel,
                 int M, int N, int K, float alpha, float beta, int num_runs = 10) {
  auto *a = new float[M * K];
  auto *b = new float[K * N];
  auto *c = new float[M * N];
  auto *ref_matrix = new float[M * N];

  set_random_matrix(a, M, K);
  set_random_matrix(b, K, N);
  fill_matrix(c, M, N, 0);
  fill_matrix(ref_matrix, M, N, 0);

  ml::operators::cpu::sgemm(a, alpha, b, beta, ref_matrix, M, N, K);

  auto *a_d = initialize_cuda_matrix(a, M * K);
  auto *b_d = initialize_cuda_matrix(b, K * N);
  auto *c_d = initialize_cuda_matrix(nullptr, M * N);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_duration = 0;

  // warm-up
  for (int i = 0; i < 10; ++i) {
    kernel(a_d, alpha, b_d, beta, c_d, M, N, K);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < num_runs; ++i) {
    cudaMemcpy(c_d, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start);

    kernel(a_d, alpha, b_d, beta, c_d, M, N, K);
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_duration += time_elapsed;
  }

  cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  bool correct = assert_correctness(c, ref_matrix, M, N);
  if (!correct) {
    std::cerr << "Kernel " << kernel_name << " produced incorrect results." << std::endl;
  } else {
    float average_duration = total_duration / num_runs;
    float gflops = (2.0f * M * N * K) / (average_duration / 1000.0f) / 1e9;

    std::cout << "Kernel: " << kernel_name << " | "
              << "Size: " << M << "x" << K << "x" << N << " | "
              << "Time: " << average_duration << " ms | "
              << "GFLOPS: " << gflops << std::endl;
  }

  // std::cout << "matrix: \n";
  // print_matrix(c, M, N);
  // std::cout << "ref: \n";
  // print_matrix(ref_matrix, M, N);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] ref_matrix;
}

void sgemm_cuda(int M, int N, int K, float alpha, float beta) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  int num_runs = 100;

  auto cublas_kernel = [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) {
    CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, c, N));
  };
  // Test CUBLAS
  test_kernel("CUBLAS", cublas_kernel, M, N, K, alpha, beta, num_runs);

  // Test custom kernels
  test_kernel("Custom Kernel V2", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { ml::operators::cuda::launch_sgemm_v2(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("Custom Kernel V3", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { ml::operators::cuda::launch_sgemm_v3(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("Custom Kernel V4", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { ml::operators::cuda::launch_sgemm_v4(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);

  cublasDestroy(handle);
}