#include <mlkl/mlkl.h>

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iomanip>
#include <iostream>

#define CHECK_CUBLAS_STATUS(val) checkCuBLASStatus((val), #val, __FILE__, __LINE__)
void checkCuBLASStatus(cublasStatus_t status, const char *const func, const char *const file, const int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS Error at : " << file << ":" << line << '\n';
    std::cerr << cublasGetStatusString(status) << " " << func << '\n';
  }
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
  auto cpu_allocator = mlkl::TensorAllocator(mlkl::Device::CPU);
  auto gpu_allocator = mlkl::TensorAllocator(mlkl::Device::CUDA);

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto a_d = gpu_allocator.randn(s1);
  auto b_d = gpu_allocator.randn(s2);
  auto c_d = gpu_allocator.empty(s3);

  auto a_cpu = cpu_allocator.empty(s1);
  auto b_cpu = cpu_allocator.empty(s2);
  auto c_cpu = cpu_allocator.empty(s3);
  auto ref_matrix = cpu_allocator.empty(s3);

  mlkl::sgemm(a_cpu, b_cpu, c_cpu, alpha, beta, mlkl::Device::CPU);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_duration = 0;

  // std::cout << "Ref Matrix: " << std::endl;
  // print_matrix(ref_matrix, M, N);

  // warm-up
  for (int i = 0; i < 10; ++i) {
    kernel(a_d, b_d, c_d, alpha, beta);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < num_runs; ++i) {
    mlkl::fill(c_d, 0, mlkl::Device::CUDA);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start);

    kernel(a_d, b_d, c_d, alpha, beta);
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_duration += time_elapsed;
  }

  CHECK_CUDA_ERROR();

  // std::cout << "Output Matrix: " << std::endl;
  // print_matrix(c, M, N);

  // bool correct = assert_correctness(c, ref_matrix, M, N);
  // if (!correct) {
  //   std::cerr << "Kernel " << kernel_name << " produced incorrect results." << std::endl;
  // }

  float average_duration = total_duration / num_runs;
  float gflops = (2.0f * M * N * K) / (average_duration / 1000.0f) / 1e9;

  std::cout << "Kernel: " << kernel_name << " | "
            << "Size: " << M << "x" << K << "x" << N << " | "
            << "Time: " << average_duration << " ms | "
            << "GFLOPS: " << gflops << std::endl;

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void sgemm_cuda(int M, int N, int K, float alpha, float beta) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  int num_runs = 1000;

  auto cublas_kernel = [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) {
    CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, c, N));
  };
  // Test CUBLAS
  test_kernel("CUBLAS", cublas_kernel, M, N, K, alpha, beta, num_runs);

  // Test custom kernels
  test_kernel("SGEMM Kernel V2", [&](mlkl::Tensor &a, mlkl::Tensor &b, mlkl::Tensor &c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v2(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V3", [&](mlkl::Tensor &a, mlkl::Tensor &b, mlkl::Tensor &c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v3(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V4", [&](mlkl::Tensor &a, mlkl::Tensor &b, mlkl::Tensor &c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v4(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V5", [&](mlkl::Tensor &a, mlkl::Tensor &b, mlkl::Tensor &c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v5(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V6", [&](mlkl::Tensor &a, mlkl::Tensor &b, mlkl::Tensor &c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v6(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);

  cublasDestroy(handle);
}