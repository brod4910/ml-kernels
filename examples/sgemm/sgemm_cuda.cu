#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <mlkl/mlkl.h>
#include <mlkl/operators/cuda/gemm.h>
#include <mlkl/utils/device.h>

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define CHECK_CUBLAS_STATUS(val) checkCuBLASStatus((val), #val, __FILE__, __LINE__)
void checkCuBLASStatus(cublasStatus_t status, const char *const func, const char *const file, const int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS Error at : " << file << ":" << line << '\n';
    std::cerr << cublasGetStatusString(status) << " " << func << '\n';
  }
}

template<typename Kernel>
void test_kernel(const char *kernel_name,
                 Kernel kernel,
                 int M, int N, int K, float alpha, float beta, int num_runs = 10) {
  auto allocator = mlkl::TensorAllocator();

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto *a_ref = allocator.randn(s1, mlkl::Device::CPU);
  auto *b_ref = allocator.randn(s2, mlkl::Device::CPU);
  auto *c_ref = allocator.empty(s3, mlkl::DType::F32, mlkl::Device::CPU);
  mlkl::fill(c_ref, 0);

  auto *a = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CUDA);
  auto *b = allocator.empty(s2, mlkl::DType::F32, mlkl::Device::CUDA);
  auto *c = allocator.empty(s3, mlkl::DType::F32, mlkl::Device::CUDA);

  mlkl::copy(a_ref, a);
  mlkl::copy(b_ref, b);
  mlkl::copy(c_ref, c);

  mlkl::sgemm(a_ref, b_ref, c_ref, alpha, beta, mlkl::Device::CPU);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_duration = 0;

  // warm-up
  for (int i = 0; i < 10; ++i) {
    kernel(a, b, c, alpha, beta);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < num_runs; ++i) {
    mlkl::fill(c, 0);

    cudaEventRecord(start);

    kernel(a, b, c, alpha, beta);
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_duration += time_elapsed;
  }

  c->to(mlkl::Device::CPU);

  if (!mlkl::equals(c, c_ref, 1e-3)) {
    std::cerr << "Kernel " << kernel_name << " produced incorrect results." << std::endl;
  }

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

void sgemm_cuda(int M, int N, int K, float alpha, float beta, int num_runs) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  auto cublas_kernel = [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) {
    int M = c->shape[0];
    int N = c->shape[1];
    int K = a->shape[1];
    CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b->fp32_(), N, a->fp32_(), K, &beta, c->fp32_(), N));
  };
  // Test CUBLAS
  test_kernel("CUBLAS", cublas_kernel, M, N, K, alpha, beta, num_runs);

  // Test custom kernels
  test_kernel("SGEMM Kernel V2", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v2(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V3", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v3(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V4", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v4(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V5", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v5(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V6", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::sgemm_v6(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);

  cublasDestroy(handle);
}