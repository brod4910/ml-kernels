#include <cassert>
#include <iostream>

#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <mlkl/mlkl.h>
#include <mlkl/operators/cuda/gemm.h>
#include <mlkl/utils/device.h>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define CHECK_CUBLAS(call)                                               \
  {                                                                      \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ \
                << std::endl;                                            \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  }

auto cublas_kernel = [](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) {
  cublasLtHandle_t handle;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  int M = c->shape[0];
  int N = c->shape[1];
  int K = a->shape[1];

  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatrixLayout_t descA, descB, descC;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&descA, CUDA_R_16BF, M, K, M));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&descB, CUDA_R_16BF, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&descC, CUDA_R_32F, M, N, M));

  CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc,
                              &alpha, a->bf16_(), descA,
                              b->bf16_(), descB,
                              &beta, c->fp32_(), descC,
                              c->fp32_(), descC, nullptr, nullptr, 0, 0));
  cublasLtDestroy(handle);
};

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
  auto *c_ref = allocator.empty(s3, mlkl::Device::CPU);
  mlkl::fill(c_ref, 0);

  auto *a = allocator.empty(s1, mlkl::Device::CUDA);
  auto *b = allocator.empty(s2, mlkl::Device::CUDA);
  auto *c = allocator.empty(s3, mlkl::Device::CUDA);

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

void attention_cuda(int M, int N, int K, float alpha, float beta, int num_runs) {
  // Test CUBLAS
  test_kernel("CUBLAS", cublas_kernel, M, N, K, alpha, beta, num_runs);

  // Test custom kernels
  test_kernel("bf16 GEMM Kernel V1", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::bf16_gemm_v1(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("bf16 GEMM Kernel V2", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::bf16_gemm_v2(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
  test_kernel("bf16 GEMM Kernel V3", [&](mlkl::Tensor *a, mlkl::Tensor *b, mlkl::Tensor *c, float alpha, float beta) { mlkl::operators::cuda::bf16_gemm_v3(a, b, c, alpha, beta); }, M, N, K, alpha, beta, num_runs);
}