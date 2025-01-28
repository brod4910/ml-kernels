#include <mlkl/mlkl.h>

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

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
  auto cpu_allocator = mlkl::TensorAllocator<float>(mlkl::CPU);
  auto gpu_allocator = mlkl::TensorAllocator<float>(mlkl::GPU);

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto a = gpu_allocator.create_tensor(s1.data);
  auto b = gpu_allocator.create_tensor(s2.data);
  auto c = gpu_allocator.create_tensor(s3.data);

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

  // std::cout << "Ref Matrix: " << std::endl;
  // print_matrix(ref_matrix, M, N);

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

  // std::cout << "Output Matrix: " << std::endl;
  // print_matrix(c, M, N);

  bool correct = assert_correctness(c, ref_matrix, M, N);
  if (!correct) {
    std::cerr << "Kernel " << kernel_name << " produced incorrect results." << std::endl;
  }

  float average_duration = total_duration / num_runs;
  float gflops = (2.0f * M * N * K) / (average_duration / 1000.0f) / 1e9;

  std::cout << "Kernel: " << kernel_name << " | "
            << "Size: " << M << "x" << K << "x" << N << " | "
            << "Time: " << average_duration << " ms | "
            << "GFLOPS: " << gflops << std::endl;

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
  int num_runs = 1000;

  auto cublas_kernel = [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) {
    CHECK_CUBLAS_STATUS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, c, N));
  };
  // Test CUBLAS
  test_kernel("CUBLAS", cublas_kernel, M, N, K, alpha, beta, num_runs);

  // Test custom kernels
  test_kernel("SGEMM Kernel V2", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { mlkl::operators::cuda::launch_sgemm_v2(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V3", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { mlkl::operators::cuda::launch_sgemm_v3(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V4", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { mlkl::operators::cuda::launch_sgemm_v4(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V5", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { mlkl::operators::cuda::launch_sgemm_v5(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);
  test_kernel("SGEMM Kernel V6", [&](float *a, float alpha, float *b, float beta, float *c, int M, int N, int K) { mlkl::operators::cuda::launch_sgemm_v6(a, alpha, b, beta, c, M, N, K); }, M, N, K, alpha, beta, num_runs);

  cublasDestroy(handle);
}