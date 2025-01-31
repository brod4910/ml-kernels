#include "mlkl/core/tensor.h"
#include <mlkl/mlkl.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

template<typename Kernel>
void test_kernel(const char *kernel_name,
                 Kernel kernel,
                 int M, int N, int num_runs = 10) {
  std::vector<int>
    shape{M, N};
  auto cpu_allocator = mlkl::TensorAllocator(mlkl::Device::CPU);
  auto gpu_allocator = mlkl::TensorAllocator(mlkl::Device::CUDA);

  std::vector<int> s1{M, N};

  auto a_d = gpu_allocator.randn(s1);
  auto b_d = gpu_allocator.randn(s1);

  auto a_cpu = cpu_allocator.empty(s1);
  auto b_cpu = cpu_allocator.empty(s1);

  mlkl::softmax(a_cpu, b_cpu, 0, mlkl::Device::CPU);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_duration = 0;

  // warm-up
  for (int i = 0; i < 10; ++i) {
    kernel(a_d, b_d, 0, shape);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < num_runs; ++i) {
    cudaMemcpy(b_d, b, M * N * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start);

    kernel(a_d, b_d, 0, shape);
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_duration += time_elapsed;
  }

  cudaMemcpy(b, b_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  bool correct = mlkl::cpu::utils::assert_correctness(b, ref_matrix, M, N);
  if (!correct) {
    std::cerr << "Kernel " << kernel_name << " produced incorrect results." << std::endl;
  }

  float average_duration = total_duration / num_runs;
  float gflops = (2.0f * M * N) / (average_duration / 1000.0f) / 1e9;

  std::cout << "Kernel: " << kernel_name << " | "
            << "Size: " << M << "x" << N << " | "
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
  delete[] a;
  delete[] b;
  delete[] ref_matrix;
}

void softmax_cuda(int M, int N) {
  int num_runs = 1000;

  // Test custom kernels
  test_kernel("Softmax Kernel V1", [&](float *a, float *b, int dim, std::vector<int> &shape) { mlkl::operators::cuda::launch_softmax_2d_v1(a, b, dim, shape); }, M, N, num_runs);
}