#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <mlkl/mlkl.h>
#include <mlkl/operators/cuda/softmax.h>
#include <mlkl/utils/device.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

template<typename Kernel>
void test_kernel(const char *kernel_name,
                 Kernel kernel,
                 int M, int N, int num_runs) {
  std::vector<int>
    shape{M, N};
  auto allocator = mlkl::TensorAllocator();

  std::vector<int> s1{M, N};

  auto *a_d = allocator.randn(s1, mlkl::Device::CUDA);
  auto *a_cpu = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CPU);
  auto *b = allocator.randn(s1, mlkl::Device::CUDA);

  mlkl::copy(a_d, a_cpu);

  auto *ref_cpu = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CPU);

  mlkl::softmax(a_cpu, ref_cpu, 0, mlkl::Device::CPU);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_duration = 0;

  // warm-up
  for (int i = 0; i < 10; ++i) {
    kernel(a_d, b, 0);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < num_runs; ++i) {
    mlkl::fill(b, 0);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start);

    kernel(a_d, b, 0);
    CHECK_CUDA_ERROR();

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);
    total_duration += time_elapsed;
  }

  b->to(mlkl::Device::CPU);
  CHECK_CUDA_ERROR();

  if (!mlkl::equals(b, ref_cpu)) {
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
}

void softmax_cuda(int M, int N, int num_runs) {
  // Test custom kernels
  test_kernel("Softmax Kernel V1", [&](mlkl::Tensor *a, mlkl::Tensor *b, int dim) { mlkl::operators::cuda::softmax(a, b, dim); }, M, N, num_runs);
}