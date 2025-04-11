
#include "mlkl/core/tensor.h"
#include <chrono>
#include <iostream>

#include <mlkl/mlkl.h>

void sgemm_cpu(int M, int N, int K, float alpha, float beta, int num_runs) {
  auto allocator = mlkl::TensorAllocator();

  std::vector<int> s1{M, K};
  std::vector<int> s2{K, N};
  std::vector<int> s3{M, N};

  auto a = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CPU);
  auto b = allocator.empty(s2, mlkl::DType::F32, mlkl::Device::CPU);
  auto c = allocator.empty(s3, mlkl::DType::F32, mlkl::Device::CPU);

  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    mlkl::randn(a);
    mlkl::randn(b);
    mlkl::fill(c, 0);

    auto start = std::chrono::high_resolution_clock::now();
    // ml::operators::cpu::transpose(b, b_T, M, N);
    mlkl::sgemm(a, b, c, alpha, beta, mlkl::Device::CPU);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    total_duration += duration.count();
  }

  long long average_duration = total_duration / num_runs;
  float gflops = (2.0f * M * N * K) / (average_duration / 1000.0f) / 1e9;
  std::cout << "Kernel: " << "CPU" << " | "
            << "Size: " << M << "x" << K << "x" << N << " | "
            << "Time: " << average_duration << " ms | "
            << "GFLOPS: " << gflops << std::endl;
}