
#include "mlkl/core/tensor.h"
#include <chrono>
#include <iostream>

#include <mlkl/mlkl.h>

void attention_cpu(int batch_size, int num_heads, int seq_len_q, int seq_len_kv, int head_dim, int num_runs) {
  auto allocator = mlkl::TensorAllocator();

  std::vector<int> s1{batch_size, num_heads, seq_len_q, head_dim};
  std::vector<int> s2{batch_size, num_heads, seq_len_kv, head_dim};

  auto q = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CPU);
  auto k = allocator.empty(s2, mlkl::DType::F32, mlkl::Device::CPU);
  auto v = allocator.empty(s2, mlkl::DType::F32, mlkl::Device::CPU);
  auto output = allocator.empty(s1, mlkl::DType::F32, mlkl::Device::CPU);

  long long total_duration = 0;

  for (int i = 0; i < num_runs; ++i) {
    mlkl::randn(q);
    mlkl::randn(k);
    mlkl::randn(v);
    mlkl::fill(output, 0);

    auto start = std::chrono::high_resolution_clock::now();
    // ml::operators::cpu::transpose(b, b_T, M, N);
    mlkl::attention(q, k, v, output, mlkl::Device::CPU);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    total_duration += duration.count();
  }

  long long average_duration = total_duration / num_runs;
  float gflops = (2.0f * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim) / (average_duration / 1000.0f) / 1e9;
  std::cout << "Kernel: " << "CPU" << " | "
            << "Size: " << batch_size << "x" << num_heads << "x" << seq_len_q << "x" << seq_len_kv << "x" << head_dim << " | "
            << "Time: " << average_duration << " ms | "
            << "GFLOPS: " << gflops << std::endl;
}