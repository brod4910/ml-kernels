#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <functional>
#include <numeric>

using fp32 = float;
using bf16 = __nv_bfloat16;
using fp16 = __half;

namespace mlkl {
enum class Device {
  CPU,
  CUDA
};

enum class DType {
  F32,
  F16,
  BF16,
  F8
};

struct Tensor {
  float *data;

  int rank;
  std::vector<int> shape;
  std::vector<int> stride;

  Device device = Device::CPU;
  DType dtype = DType::F32;

  int num_bytes() {
    return numel() * sizeof(*data);
  }

  int numel() {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }
};

}// namespace mlkl