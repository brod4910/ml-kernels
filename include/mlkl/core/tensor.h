#pragma once

#include <functional>
#include <numeric>

#ifdef __CUDACC__
#include <cuda_fp16.h>

using fp16 = __half;
#else
using fp16 = uint16_t;
#endif

using fp32 = float;

namespace mlkl {
enum class Device {
  CPU,
  CUDA
};

enum class DType {
  F32,
  F16,
  F8
};

struct Tensor {
  float *data;

  int rank;
  std::vector<int> shape;
  std::vector<int> stride;

  Device device = Device::CPU;
  DType dtype = DType::F32;

  size_t num_bytes() {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(*data);
  }

  size_t numel() {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }

  void calculate_stride() {
    int stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
      this->stride[i] = stride;
      stride *= shape[i];
    }
  }
};

}// namespace mlkl