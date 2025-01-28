#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef __CUDA__
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
  int *shape;
  int *stride;

  Device device = Device::CPU;
  DType dtype = DType::F32;

  void to_fp16() {
    throw std::runtime_error("fp16 conversion not supported.");
  }
};

size_t num_bytes(Tensor &tensor) {
  return std::accumulate(tensor.shape, tensor.shape + tensor.rank, 1, std::multiplies<int>()) * sizeof(*tensor.data);
}

size_t numel(Tensor &tensor) {
  return std::accumulate(tensor.shape, tensor.shape + tensor.rank, 1, std::multiplies<int>());
}

void calculate_stride(Tensor &tensor) {
  int stride = 1;
  for (int i = tensor.rank - 1; i > 0; --i) {
    tensor.stride[i] = stride;
    stride *= tensor.shape[i];
  }
}
}// namespace mlkl