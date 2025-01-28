#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mlkl/core/tensor_ops.h>

namespace mlkl {
enum Device {
  CPU,
  CUDA
};

struct Tensor {
  void *data;

  int rank;
  int *shape;
  int *stride;
};

template<typename T>
struct TensorAllocator {
  std::vector<Tensor<T>> tensors_;
  Device device_;

  TensorAllocator(Device device) : device_(device) {}

  ~TensorAllocator() {
    for (auto tensor : tensors_) {
      destroy(tensor, device_);
    }
  }

  Tensor<T> alloc(int *shape) {
    auto tensor = create_tensor<T>(shape, device_);
    tensors_.push_back(tensor);
    return tensor;
  }
};

template<typename T>
size_t num_bytes(Tensor<T> &tensor) {
  return std::accumulate(tensor.shape, tensor.shape + tensor.rank, 1, std::multiplies<int>()) * sizeof(T);
}

template<typename T>
size_t numel(Tensor<T> &tensor) {
  return std::accumulate(tensor.shape, tensor.shape + tensor.rank, 1, std::multiplies<int>());
}

template<typename T>
void calculate_stride(Tensor<T> &tensor) {
  int stride = 1;
  for (int i = tensor.rank - 1; i > 0; --i) {
    tensor.stride[i] = stride;
    stride *= tensor.shape[i];
  }
}
}// namespace mlkl