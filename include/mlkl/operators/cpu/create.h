#pragma once

#include "mlkl/tensor/tensor.h"

namespace mlkl::operators::cpu {

template<typename T>
Tensor<T> create_tensor(int *shape) {
  // 1) Determine the rank by scanning until sentinel (0 here)
  int rank = 0;
  while (shape[rank] != 0) {
    ++rank;
  }

  // 2) Allocate the Tensor and its shape/stride arrays
  Tensor<T> tensor{};
  tensor.rank = rank;
  tensor.shape = new int[rank];
  tensor.stride = new int[rank];

  // 3) Copy the shape values and compute total number of elements
  int totalElements = 1;
  for (int i = 0; i < rank; ++i) {
    tensor.shape[i] = shape[i];
    totalElements *= shape[i];
  }

  // 4) Compute strides (row-major layout as an example)
  //    stride[rank-1] = 1
  //    stride[i] = stride[i+1] * shape[i+1]
  if (rank > 0) {
    tensor.stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
      tensor.stride[i] = tensor.stride[i + 1] * tensor.shape[i + 1];
    }
  }

  // 5) Allocate data for the Tensor
  tensor.data = new T[totalElements];

  return tensor;
}
}// namespace mlkl::operators::cpu