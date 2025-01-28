#pragma once

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
template<typename T>
Tensor<T> create_tensor(int *shape) {
  int rank = 0;
  while (shape[rank] != 0) {
    ++rank;
  }

  Tensor<T> tensor;
  tensor.rank = rank;
  tensor.shape = new int[rank];
  tensor.stride = new int[rank];

  int totalElements = 1;
  for (int i = 0; i < rank; ++i) {
    tensor.shape[i] = shape[i];
    totalElements *= shape[i];
  }

  if (rank > 0) {
    tensor.stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
      tensor.stride[i] = tensor.stride[i + 1] * tensor.shape[i + 1];
    }
  }

  tensor.data = new T[totalElements];

  return tensor;
}

template<typename T>
void fill_tensor(Tensor<T> &tensor, int value) {
  for (int i = 0; i < utils::numel(tensor); ++i) {
    matrix[i] = value;
  }
}

template<typename T>
void destroy(Tensor<T> &tensor) {
  delete tensor.data;
  delete tensor.shape;
  delete tensor.stride;
}
}// namespace mlkl::operators::cpu