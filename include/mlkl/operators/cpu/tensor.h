#pragma once

#include <cstdlib>
#include <random>
#include <vector>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
Tensor create_tensor(std::vector<int> &shape) {
  Tensor tensor;
  tensor.rank = shape.size();
  tensor.shape = new int[tensor.rank];
  tensor.stride = new int[tensor.rank];

  for (int i = 0; i < tensor.rank; ++i) {
    tensor.shape[i] = shape[i];
  }

  if (tensor.rank > 0) {
    tensor.stride[tensor.rank - 1] = 1;
    for (int i = tensor.rank - 2; i >= 0; --i) {
      tensor.stride[i] = tensor.stride[i + 1] * tensor.shape[i + 1];
    }
  }

  tensor.data = reinterpret_cast<decltype(tensor.data)>(malloc(num_bytes(tensor)));

  return tensor;
}

void fill_tensor(Tensor &tensor, int value) {
  for (int i = 0; i < numel(tensor); ++i) {
    tensor.data[i] = value;
  }
}

void destroy(Tensor &tensor) {
  free(tensor.data);
  delete[] tensor.shape;
  delete[] tensor.stride;
}

Tensor randn(std::vector<int> &shape) {
  auto tensor = create_tensor(shape);

  randn(tensor.data, mlkl::numel(tensor));

  return tensor;
}

void randn(Tensor &tensor) {
  randn(tensor.data, mlkl::numel(tensor));
}

namespace {
void randn(float *data, size_t numel) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < numel; ++i) {
    data[i] = dist(gen);
  }
}
}// namespace
}// namespace mlkl::operators::cpu