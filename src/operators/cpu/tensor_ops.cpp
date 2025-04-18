#include "mlkl/core/tensor.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <mlkl/operators/cpu/tensor_ops.h>

namespace mlkl::operators::cpu {
Tensor *empty(std::vector<int> &shape) {
  Tensor *tensor = new Tensor;
  tensor->rank = shape.size();
  tensor->shape.reserve(tensor->rank);
  tensor->stride.reserve(tensor->rank);

  for (int i = 0; i < tensor->rank; ++i) {
    tensor->shape.push_back(shape[i]);
  }

  if (tensor->rank > 0) {
    tensor->stride[tensor->rank - 1] = 1;
    for (int i = tensor->rank - 2; i >= 0; --i) {
      tensor->stride.push_back(tensor->stride[i + 1] * tensor->shape[i + 1]);
    }
  }

  tensor->data = malloc(tensor->num_bytes());

  return tensor;
}

void fill(Tensor *tensor, int value) {
  for (size_t i = 0; i < tensor->numel(); ++i) {
    // TODO: Any better way then to pin to fp32 besides templates?
    tensor->fp32_()[i] = value;
  }
}

void copy(Tensor *src, Tensor *dst) {
  memcpy(dst->data, src->data, dst->num_bytes());
}

void destroy(Tensor *tensor) {
  free(tensor->data);
  delete tensor;
}

namespace {
void randn(float *data, size_t numel) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (size_t i = 0; i < numel; ++i) {
    data[i] = dist(gen);
  }
}

}// namespace

Tensor *randn(std::vector<int> &shape) {
  auto tensor = empty(shape);

  randn(tensor->fp32_(), tensor->numel());

  return tensor;
}

void randn(Tensor *tensor) {
  randn(tensor->fp32_(), tensor->numel());
}
}// namespace mlkl::operators::cpu