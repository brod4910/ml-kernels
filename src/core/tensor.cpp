#include <cstddef>
#include <functional>
#include <numeric>

#include <mlkl/core/tensor_ops.h>

namespace mlkl {
size_t Tensor::num_bytes() {
  return numel() * sizeof(*data);
}

size_t Tensor::numel() {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void Tensor::to(Device device) {
  if (this->device == device) {
    return;
  }

  auto *temp = empty(shape, device);
  copy(this, temp);
  destroy(this);

  this->device = temp->device;
  this->data = temp->data;
}
}// namespace mlkl