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
  mlkl::to(*this, device);
}
}// namespace mlkl