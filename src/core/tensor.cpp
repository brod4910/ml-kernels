#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>

#include <mlkl/core/tensor_ops.h>

namespace mlkl {
size_t Tensor::num_bytes() {
  return numel() * dtype_size();
}

size_t Tensor::dtype_size() {
  switch (dtype) {
    case DType::F8:
      return 1;
    case DType::F16:
    case DType::BF16:
      return 2;
    default:
      return 4;
  }
}

bf16* Tensor::bf16_() {
  assert(dtype == DType::BF16);
  return get_data<bf16>();
}

fp16* Tensor::fp16_() {
  assert(dtype == DType::F16);
  return get_data<fp16>();
}

fp32* Tensor::fp32_() {
  assert(dtype == DType::F32);
  return get_data<fp32>();
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