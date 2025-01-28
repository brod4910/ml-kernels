#pragma once
#include <mlkl/core/tensor.h>
#include <mlkl/operators/operators.h>

namespace mlkl {
template<typename T>
void softmax(Tensor<T> &tensor, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::softmax<T>(tensor);
  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::softmax<T>(tensor);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}
}// namespace mlkl