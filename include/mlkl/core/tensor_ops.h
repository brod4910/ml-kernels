#pragma once

#include <mlkl/core/tensor.h>

#include <mlkl/operators/cpu/tensor.h>

#ifdef __CUDA__
#include <mlkl/operators/cuda/tensor.h>
#endif

namespace mlkl {
template<typename T>
Tensor<T> create_tensor(int *shape, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::create_tensor<T>(shape);
  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::create_tensor<T>(shape);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

template<typename T>
void fill_tensor(Tensor<T> &tensor, int value, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::fill_tensor<T>(tensor, value);
  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::fill_tensor<T>(tensor, value);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

template<typename T>
void destroy(Tensor<T> &tensor, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::destroy<T>(tensor);
  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::destroy<T>(tensor);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}
}// namespace mlkl