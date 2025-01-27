#pragma once
#include <stdexcept>

#include "mlkl/operators/cpu/create.h"
#include "mlkl/tensor/tensor.h"

#ifdef __CUDA__
#include "mlkl/operators/cuda/create.h"
#endif

namespace mlkl {
template<typename T>
Tensor<T> create_tensor(int *shape, Device device) {
  if (device == Device::CPU) {

  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::create_tensor<T>(shape);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

template<typename T>
void fill_tensor(Tensor<T> &tensor, Device device) {
  if (device == Device::CPU) {

  } else if (device == Device::CUDA) {
#ifdef __CUDA__
    return operators::cuda::fill_tensor<T>(tensor);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}
}// namespace mlkl