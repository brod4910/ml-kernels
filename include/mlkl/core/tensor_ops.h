#pragma once
#include <mlkl/operators/cpu/tensor.h>

#ifdef __CUDACC__
#include <mlkl/operators/cuda/tensor.h>
#else
#include <stdexcept>
#endif

#include <mlkl/core/tensor.h>

#include <vector>

namespace mlkl {
mlkl::Tensor empty(std::vector<int> &shape, Device device) {
  if (device == mlkl::Device::CPU) {
    return operators::cpu::empty(shape);
  } else {
#ifdef __CUDACC__
    return operators::cuda::empty(shape);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

void fill(mlkl::Tensor &tensor, int value) {
  if (tensor.device == mlkl::Device::CPU) {
    return operators::cpu::fill(tensor, value);
  } else {
#ifdef __CUDACC__
    return operators::cuda::fill(tensor, value);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

void destroy(mlkl::Tensor &tensor) {
  if (tensor.device == mlkl::Device::CPU) {
    return operators::cpu::destroy(tensor);
  } else {
#ifdef __CUDACC__
    return operators::cuda::destroy(tensor);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

Tensor randn(std::vector<int> &shape, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::randn(shape);
  } else {
#ifdef __CUDACC__
    return operators::cuda::randn(shape);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

void randn(mlkl::Tensor &tensor) {
  if (tensor.device == Device::CPU) {
    return operators::cpu::randn(tensor);
  } else {
#ifdef __CUDACC__
    return operators::cuda::randn(tensor);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

mlkl::Tensor to(Tensor &tensor, Device device) {
  if (tensor.device == device) {
    return tensor;
  }

  if (device == Device::CPU) {
  }
}
}// namespace mlkl