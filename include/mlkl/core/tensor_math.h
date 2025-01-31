#pragma once
#include <mlkl/core/tensor.h>
#include <mlkl/operators/operators.h>

namespace mlkl {
void softmax(Tensor &input, Tensor &output, int dim, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::softmax(input, output, dim);
  } else if (device == Device::CUDA) {
#ifdef __CUDACC__
    return operators::cuda::softmax(input, output, dim);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}

void sgemm(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::sgemm(a, b, c, alpha, beta);
  } else if (device == Device::CUDA) {
#ifdef __CUDACC__
    return operators::cuda::sgemm_v6(a, b, c, alpha, beta);
#else
    throw std::runtime_error("GPU not supported in this build.");
#endif
  }
}
}// namespace mlkl