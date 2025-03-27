#include <stdexcept>

#include <mlkl/core/tensor_math.h>
#include <mlkl/operators/operators.h>

namespace mlkl {
void softmax(Tensor *input, Tensor *output, int dim, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::softmax(input, output, dim);
  } else if (device == Device::CUDA) {
    return operators::cuda::softmax(input, output, dim);
  }
}

void sgemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::sgemm(a, b, c, alpha, beta);
  } else if (device == Device::CUDA) {
    return operators::cuda::sgemm(a, b, c, alpha, beta);
  }
}

void bf16_gemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta, Device device) {
  if (device == Device::CPU) {
    throw std::runtime_error("CPU bf16 gemm not implemented.");
  } else if (device == Device::CUDA) {
    return operators::cuda::bf16_gemm(a, b, c, alpha, beta);
  }
}

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output, Device device) {
  if (device == Device::CPU) {
    operators::cpu::attention(q, k, v, output);
  } else if (device == Device::CUDA) {
    throw std::runtime_error("GPU attention not implemented.");
  }
}

}// namespace mlkl