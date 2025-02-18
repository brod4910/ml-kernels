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
}// namespace mlkl