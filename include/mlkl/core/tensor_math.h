#pragma once
#include <mlkl/core/tensor.h>

namespace mlkl {
void softmax(Tensor *input, Tensor *output, int dim, Device device);

void sgemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta, Device device);

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output, Device device);
}// namespace mlkl