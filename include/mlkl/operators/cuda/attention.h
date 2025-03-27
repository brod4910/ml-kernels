#pragma once
#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
void attention_v1(Tensor *q, Tensor *k, Tensor *v, Tensor *output);

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output);
}// namespace mlkl::operators::cuda
