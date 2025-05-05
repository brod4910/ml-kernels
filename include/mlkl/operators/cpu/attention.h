#pragma once
#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output);
}// namespace mlkl::operators::cpu
