#pragma once
#include <cstdio>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
void softmax(Tensor *input, Tensor *output, int dim);
}// namespace mlkl::operators::cuda