
#pragma once
#include <cstddef>
#include <vector>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
void softmax(Tensor *input, Tensor *output, int dim);
}// namespace mlkl::operators::cpu