#pragma once

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
Tensor empty(std::vector<int> &shape);

void fill(Tensor &tensor, int value);

void copy(Tensor &src, Tensor &dst);

void destroy(Tensor &tensor);

Tensor randn(std::vector<int> &shape);

void randn(Tensor &tensor);
}// namespace mlkl::operators::cuda