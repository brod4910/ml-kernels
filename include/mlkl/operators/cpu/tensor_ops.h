#pragma once

#include <cstdlib>
#include <vector>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
Tensor empty(std::vector<int> &shape);

void fill(Tensor &tensor, int value);

void destroy(Tensor &tensor);

Tensor randn(std::vector<int> &shape);

void randn(Tensor &tensor);
}// namespace mlkl::operators::cpu