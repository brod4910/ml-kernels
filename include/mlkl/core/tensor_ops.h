#pragma once
#include <mlkl/core/tensor.h>

#include <vector>

namespace mlkl {
Tensor empty(std::vector<int> &shape, Device device);

void fill(Tensor &tensor, int value);

void destroy(Tensor &tensor);

Tensor randn(std::vector<int> &shape, Device device);

void randn(Tensor &tensor);

void to(Tensor &tensor, Device device);

bool equals(Tensor &a, Tensor &b, float epsilon = 1e-6);
}// namespace mlkl