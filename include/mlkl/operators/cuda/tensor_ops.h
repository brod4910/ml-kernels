#pragma once

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
Tensor empty(std::vector<int> &shape);

void fill(Tensor &tensor, int value);

void copy(Tensor &src, Device src_device, Tensor &dst, Device dst_device);

void destroy(Tensor &tensor);

Tensor randn(std::vector<int> &shape);

void randn(Tensor &tensor);
}// namespace mlkl::operators::cuda