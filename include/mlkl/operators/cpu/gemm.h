//
// Created by Brian Rodriguez on 8/26/23.
//

#pragma once
#include <cstddef>
#include <mlkl/core/tensor.h>

namespace mlkl::operators::cpu {
void sgemm(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta);

}// namespace mlkl::operators::cpu
