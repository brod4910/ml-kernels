#pragma once
#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
void bf16_gemm_v1(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void bf16_gemm_v2(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void bf16_gemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);
}// namespace mlkl::operators::cuda
