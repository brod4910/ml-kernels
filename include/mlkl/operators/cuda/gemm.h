#pragma once
#include <cstdio>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
void sgemm_v1(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm_v2(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm_v3(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm_v4(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm_v5(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm_v6(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);

void sgemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta);
}// namespace mlkl::operators::cuda
