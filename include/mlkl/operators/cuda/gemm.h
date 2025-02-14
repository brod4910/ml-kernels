#pragma once
#include <cstddef>
#include <cstdio>

#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
namespace {
void launch_sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
void launch_sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
void launch_sgemm_v3(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
void launch_sgemm_v4(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
void launch_sgemm_v5(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
void launch_sgemm_v6(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
}// namespace

void sgemm_v1(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v1(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}

void sgemm_v2(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v2(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}

void sgemm_v3(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v3(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}

void sgemm_v4(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v4(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}

void sgemm_v5(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v5(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}

void sgemm_v6(Tensor &a, Tensor &b, Tensor &c, float alpha, float beta) {
  launch_sgemm_v6(a.data, alpha, b.data, beta, c.data, c.shape[0], c.shape[1], a.shape[1]);
}
}// namespace mlkl::operators::cuda
