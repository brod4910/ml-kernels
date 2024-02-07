
#pragma once
#include <cstddef>

namespace ml::operators::cuda {
void sgemm(const float *__restrict__ a, float alpha, const float *__restrict__ b, float beta, float *__restrict__ c, size_t M, size_t N, size_t K);
} // namespace ml::operators::cuda
