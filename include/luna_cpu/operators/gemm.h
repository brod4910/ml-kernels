//
// Created by Brian Rodriguez on 8/26/23.
//

#pragma once
#include <cstddef>

namespace luna::operators {
void sgemm(const float *a, const float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);

} // namespace luna_cpu::math
