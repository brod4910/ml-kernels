//
// Created by Brian Rodriguez on 8/26/23.
//

#pragma once
#include <cstddef>

namespace ml::operators::cpu {
void sgemm(const float *__restrict__ a, float alpha, const float *__restrict__ b, float beta, float *__restrict__ c, size_t M, size_t N, size_t K);

}// namespace ml::operators::cpu
