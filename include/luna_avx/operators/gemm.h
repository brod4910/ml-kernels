//
// Created by Brian Rodriguez on 10/20/23.
//

#pragma once
#include <cstddef>

namespace luna::operators::avx {
void sgemm(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);

}// namespace luna::operators::avx