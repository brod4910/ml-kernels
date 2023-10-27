//
// Created by Brian Rodriguez on 10/20/23.
//

#pragma once
#include <cstddef>
#include <immintrin.h>

namespace luna::operators::avx {
void sgemm(const __m256 *a, float alpha, const __m256 *b, float beta, __m256 *c, size_t M, size_t N, size_t K);

}// namespace luna::operators::avx