//
// Created by Brian Rodriguez on 10/29/23.
//

#pragma once
#include <cstddef>

namespace luna::operators::avx {
void transpose(const float *__restrict__ a, float * __restrict__ b, size_t M, size_t N);

}// namespace luna::operators::avx
