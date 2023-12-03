//
// Created by Brian Rodriguez on 12/2/23.
//
#pragma once
#include <cstddef>

namespace luna::operators::cpu {
void transpose(const float *__restrict__ a, float * __restrict__ b, size_t M, size_t N);
}
