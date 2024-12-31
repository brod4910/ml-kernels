
#pragma once
#include <cstddef>
#include <vector>

namespace ml::operators::cpu {
void softmax(const float *__restrict__ input, float *__restrict__ output, int dim, std::vector<int> &shape);
}// namespace ml::operators::cpu