#include <cmath>
#include <limits>

#include <mlkl/cpu/operators/softmax.h>

namespace ml::operators::cpu {
void softmax(const float *__restrict__ input, float *__restrict__ output, int dim, std::vector<int> &shape) {
  float curr_max = -std::numeric_limits<float>::infinity();
  float norm_factor = 0.f;

  for (int i = 0; i < shape[dim]; ++i) {
    float new_max = std::max(input[i], curr_max);
    float correction = std::exp(curr_max - new_max);

    norm_factor = (norm_factor * correction) + std::exp(input[i] - new_max);
    curr_max = new_max;
  }

  for (int i = 0; i < shape[dim]; ++i) {
    output[i] = std::exp(input[i] - curr_max) / norm_factor;
  }
}
}// namespace ml::operators::cpu