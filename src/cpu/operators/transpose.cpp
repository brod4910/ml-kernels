//
// Created by Brian Rodriguez on 12/2/23.
//
#include <luna_cpu/operators/transpose.h>

namespace ml::operators::cpu {
void transpose(const float *__restrict__ a, float *__restrict__ b, size_t M, size_t N) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      b[n * M + m] = a[m * N + n];
    }
  }
}
}