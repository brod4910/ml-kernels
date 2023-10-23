//
// Created by Brian Rodriguez on 8/26/23.
//
//
// Created by Brian Rodriguez on 6/17/23.
//
#include "luna/operators/gemm.h"
#include <memory>

void initialize_matrix(float* matrix, size_t size, float value, int skip = 1) {
  for (size_t i = 0; i < size; i += skip) {
    matrix[i] = value;
  }
}

int main() {
  size_t M = 1;
  size_t N = 1;
  size_t K = 2;

  auto* a = new float[M * K];
  auto* b = new float[K * N];
  auto* c = new float[M * N];
  initialize_matrix(a, M * K, 1, 1);
  initialize_matrix(b, N * K, 2, 1);
  initialize_matrix(c, M * N, 0, 1);
  luna::operators::sgemm(a, 1.0, b, 1.0, c, M, N, K);

  delete[] a, delete[] b, delete[] c;

  return 0;
}