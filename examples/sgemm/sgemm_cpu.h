//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include "luna_cpu/operators/gemm.h"
#include <memory>
#include <iostream>

void initialize_matrix_cpu(float* matrix, size_t size, float value, int skip = 1) {
    for (size_t i = 0; i < size; i += skip) {
        matrix[i] = value;
    }
}

void print_matrix_cpu(const float* matrix, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

void sgemm_cpu(size_t M, size_t N, size_t K, float alpha, float beta) {
  auto* a = new float[M * K];
  auto* b = new float[K * N];
  auto* c = new float[M * N];
  initialize_matrix_cpu(a, M * K, 1, 1);
  initialize_matrix_cpu(b, N * K, 2, 1);
  initialize_matrix_cpu(c, M * N, 0, 1);

  luna::operators::cpu::sgemm(a, alpha, b, beta, c, M, N, K);

//  print_matrix_cpu(c, M, N);
  delete[] a, delete[] b, delete[] c;
}