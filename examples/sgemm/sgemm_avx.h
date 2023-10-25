//
// Created by Brian Rodriguez on 10/24/23.
//

#pragma once

#include <luna_avx/operators/gemm.h>
#include <iostream>

void initialize_matrix(float* matrix, size_t size, float value, int skip = 1) {
    for (size_t i = 0; i < size; i += skip) {
        matrix[i] = value;
    }
}

void print_matrix(const float* matrix, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

void sgemm_avx(size_t M, size_t N, size_t K, float alpha, float beta) {
    auto *a = static_cast<__m256*>(_mm_malloc(32, M * K * sizeof(float)));
    auto *b = static_cast<__m256*>(_mm_malloc(32, N * K * sizeof(float)));
    auto *c = static_cast<__m256*>(_mm_malloc(32, M * N * sizeof(float)));
    initialize_matrix(reinterpret_cast<float*>(a), M * K, 1);
    initialize_matrix(reinterpret_cast<float*>(b), N * K, 2);
    initialize_matrix(reinterpret_cast<float*>(c), M * N, 0);

    luna::operators::avx::sgemm(a, alpha, b, beta, c, M, N, K);
    print_matrix(reinterpret_cast<float*>(c), M, N);

    _mm_free(a);
    _mm_free(b);
    _mm_free(c);
}
