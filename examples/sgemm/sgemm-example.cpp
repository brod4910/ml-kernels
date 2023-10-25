//
// Created by Brian Rodriguez on 8/26/23.
//
//
// Created by Brian Rodriguez on 6/17/23.
//
#ifdef __AVX2__
#include "sgemm_avx.h"
#endif

#include "sgemm_cpu.h"

int main() {
    size_t M = 32;
    size_t N = 32;
    size_t K = 32;
    float alpha = 1.0;
    float beta = 0.0;

    std::cout << "CPU" << std::endl;
    sgemm_cpu(M, N, K, alpha, beta);
#ifdef __AVX2__
    std::cout << "AVX2" << std::endl;
    sgemm_avx(M, N, K, alpha, beta);
#endif
}
