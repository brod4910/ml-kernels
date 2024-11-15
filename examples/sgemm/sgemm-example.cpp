//
// Created by Brian Rodriguez on 8/26/23.
//
//
// Created by Brian Rodriguez on 6/17/23.
//
#ifdef __AVX2__
#include "sgemm_avx.h"
#endif

#ifdef __CUDA__
#include "sgemm_cuda.h"
#endif

#include "sgemm_cpu.h"

int main() {
  size_t M = 1024;
  size_t N = 1024;
  size_t K = 1024;
  float alpha = 1.0;
  float beta = 0.0;

//   std::cout << "CPU" << std::endl;
//   sgemm_cpu(M, N, K, alpha, beta);
#ifdef __AVX2__
  std::cout << "AVX2" << std::endl;
  sgemm_avx(M, N, K, alpha, beta);
#endif
#ifdef __CUDA__
  std::cout << "CUDA" << std::endl;
  sgemm_cuda(M, N, K, alpha, beta);
#endif
}
