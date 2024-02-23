//
// Created by Brian Rodriguez on 2/21/24.
//
#include <mlkl/cuda/operators/gemm.h>

#include "cuda_runtime.h"


namespace ml::operators::cuda {
void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  launch_sgemm_v1(a, alpha, b, beta, c, M, N , K, blk_size);
}

void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  launch_sgemm_v2(a, alpha, b, beta, c, M, N , K, blk_size);
}
}