//
// Created by Brian Rodriguez on 2/21/24.
//
#include "cuda/operators/gemm.cuh"

#include "cuda_runtime.h"


namespace ml::operators::cuda {
void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {

  dim3 grid_dim(M / blk_size, N / blk_size);
  dim3 block_dim(blk_size, blk_size);

  kernel::sgemm_v1<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  dim3 gridDim(M / blk_size, N / blk_size);
  dim3 blockDim(blk_size * blk_size);
  kernel::sgemm_v2<<<gridDim, blockDim>>>(a, alpha, b, beta, c, M, N, K, blk_size);
}
}