//
// Created by Brian Rodriguez on 2/21/24.
//

#pragma once

namespace ml::operators::cuda::kernel {
__global__ void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);

__global__ void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);

};
