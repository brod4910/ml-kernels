#include <cuda_runtime.h>

__global__ void gemm(const float *__restrict__ a, float alpha, const float *__restrict__ b, float beta, float *__restrict__ c, size_t M, size_t N, size_t K) {
}