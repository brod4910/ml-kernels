#include <cuda_runtime.h>

// naive
__global__ void gemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.x;

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
    accumulator += a[x * K + k] * b[y * K + k];
  }

  float cv = c[x * M + y];
  c[x * M + y] = alpha * accumulator + beta * cv;
}

// Global Memory Coalescing
__global__ void gemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
}