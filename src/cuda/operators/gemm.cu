#include <cuda_runtime.h>
namespace ml::operators::cuda::kernel {
// naive
__global__ void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x > M || y > N) {
    return;
  }

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
    accumulator += a[x * K + k] * b[k * N + y];
  }

  c[x * N + y] = alpha * accumulator + beta * c[x * N + y];
}

// Global Memory Coalescing, vectorize 2-D indices to 1-D
__global__ void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  int x = blockDim.x * blk_size + (threadIdx.x / blk_size);
  int y = blockDim.y * blk_size + (threadIdx.x % blk_size);

  if (x > M || y > N) {
    return;
  }

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
      accumulator += a[x * K + k] * b[k * N + y];
    }
  }

  c[x * N + y] = alpha * accumulator + beta * c[x * N + y];
}
}