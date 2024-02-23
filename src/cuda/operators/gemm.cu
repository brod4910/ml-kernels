#include <cuda_runtime_api.h>

#include <mlkl/cuda/operators/gemm.h>

namespace ml::operators::cuda {
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

namespace kernel {
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

  c[x * N + y] = alpha * accumulator + beta * c[x * N + y];
}
}// namespace kernel

void launch_sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  dim3 grid_dim(ceil_div(M, blk_size), ceil_div(N, blk_size));
  dim3 block_dim(blk_size, blk_size);

  kernel::sgemm_v1<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size) {
  dim3 grid_dim(ceil_div(M, blk_size), ceil_div(N, blk_size));
  dim3 block_dim(blk_size * blk_size);
  kernel::sgemm_v2<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K, blk_size);
}

}// namespace ml::operators::cuda