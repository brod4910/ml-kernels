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

  if (x > N || y > M) {
    return;
  }

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
    accumulator += a[x * K + k] * b[k * N + y];
  }

  c[x * N + y] = alpha * accumulator + beta * c[x * N + y];
}

// Shared memory utilization single-thread responsible for single-output
template<int block_size>
__global__ void sgemm_v3(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ float shared_a[block_size * block_size];
  __shared__ float shared_b[block_size * block_size];

  float accumulator = 0.0;

  int thread_x = threadIdx.x % block_size;
  int thread_y = threadIdx.x / block_size;

  a += blockIdx.x * block_size * K;
  b += blockIdx.y * block_size;
  c += blockIdx.x * block_size * N + blockIdx.y * block_size;

  for (int k = 0; k < K; k += block_size) {

    shared_a[thread_y * block_size + thread_x] = a[thread_y * K + thread_x];
    shared_b[thread_y * block_size + thread_x] = b[thread_y * N + thread_x];

    __syncthreads();

    for (int i = 0; i < block_size; ++i) {
      accumulator += shared_a[thread_y * block_size + i] * shared_b[i * block_size + thread_y];
    }

    a += block_size;
    b += block_size * N;

    __syncthreads();
  }

  c[thread_y * N + thread_x] = alpha * accumulator + beta * c[thread_y * N + thread_x];
}

// Shared memory utilization
template<int block_size>
__global__ void sgemm_v4(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
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

void launch_sgemm_v3(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, 32), ceil_div(N, 32));
  dim3 block_dim(32 * 32);
  kernel::sgemm_v3<32><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_sgemm_v4(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, 32), ceil_div(N, 32));
  dim3 block_dim(32 * 32);
  kernel::sgemm_v4<32><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

}// namespace ml::operators::cuda