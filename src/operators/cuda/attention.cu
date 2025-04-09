#include <cstddef>
#include <cstdio>

#include <device_types.h>
#include <math_constants.h>
#include <mma.h>
#include <vector_types.h>

#include "mlkl/core/tensor.h"
#include <mlkl/core/basic_math.h>
#include <mlkl/operators/cuda/bf16_gemm.h>

// TODO: Delete this and make functions templates
#define WARP_SIZE 32

using namespace nvcuda;

namespace mlkl::operators::cuda {
namespace kernel {

__device__ void off_band_S(fp32 *s, fp32 *m, fp32 *l, fp32 qk_scale, int block_dx, int block_dy) {
  fp32 m_ij[16] = {-CUDART_INF_F};// intermediate row max
  fp32 l_ij[16] = {0.};           // intermediate norm factors
  fp32 alpha[16];

  for (int y = 0; y < block_dy; ++y) {
    for (int x = 0; x < block_dx; ++x) {
      float new_max = fmaxf(m_ij[y], s[y * block_dx + x]) * qk_scale;
      m_ij[y] = new_max;
    }
  }

  for (int y = 0; y < block_dy; ++y) {
    for (int x = 0; x < block_dx; ++x) {
      s[y * block_dx + x] = exp2f(s[y * block_dx + x] * qk_scale - m_ij[y]);
    }
  }

  for (int y = 0; y < block_dy; ++y) {
    for (int x = 0; x < block_dx; ++x) {
      l_ij[y] += s[y * block_dx + x];
    }

    alpha[y] = exp2f(m[y] - m_ij[y]);
    m[y] = m_ij[y];
    l[y] = l[y] * alpha[y] + l_ij[y];
  }
}

__device__ void off_band_range(int *lo, int *hi) {
}

__device__ void on_band_S() {
}

__device__ void on_band_range(int *lo, int *hi) {
}

__device__ void
s_max(fp32 *s, fp32 *m_ij, fp32 qk_scale, int block_dx, int block_dy) {
}

__device__ void softmax_star(fp32 *s, fp32 *m_ij, fp32 qk_scale, int block_dx, int block_dy) {
}

__device__ void nf_sum(fp32 *s, fp32 *l_ij, int block_dx, int block_dy) {
  for (int y = 0; y < block_dy; ++y) {
    for (int x = 0; x < block_dx; ++x) {
      l_ij[y] += s[y * x];
    }
  }
}
// naive
__global__ void
attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim, float qk_scale) {
  // The strange thing about a naive implementation is the intermediate S = Q @ K.T matrix that is produced
  // if we take q and k.T in blocks of 64x64 then we are computing 64x64 elements of the intermediate S
  // even for a naive implementation that doesn't consume tons of memory, we'll need to use shared memory
  // to calculate the S matrix

  int head = blockIdx.y % head_dim;
  int batch = blockIdx.y / head_dim;
  int qk_index = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ fp32 s[16 * 16];
  __shared__ fp32 m[16];// row max
  __shared__ fp32 l[16];// norm factors

  s[tid] = 0.;
  if (tid % 16 == 0) {
    m[tid / 16] = -CUDART_INF_F;
    l[tid / 16] = 0.;
  }

  for (int hd = 0; hd < head_dim; hd += 16) {
    const fp32 *q_block = q + (batch * num_heads * seq_len * head_dim) + (head * seq_len * head_dim) + (qk_index * head_dim) + hd;
    const fp32 *k_block = k + (batch * num_heads * seq_len * head_dim) + (head * seq_len * head_dim) + (qk_index * head_dim) + hd;

    for (int d = 0; d < 16; ++d) {
      s[tid] += q_block[d] * k_block[d];
    }
  }

  __syncthreads();

  if (tid == 0) {
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 16; ++x) {
        s[y * x] = expf(s[y * x] - m[y]);// / l[y];
      }
    }
  }

  __syncthreads();
}
}// namespace kernel

namespace {
void launch_attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim, float qk_scale) {
  dim3 block_dim(256);
  dim3 grid_dim(math::ceil_div(seq_len, block_dim.x), batch_size * num_heads);
  kernel::attention_v1<<<grid_dim, block_dim>>>(q, k, v, output, batch_size, num_heads, seq_len, head_dim, qk_scale);
}
}// namespace

void attention_v1(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  int batch_size = q->shape[0];
  int num_heads = q->shape[1];
  int seq_len = q->shape[2];
  int head_dim = q->shape[3];

  float qk_scale = (1 / sqrt(head_dim)) * 1.44269504;// 1 / log(2);

  launch_attention_v1(q->fp32_(), k->fp32_(), v->fp32_(), output->fp32_(), batch_size, num_heads, seq_len, head_dim, qk_scale);
}

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  attention_v1(q, k, v, output);
}
}// namespace mlkl::operators::cuda
