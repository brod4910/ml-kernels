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
// naive
__global__ void attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim) {
  // The strange thing about a naive implementation is the intermediate S = Q @ K.T matrix that is produced
  // if we take q and k.T in blocks of 64x64 then we are computing 64x64 elements of the intermediate S
  // even for a naive implementation that doesn't consume tons of memory, we'll need to use shared memory
  // to calculate the S matrix

  int head = blockIdx.y % head_dim;
  int batch = blockIdx.y / head_dim;
  int qk_index = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ fp32 S[16 * 16];
  __shared__ fp32 M[16];// row max
  __shared__ fp32 L[16];// norm factors

  S[tid] = 0.;
  if (tid % 16 == 0) {
    M[tid / 16] = -CUDART_INF_F;
    L[tid / 16] = 0.;
  }

  for (int hd = 0; hd < head_dim; hd += 16) {
    const fp32 *q_block = q + (batch * num_heads * seq_len * head_dim) + (head * seq_len * head_dim) + (qk_index * head_dim) + hd;
    const fp32 *k_block = k + (batch * num_heads * seq_len * head_dim) + (head * seq_len * head_dim) + (qk_index * head_dim) + hd;

    for (int d = 0; d < 16; ++d) {
      S[tid] += q_block[d] * k_block[d];
    }
  }

  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      fmaxf(M[y], S[])
    }
  }
}
}// namespace kernel

namespace {
void launch_attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim) {
  dim3 block_dim(256);
  dim3 grid_dim(math::ceil_div(seq_len, block_dim.x), batch_size * num_heads);
  kernel::attention_v1<<<grid_dim, block_dim>>>(q, k, v, output, batch_size, num_heads, seq_len, head_dim);
}
}// namespace

void attention_v1(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  int batch_size = q->shape[0];
  int num_heads = q->shape[1];
  int seq_len = q->shape[2];
  int head_dim = q->shape[3];

  launch_attention_v1(q->fp32_(), k->fp32_(), v->fp32_(), output->fp32_(), batch_size, num_heads, seq_len, head_dim);
}

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  attention_v1(q, k, v, output);
}
}// namespace mlkl::operators::cuda
