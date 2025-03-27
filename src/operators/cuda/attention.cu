#include "mlkl/core/tensor.h"
#include <cstddef>
#include <cstdio>

#include <device_types.h>
#include <mma.h>
#include <vector_types.h>

#include <mlkl/core/basic_math.h>
#include <mlkl/operators/cuda/bf16_gemm.h>

// TODO: Delete this and make functions templates
#define WARP_SIZE 32

using namespace nvcuda;

namespace mlkl::operators::cuda {
namespace kernel {
// naive
__global__ __launch_bounds__(32) void attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim) {
}
}// namespace kernel

namespace {
void launch_attention_v1(const fp32 *q, const fp32 *k, const fp32 *v, fp32 *output, int batch_size, int num_heads, int seq_len, int head_dim) {
  // kernel::attention_v1<WM, WN, WK><<<grid_dim, block_dim>>>(q, k, v, output, batch_size, num_heads, seq_len, head_dim);
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
