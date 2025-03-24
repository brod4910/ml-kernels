#include <cmath>
#include <mlkl/operators/cpu/gemm.h>

namespace mlkl::operators::cpu {
namespace {
void attention(const float *__restrict__ q,
               const float *__restrict__ k,
               float *__restrict__ v,
               float *output,
               int batch_size, int num_heads, int seq_len, int head_dim) {
  float sqrt_hd = sqrtf((float) head_dim);

  for (int b = 0; b < batch_size; ++b) {
    for (int nh = 0; nh < num_heads; ++nh) {
      for (int sl = 0; sl < seq_len; ++sl) {
        for (int d = 0; d < head_dim; ++d) {
          dot = q[b * num_heads * seq_len * head_dim + nh * seq_len * head_dim + sl * head_dim + d] * k[d * seq_len * num_heads * batch_size + sl * num_heads * batch_size + nh * batch_size + b];
        }
      }
    }
  }
}
}// namespace

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  int batch_size = q->shape[0];
  int num_heads = q->shape[1];
  int seq_len = q->shape[2];
  int head_dim = q->shape[3];

  attention(q->fp32_(), k->fp32_(), v->fp32_(), output->fp32_(), batch_size, num_heads, seq_len, head_dim);
}
}// namespace mlkl::operators::cpu