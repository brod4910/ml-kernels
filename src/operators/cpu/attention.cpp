#include <cmath>
#include <mlkl/operators/cpu/attention.h>

namespace mlkl::operators::cpu {
namespace {
void softmax(const float *__restrict__ input, float *__restrict__ output, int batch_size, int num_heads, int seq_len_q, int seq_len_kv, int head_dim) {

  for (int b = 0; b < batch_size; ++b) {
    for (int head = 0; head < num_heads; ++head) {
      for (int i = 0; i < seq_len_q; ++i) {
        float curr_max = -std::numeric_limits<float>::infinity();
        float norm_factor = 0.f;
        for (int i = 0; i < seq_len_kv; ++i) {
          float new_max = std::fmax(input[i], curr_max);
          float correction = std::exp(curr_max - new_max);

          norm_factor = (norm_factor * correction) + std::exp(input[i] - new_max);
          curr_max = new_max;
        }

        for (int i = 0; i < seq_len_kv; ++i) {
          output[b * num_heads * seq_len_q * seq_len_kvi] = std::exp(input[i] - curr_max) / norm_factor;
        }
      }
    }
  }
}

void attention(const float *__restrict__ q,
               const float *__restrict__ k,
               float *__restrict__ v,
               float *output,
               int batch_size, int num_heads, int seq_len_q, int seq_len_kv, int head_dim) {
  float attn_scale = 1 / std::sqrt((float) head_dim);

  float *scores = new float[seq_len_q * seq_len_kv]();
  float *attn = new float[seq_len_q * seq_len_kv]();

  for (int b = 0; b < batch_size; ++b) {
    for (int head = 0; head < num_heads; ++head) {
      for (int slq = 0; slq < seq_len_q; ++slq) {
        for (int slk = 0; slk < seq_len_kv; ++slk) {
          for (int d = 0; d < head_dim; ++d) {
            scores[slq * seq_len_kv + slk] += q[b * num_heads * seq_len_q * head_dim + head * seq_len_q * head_dim + slq * head_dim + d] * k[d * seq_len_kv * num_heads * batch_size + slk * num_heads * batch_size + head * batch_size + b];
          }
        }
      }
    }
  }

  for (int slq = 0; slq < seq_len_q; ++slq) {
    for (int slk = 0; slk < seq_len_kv; ++slk) {
      scores[slq * seq_len_kv + slk] *= attn_scale;
    }
  }
}
}// namespace

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  int batch_size = q->shape[0];
  int num_heads = q->shape[1];
  int seq_len_q = q->shape[2];
  int seq_len_k = k->shape[2];
  int head_dim = q->shape[3];

  attention(q->fp32_(), k->fp32_(), v->fp32_(), output->fp32_(), batch_size, num_heads, seq_len_q, seq_len_k, head_dim);
}
}// namespace mlkl::operators::cpu