#include <cmath>
#include <mlkl/operators/cpu/attention.h>

namespace mlkl::operators::cpu {
namespace {
void softmax(float *input, float *output, int seq_len_q, int seq_len_kv, int head_dim) {
  for (int slq = 0; slq < seq_len_q; ++slq) {
    float curr_max = -std::numeric_limits<float>::infinity();
    float norm_factor = 0.f;
    for (int slkv = 0; slkv < seq_len_kv; ++slkv) {
      float new_max = std::fmax(input[slq * seq_len_kv + slkv], curr_max);
      float correction = std::exp(curr_max - new_max);

      norm_factor = (norm_factor * correction) + std::exp(input[slq * seq_len_kv + slkv] - new_max);
      curr_max = new_max;
    }

    for (int slkv = 0; slkv < seq_len_kv; ++slkv) {
      output[slq * seq_len_kv + slkv] = std::exp(input[slq * seq_len_kv + slkv] - curr_max) / norm_factor;
    }
  }
}

void head_attention(const float *__restrict__ q,
                    const float *__restrict__ k,
                    const float *__restrict__ v,
                    float *__restrict__ output,
                    int seq_len_q, int seq_len_kv, int head_dim) {
  float attn_scale = 1 / std::sqrt((float) head_dim);
  float *scores = new float[seq_len_q * seq_len_kv]();

  for (int slq = 0; slq < seq_len_q; ++slq) {
    for (int slkv = 0; slkv < seq_len_kv; ++slkv) {
      for (int d = 0; d < head_dim; ++d) {
        scores[slq * seq_len_kv + slkv] += q[slq * head_dim + d] * k[d * seq_len_kv + slkv];
      }
    }
  }

  for (int slq = 0; slq < seq_len_q; ++slq) {
    for (int slkv = 0; slkv < seq_len_kv; ++slkv) {
      scores[slq * seq_len_kv + slkv] *= attn_scale;
    }
  }

  softmax(scores, scores, seq_len_q, seq_len_kv, head_dim);

  for (int slq = 0; slq < seq_len_q; ++slq) {
    for (int d = 0; d < head_dim; ++d) {
      for (int slkv = 0; slkv < seq_len_kv; ++slkv) {
        output[slq * head_dim + d] += scores[slq * seq_len_kv + slkv] * v[slkv * head_dim + d];
      }
    }
  }
}

void attention(const float *__restrict__ q,
               const float *__restrict__ k,
               float *__restrict__ v,
               float *output,
               int batch_size, int num_heads, int seq_len_q, int seq_len_kv, int head_dim) {
  for (int b = 0; b < batch_size; ++b) {
    for (int head = 0; head < num_heads; ++head) {
      const float *q_head = &q[b * num_heads * seq_len_q * head_dim + head * seq_len_q * head_dim];
      const float *k_head = &k[b * num_heads * seq_len_kv * head_dim + head * seq_len_kv * head_dim];
      const float *v_head = &v[b * num_heads * seq_len_kv * head_dim + head * seq_len_kv * head_dim];
      float *o_head = &output[b * num_heads * seq_len_q * head_dim + head * seq_len_q * head_dim];

      head_attention(q_head, k_head, v_head, o_head, seq_len_q, seq_len_kv, head_dim);
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