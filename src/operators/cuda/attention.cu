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
template<int WM, int WN, int WK>
__global__ __launch_bounds__(32) void attention_v1(const fp32 *q, const fp32 *k, fp32 *v, size_t M, size_t N, size_t K) {
  // suppose we are processing block (2, 3) and WM = WN = WK = 16
  // warp_col = 2 * WN = 2 * 16 = 32
  // warp row = 3 * WN = 3 * 16 = 48
  //
  // block (7, 7)
  // warp_col = 7 * WN = 7 * 16 = 112
  // warp row = 7 * WN = 7 * 16 = 112
  int warp_row = blockIdx.y * WM;
  int warp_col = blockIdx.x * WN;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, bf16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, bf16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> accumulator;
  wmma::fill_fragment(accumulator, 0.);

  for (int step = 0; step < K / WK; ++step) {

    wmma::load_matrix_sync(a_frag, &a[(warp_row * WM) * K + (step * WK)], K);
    wmma::load_matrix_sync(b_frag, &b[(step * WK) * N + (warp_col * WN)], N);

    wmma::mma_sync(accumulator, a_frag, b_frag, accumulator);
  }

  wmma::store_matrix_sync(&c[warp_row * N + warp_col], accumulator, N, wmma::mem_row_major);
}
}// namespace kernel

namespace {
void launch_attention_v1(const fp32 *a, float alpha, const fp32 *b, float beta, fp32 *c, size_t M, size_t N, size_t K) {
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;

  // if M = 128, M = 128
  // (128 / WN, 128 / WM) = (128 / 16, 128 / 16)
  // so we will launch a 8x8 grid where each block is comprised
  // a collection of 32-threads
  dim3 block_dim(WARP_SIZE);
  dim3 grid_dim(math::ceil_div(N, WN), math::ceil_div(M, WM));

  kernel::attention_v1<WM, WN, WK><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}
}// namespace

void attention_v1(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  launch_attention_v1(q->fp32_(), k->fp32_(), v->fp32_(), c->shape[0], c->shape[1], a->shape[1]);
}

void attention(Tensor *q, Tensor *k, Tensor *v, Tensor *output) {
  attention_v1(q, k, v, output);
}
}// namespace mlkl::operators::cuda
