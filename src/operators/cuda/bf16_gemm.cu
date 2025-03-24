#include <cstddef>
#include <cstdio>

#include <device_types.h>
#include <vector_types.h>
#include <mma.h>

#include <mlkl/core/basic_math.h>
#include <mlkl/operators/cuda/bf16_gemm.h>

// TODO: Delete this and make functions templates
#define WARP_SIZE 32

using namespace nvcuda;

namespace mlkl::operators::cuda {
namespace kernel {
// naive
template<int WM, int WN, int WK>
__global__ __launch_bounds__(32) void bf16_gemm_v1(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
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

// Adding shared memory to kernel still bounded by WARP-SIZE threads
template<int WM, int WN, int WK>
__global__ __launch_bounds__(32) void bf16_gemm_v2(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ bf16 As[WM * WK];// 16 * 16 = 256
  __shared__ bf16 Bs[WK * WN];// 16 * 16 = 256

  int warp_row = blockIdx.y * WM;
  int warp_col = blockIdx.x * WN;
  int tid = threadIdx.x;// lane-id

  // num elements to load per thread into SMEM
  constexpr int ld_offset_a = WM * WK / WARP_SIZE;// 16 * 16 / 32 = 8
  constexpr int ld_offset_b = WK * WN / WARP_SIZE;// 16 * 16 / 32 = 8

  wmma::fragment<wmma::matrix_a, WM, WN, WK, bf16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, bf16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> accumulator;
  wmma::fill_fragment(accumulator, 0.);

  for (int step = 0; step < K / WK; ++step) {
    const bf16 *a_tile = &a[(warp_row * WM) * K + (step * WK)];
    const bf16 *b_tile = &b[(step * WK) * N + (warp_col * WN)];

    // each thread needs to load some global memory values into shared memory.
    // the access pattern doesn't matter as much just as long as we
    // coalesce memory reads. We jump by some number of offets and use the tid to
    // read the appropriate values from global.
    // In this case, we jump by the warp-size since in this kernel we are launching
    // each block with warp-size threads so calculation is a bit easier
    for (int offset_a = 0; offset_a < ld_offset_a * WARP_SIZE; offset_a += WARP_SIZE) {
      As[tid + offset_a] = a_tile[tid + offset_a];
    }

    for (int offset_b = 0; offset_b < ld_offset_b * WARP_SIZE; offset_b += WARP_SIZE) {
      Bs[tid + offset_b] = b_tile[tid + offset_b];
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, As, WK);
    wmma::load_matrix_sync(b_frag, Bs, WN);

    wmma::mma_sync(accumulator, a_frag, b_frag, accumulator);
  }

  wmma::store_matrix_sync(&c[warp_row * N + warp_col], accumulator, N, wmma::mem_row_major);
}

// increase the number of warps per block to 8 -> 256 / 32 = 8
// increase shared memory to fit 8 warps worth of work
// As = BM * BK = 16 * 16 = 2048
// Bs = WK * WN * 8 = 16 * 16 * 8 = 2048
template<int BM, int BN, int BK, int WM = 16, int WN = 16, int WK = 16, int NUM_THREADS = 256>
__global__ __launch_bounds__(256) void bf16_gemm_v3(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ bf16 As[BM * BK];// BM * BK =
  __shared__ bf16 Bs[BK * BN];// BK * BN =

  int tid = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE; // lane-id
  int warp_num = threadIdx.x / WARP_SIZE;// linear warp-num
  // 8 warps distributed in a 2x4 fashion within the block
  // a standard shape for work amongst a block
  int warp_col = warp_num % 4;
  int warp_row = warp_num / 4;

  // num elements to load per thread into SMEM
  constexpr int ld_elems_a = BM * BK / NUM_THREADS;// 128 * 128 / 256 = 64
  constexpr int ld_elems_b = BK * BN / NUM_THREADS;// 128 * 128 / 256 = 64

  wmma::fragment<wmma::matrix_a, WM, WN, WK, bf16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, bf16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> accumulator;
  wmma::fill_fragment(accumulator, 0.);

  for (int step = 0; step < K / BK; ++step) {
    const bf16 *a_tile = &a[(warp_row * WM) * K + (step * WK)];
    const bf16 *b_tile = &b[(step * WK) * N + (warp_col * WN)];

    // each thread needs to load some global memory values into shared memory.
    // the access pattern doesn't matter as much just as long as we
    // coalesce memory reads. We jump by some number of offets and use the tid to
    // read the appropriate values from global.
    // In this case, we jump by the warp-size since in this kernel we are launching
    // each block with warp-size threads so calculation is a bit easier
    for (int offset_a = 0; offset_a < ld_elems_a * NUM_THREADS; offset_a += NUM_THREADS) {
      As[tid + offset_a] = a_tile[tid + offset_a];
    }

    for (int offset_b = 0; offset_b < ld_elems_b * NUM_THREADS; offset_b += NUM_THREADS) {
      Bs[tid + offset_b] = b_tile[tid + offset_b];
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, &As[warp_row * WM + (step * WK)], WK);
    wmma::load_matrix_sync(b_frag, &Bs[step * WK + (warp_row * WN)], WN);

    wmma::mma_sync(accumulator, a_frag, b_frag, accumulator);
  }

  wmma::store_matrix_sync(&c[warp_row * N + warp_col], accumulator, N, wmma::mem_row_major);
}
}// namespace kernel

namespace {
void launch_bf16_gemm_v1(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;

  // if M = 128, M = 128
  // (128 / WN, 128 / WM) = (128 / 16, 128 / 16)
  // so we will launch a 8x8 grid where each block is comprised
  // a collection of 32-threads
  dim3 block_dim(WARP_SIZE);
  dim3 grid_dim(math::ceil_div(N, WN), math::ceil_div(M, WM));

  kernel::bf16_gemm_v1<WM, WN, WK><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_bf16_gemm_v2(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;

  dim3 block_dim(WARP_SIZE);
  dim3 grid_dim(math::ceil_div(N, WN), math::ceil_div(M, WM));

  kernel::bf16_gemm_v2<WM, WN, WK><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_bf16_gemm_v3(const bf16 *a, float alpha, const bf16 *b, float beta, float *c, size_t M, size_t N, size_t K) {
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int WARP_COLS = 4;
  constexpr int WARP_ROWS = 2;
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;

  // Need to calcualte the correct amount work in each block
  // which is determined by our WMMA shape and warp cols/rows
  // For example, M = N = K = 128
  // BM = 16 * 2 = 32
  // BN = 16 * 4 = 64
  // BK = 16
  // The size of our blocks in our grid will be: 32 x 64
  // The size of our grid will be (N / BN, M, BM) = (4, 2)
  // BK is left as WK for now. In the following kernels,
  // we'll see how BK can be used as a tuning mechanism
  constexpr int BM = WM * WARP_ROWS;
  constexpr int BN = WN * WARP_COLS;
  constexpr int BK = WK;

  dim3 block_dim(256);
  dim3 grid_dim(math::ceil_div(N, BN), math::ceil_div(M, BM));

  kernel::bf16_gemm_v3<BM, BN, BK, WM, WN, WK, NUM_THREADS><<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}
}// namespace

void bf16_gemm_v1(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta) {
  launch_bf16_gemm_v1(a->bf16_(), alpha, b->bf16_(), beta, c->fp32_(), c->shape[0], c->shape[1], a->shape[1]);
}

void bf16_gemm_v2(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta) {
  launch_bf16_gemm_v2(a->bf16_(), alpha, b->bf16_(), beta, c->fp32_(), c->shape[0], c->shape[1], a->shape[1]);
}

void bf16_gemm_v3(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta) {
  launch_bf16_gemm_v3(a->bf16_(), alpha, b->bf16_(), beta, c->fp32_(), c->shape[0], c->shape[1], a->shape[1]);
}

void bf16_gemm(Tensor *a, Tensor *b, Tensor *c, float alpha, float beta) {
  bf16_gemm_v3(a, b, c, alpha, beta);
}
}// namespace mlkl::operators::cuda
