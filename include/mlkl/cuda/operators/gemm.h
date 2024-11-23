
#pragma once
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// TODO: Delete this and make functions templates
#define TILE_X 16
#define TILE_Y 16
#define WARP_SIZE 32

namespace ml::operators::cuda {
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

namespace kernel {
// naive
__global__ void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > M || y > N) {
    return;
  }

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
    accumulator += a[x * K + k] * b[k * N + y];
  }

  c[x * N + y] = accumulator;
}

/*
  Global memory coalescing of A, B, and C by virtue of swapping which dimensions we traverse first. In the naive implementation above,
  we find ourselves traversing along the slow changing axis of A first which is M. Thus writing a value to C column-wise. In this implementation, 
  if we swap the x, y values, we are now traversing the same row of A (coalesced/multi-casted reads) but changing the columns of B
  thus traversing along the fastest changing axis of B which is N.

  One thing to keep in mind that tripped me up was that in our CUDA kernel, if we launch a grid with (2,2) blocks of size (4, 4), the way we traverse
  the block is (0,0), (1,0), (2,0)... thus, the fastest changing axis of our kernel is the first. This broke my brain when it came to GEMMs since when
  we write to the C matrix, we are using the x value as our y and our y value as our x. In other words, as we compute values of C, we're traversing like so,
  (x, y), (x + 1, y), (x + 2, y)...
*/
__global__ void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > N || y > M) {
    return;
  }

  float accumulator = 0.0;

  for (int k = 0; k < K; ++k) {
    accumulator += a[y * K + k] * b[k * N + x];
  }

  c[y * N + x] = accumulator;
}

/*
Proper block tiling of A & B loading tiles from global to shared memory.
*/
__global__ void
sgemm_v3(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ float ATile[TILE_Y][TILE_X];
  __shared__ float BTile[TILE_Y][TILE_X];

  // Block indices dictate the C-block we are going to process
  // We still need to process an entire row of A and an entire column of B
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;

  // Dictates which value of C we're computing within the C-block
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  float accumulator = 0.0;

  // M / TILE_X = number of blocks we need to traverse till end of matrices
  // assuming square matrix
  for (int step = 0; step < M / TILE_X; ++step) {
    // calculate the start of both A and B tiles for shared memory.
    // This is quite an annoying calculation to get correct...
    // Essentially, we use the block indices of our kernel to get the
    // corners of each tile we want to compute. Step moves us in the
    // direction we want to move each tile as we traverse memory.
    // For the A tile that is to the right →
    // For the B tile that is downward ↓
    // We need to move by the number of elements in our block, in this case 16
    // Thus, for each iteration of the loop, we're moving (16 * step) elements
    // of our tiles to the right for A and down for B.
    const float *tile_a = &a[(block_y * TILE_X) * K + step * TILE_X];
    const float *tile_b = &b[(step * TILE_X) * N + block_x * TILE_X];

    // Loads the inner-tile elements using the thread indices
    // Don't forget to multiply by the widths of matrices...
    // Ooopsies, I may have spent several hours on this... :)
    ATile[tid_y][tid_x] = tile_a[tid_y * K + tid_x];
    BTile[tid_y][tid_x] = tile_b[tid_y * N + tid_x];

    __syncthreads();

    for (int k = 0; k < TILE_X; ++k) {
      accumulator += ATile[tid_y][k] * BTile[k][tid_x];
    }

    __syncthreads();
  }

  int linear = (blockIdx.y * blockDim.y + tid_y) * N + (blockIdx.x * blockDim.x + tid_x);
  c[linear] = accumulator;
}

/*
  There is a lot packed into this function with another another level of tiling.
  This function implements thread-tiling by calculating multiple output values per thread.
*/
// TODO: Delete this and make functions templates
#define BLOCK_TILE_X 64
#define BLOCK_TILE_Y 64
#define WARP_SIZE 32
#define NUM_TH_ITEMS_M 4
#define NUM_TH_ITEMS_N 4

__global__ void sgemm_v4(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ float ATile[BLOCK_TILE_Y][BLOCK_TILE_X];
  __shared__ float BTile[BLOCK_TILE_Y][BLOCK_TILE_X];

  // Block indices dictate the C-block we are going to process
  // We still need to process an entire row of A and an entire column of B
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;

  // Dictates which starting value of C we're computing within the C-block
  // since we're implementing thread-tiling, each thread computes (m, n)
  // outputs of C.
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // Local A and B memory fragments we're using to compute our C output thread-tile
  float a_frag[NUM_TH_ITEMS_M];
  float b_frag[NUM_TH_ITEMS_N];
  float accumulator[NUM_TH_ITEMS_M][NUM_TH_ITEMS_N] = {0.0};

  // Block of code is similar to our V3. We first compute the starting points
  // of our A and B tiles for easier accessing when loading from GMEM.
  for (int step = 0; step < K / BLOCK_TILE_X; ++step) {
    // still in GMEM here since we're only computing the starting addresses
    const float *tile_a = &a[(block_y * BLOCK_TILE_X) * K + (step * BLOCK_TILE_X)];
    const float *tile_b = &b[(step * BLOCK_TILE_X) * N + (block_x * BLOCK_TILE_X)];

    // Since each thread calculates more than one output value,
    // we need to also account for this when we load to SMEM.
    // We don't have enough threads launched in our blocks to load the values
    // we need from GMEM so this loop accounts for that using the number of
    // items computed by each thread (M, N)
    for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
      for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
        ATile[tid_y + (j * blockDim.x)][tid_x + (i * blockDim.y)] = tile_a[(tid_y + (j * blockDim.x)) * K + ((tid_x + (i * blockDim.y)))];
        BTile[tid_y + (j * blockDim.x)][tid_x + (i * blockDim.y)] = tile_b[(tid_y + (j * blockDim.x)) * N + ((tid_x + (i * blockDim.y)))];
      }
    }

    __syncthreads();

    // Now we need to process the items in SMEM by moving them to
    // our fragment arrays which are thread-local. We move m items
    // to a_frag and n items to b_frag. This will give us a vector of
    // (m, 1) and (1, n) since we're accessing A column-wise and B row-wise.
    for (int k = 0; k < BLOCK_TILE_X; ++k) {
      for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
        a_frag[i] = ATile[tid_y * NUM_TH_ITEMS_M + i][k];
      }
      for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
        b_frag[j] = BTile[k][tid_x * NUM_TH_ITEMS_N + j];
      }

      // Items are now at the thread-level so we can finally compute
      // our dot-products of our values in registers.
      for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
        for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
          accumulator[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

  // Write our values that are in our registers to GMEM.
  // Important to write values coalesced by swapping the N and M loops
  // to traverse column-wise first instead of row-wise
  for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
    for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
      int linear = ((blockIdx.y * BLOCK_TILE_Y) + (tid_y * NUM_TH_ITEMS_M + i)) * N + ((blockIdx.x * BLOCK_TILE_X) + (tid_x * NUM_TH_ITEMS_N + j));
      c[linear] = accumulator[i][j];
    }
  }
}

/*
  Building on our last function, we're now going to implement warp-tiling which is yet another level of tiling...
  There aren't any "plain" CUDA-specific constructs to interact with warps and to achieve warp-tiling.
  So we must modify the existing kernel in a way that makes sense for tiling along our hidden "warp" dimension.
*/

#define WARP_M 32
#define WARP_N 16
#define WARP_INNER_M 2
#define WARP_INNER_N 2
#define THREAD_M 2
#define THREAD_N 2

__global__ void sgemm_v5(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  __shared__ float ATile[BLOCK_TILE_Y][BLOCK_TILE_X];
  __shared__ float BTile[BLOCK_TILE_Y][BLOCK_TILE_X];

  // Block indices dictate the C-block we are going to process
  // We still need to process an entire row of A and an entire column of B
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;

  // Dictates which value of C we're computing within the C-block
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  int warp_id = (tid_x * tid_y) / WARP_SIZE;
  int lane_id = (tid_x * tid_y) % WARP_SIZE;
  int warp_x = tid_x / WARP_SIZE;
  int warp_y = tid_y / WARP_SIZE;

  float accumulator[NUM_TH_ITEMS_M][NUM_TH_ITEMS_N] = {0.0};
  float a_frag[NUM_TH_ITEMS_M];
  float b_frag[NUM_TH_ITEMS_N];

  for (int step = 0; step < K / BLOCK_TILE_X; ++step) {
    const float *tile_a = &a[(block_y * BLOCK_TILE_X) * K + (step * BLOCK_TILE_X)];
    const float *tile_b = &b[(step * BLOCK_TILE_X) * N + (block_x * BLOCK_TILE_X)];

    for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
      for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
        ATile[tid_y + (j * blockDim.x)][tid_x + (i * blockDim.y)] = tile_a[(tid_y + (j * blockDim.x)) * K + ((tid_x + (i * blockDim.y)))];
        BTile[tid_y + (j * blockDim.x)][tid_x + (i * blockDim.y)] = tile_b[(tid_y + (j * blockDim.x)) * N + ((tid_x + (i * blockDim.y)))];
      }
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_TILE_X; ++k) {
      for (int warp_m = 0; warp_m < WARP_INNER_M; ++warp_m) {
        for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
          a_frag[i] = ATile[tid_y * NUM_TH_ITEMS_M + i][k];
        }
      }

      for (int warp_n = 0; warp_n < WARP_INNER_N; ++warp_n) {
        for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
          b_frag[j] = BTile[k][tid_x * NUM_TH_ITEMS_N + j];
        }
      }

      for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
        for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
          accumulator[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

  for (int j = 0; j < NUM_TH_ITEMS_N; ++j) {
    for (int i = 0; i < NUM_TH_ITEMS_M; ++i) {
      int linear = ((blockIdx.y * BLOCK_TILE_Y) + (tid_y * NUM_TH_ITEMS_M + i)) * N + ((blockIdx.x * BLOCK_TILE_X) + (tid_x * NUM_TH_ITEMS_N + j));
      c[linear] = accumulator[i][j];
    }
  }
}
}// namespace kernel

void launch_sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, TILE_X), ceil_div(N, TILE_Y));
  dim3 block_dim(TILE_X, TILE_Y);

  kernel::sgemm_v1<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, TILE_X), ceil_div(N, TILE_Y));
  dim3 block_dim(TILE_X, TILE_Y);
  kernel::sgemm_v2<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_sgemm_v3(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, TILE_X), ceil_div(N, TILE_Y));
  dim3 block_dim(TILE_X, TILE_Y);
  kernel::sgemm_v3<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}

void launch_sgemm_v4(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K) {
  dim3 grid_dim(ceil_div(M, BLOCK_TILE_X), ceil_div(N, BLOCK_TILE_Y));
  dim3 block_dim(BLOCK_TILE_X / NUM_TH_ITEMS_M, BLOCK_TILE_Y / NUM_TH_ITEMS_N);
  kernel::sgemm_v4<<<grid_dim, block_dim>>>(a, alpha, b, beta, c, M, N, K);
}
}// namespace ml::operators::cuda
