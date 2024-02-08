
#pragma once
#include <cstddef>

/*
 * The naive implementation of CUDA GEMM would be to launch an equal number of thread as there are elements 
 * in the matrix.
 * 
 * That would look something like this:
 * 
 * int x = blockDim.x * blockIdx.x + threadIdx.x;
 * int y = blockDim.y * blockIdx.y + threadIdx.y;
 * 
 * float accumulator = 0.0;
 * 
 * for (int k = 0; k < K; ++k) {
 *     accumulator += m[x * K + k] * n[y * K + k];
 * }
 * 
 *  float c_v = c[n * M + m];
 *  c[x * M + y] = alpha * accumulator + beta * c_v;
 * 
 * However, this is slow and inefficient since we aren't taking advantage of many CUDA-specific
 * kernel optimizations:
 * 
 *  1. Memory coalescing
 *  2. Shared memory
 *  3. Tiling
 *  4. Prefetching
 *  
 * ... etc.
 * 
 * Tiling:
 * Assuming we have square matrices and our tile size is 16 which gives us tiles 
 * of size 16x16 = 256 elements within each tile. In order to take advantage of 
 * this technique, we need to launch a kernel that contains 
 * (M x N) // TILE_SIZE + 1 blocks with TILE_SIZE x TILE_SIZE threads.
 * 
 * dim3 gridDim((M x N) // TILE_SIZE + 1, 1);
 * dim3 blockDim(TILE_SIZE * TILE_SIZE, 1);
 * 
 * sgemm<<gridDim, blockDim, TILE_SIZE * TILE_SIZE>>(...);
 * 
 * TILE_SIZE = 16
 *  
 * __shared__ tileM[TILE_SIZE][TILE_SIZE], tileN[TILE_SIZE][TILE_SIZE]; 
 * 
 * int x = blockDim.x * blockIdx.x + threadIdx.x;
 * int y = blockDim.y * blockIdx.y + threadIdx.y;
 * 
 * float accumulator = 0.0;
 * 
 * tileM[y][x] = M[y * TILE_SIZE + ]
 * 
 *   
 */
namespace ml::operators::cuda {
void sgemm(const float * a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);
} // namespace ml::operators::cuda
