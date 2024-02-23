
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

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
 * Shared Memory:
 * Assuming we have square matrices and our tile size is 16 which gives us tiles 
 * of size 16x16 = 256 elements within each tile. In order to take advantage of 
 * this technique, we need to launch a kernel that contains 
 * (M x N) // TILE_SIZE + 1 blocks with TILE_SIZE x TILE_SIZE threads.
 * 
 * dim3 gridDim((M x N) // TILE_SIZE + 1, (M x N) // TILE_SIZE + 1);
 * dim3 blockDim((TILE_SIZE * TILE_SIZE) // 2, (TILE_SIZE * TILE_SIZE) // 2);
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
 * tileM[threadIdx.y][threadIdx.x] = M[y * TILE_SIZE + threadIdx.x];
 * tileN[threadIdx.y][threadIdx.x] = N[threadIdx.y * K + x];
 * __syncthreads();
 * 
 * for (int k = 0; k < TILE_SIZE; ++k) {
 *    accumulator += tileM[threadIdx.y][k] * tileN[k][threadIdx.x];
 * }
 * 
 * // can probably be optimized by adding another tile for the accumulator
 * c[x * M + y] = alpha * accumulator + beta * c[x * M + y];
 * 
 * 
 * The above is a naive tiling implementation that doesn't really take advantage of the true
 * power of using shared memory
 */
namespace ml::operators::cuda {
    namespace kernel
    {   
        __global__ void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K);

        __global__ void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);

    } // namespace kernel
    
void launch_sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);

void launch_sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);

void sgemm_v1(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);

void sgemm_v2(const float *a, float alpha, const float *b, float beta, float *c, size_t M, size_t N, size_t K, int blk_size);
} // namespace ml::operators::cuda
