//
// Created by Brian Rodriguez on 10/20/23.
//

#pragma once
#include <cstddef>
/*
 *  8x8 GEMM
 *            M X 8                            N X 8
 *         Matrix A:                       Matrix B:
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 * [1. 1. 1. 1. 1. 1. 1. 1.]        [2. 2. 2. 2. 2. 2. 2. 2.]
 *
 *  Assumption: B is transposed.
 *
 *  In this case, the solution is trivial. We traverse the row by row, traversing N first, then M.
 *  Since addition is associative, we can work in chunks. However, this assumes a square matrix.
 *  For the non-square case, we have bounds checking but can do bounds checking when we compute the
 *  tiles, instead of during the loop.
 *
 *  Psuedo-code:
 *  If we tile 8x8. This might be inefficient since we are doing
 *  multiple shuffles and H-adds per tile instead of doing a single
 *  after we finish a row. Need to dig further into this.
 *
 *  for m in range(8)
 *    __m256 m_v = load &M[m]
 *    for n in range(8)
 *      __m256 n_v = load &N[n]
 *      __m256 c_v = m_v * n_v
 *      v = shuffle hi -> lo
 *      v = h-add c_v, v
 *      # This should get us answer in lo lane of v
 *      c[m * n] = extract v-lo
 *
 *
 */
namespace mlkl::operators::avx {
void sgemm(const float *__restrict__ a, float alpha, const float *__restrict__ b, float beta, float *__restrict__ c, size_t M, size_t N, size_t K);

}// namespace mlkl::operators::avx