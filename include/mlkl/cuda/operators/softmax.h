#pragma once
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_types.h>
#include <iostream>
#include <vector_types.h>

namespace ml::operators::cuda {
namespace kernel {
template<int NUM_THREADS>
__global__ void softmax_2d_v1(float *input, float *output, int size) {
  int tid = threadIdx.x;

  float norm_factor = 0.f;
  float maximum = 0.f;

  for (int i = 0; i < size; ++i) {
    float value = input[tid * size + i];
    float new_max = fmax(value, maximum);
    float correction = expf(maximum - new_max);

    norm_factor = (norm_factor * correction) + expf(value - new_max);
    maximum = new_max;
  }

  for (int i = 0; i < size; ++i) {
    output[tid * size + i] = expf(input[tid * size + i] - maximum) / norm_factor;
  }
}
}// namespace kernel

void launch_softmax_2d_v1(float *input, float *output, int dim, int *shape) {
  constexpr NUM_THREADS = 16;
  int size = shape[dim];

  dim3 grid_dim(ceil_div(size, 16));
  dim3 block_dim(NUM_THREADS);

  kernel::softmax_2d_v1<NUM_THREADS><<<grid_dim, block_dim>>>(input, output, int dim, int size);
}
}// namespace ml::operators::cuda