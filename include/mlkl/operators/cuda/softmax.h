#pragma once
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_types.h>
#include <vector>

#include <mlkl/core/basic_math.h>
#include <mlkl/core/tensor.h>

namespace mlkl::operators::cuda {
namespace kernel {
template<int NUM_THREADS>
__global__ void softmax_2d_v1(float *input, float *output, int batch_size, int dim_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row > batch_size) {
    return;
  }

  float norm_factor = 0.f;
  float maximum = 0.f;

  for (int i = 0; i < dim_size; ++i) {
    float value = input[row * dim_size + i];
    float new_max = fmax(value, maximum);
    float correction = expf(maximum - new_max);

    norm_factor = (norm_factor * correction) + expf(value - new_max);
    maximum = new_max;
  }

  for (int i = 0; i < dim_size; ++i) {
    output[row * dim_size + i] = expf(input[row * dim_size + i] - maximum) / norm_factor;
  }
}
}// namespace kernel

namespace {
void softmax_2d_v1(float *input, float *output, int dim, int *shape) {
  constexpr int NUM_THREADS = 16;
  int batch_size = shape[0];
  int dim_size = shape[dim];

  dim3 grid_dim(math::ceil_div(dim_size, 16));
  dim3 block_dim(NUM_THREADS);

  kernel::softmax_2d_v1<NUM_THREADS><<<grid_dim, block_dim>>>(input, output, batch_size, dim_size);
}
}// namespace

void softmax(Tensor &input, Tensor &output, int dim) {
  softmax_2d_v1(input.data, output.data, dim, output.shape);
}
}// namespace mlkl::operators::cuda