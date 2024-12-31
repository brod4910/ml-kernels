#pragma once
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_types.h>
#include <iostream>
#include <vector_types.h>

namespace ml::operators::cuda {
namespace kernel {
__global__ void softmax_v1(float *input, float *output, int dim) {
  int tid = threadIdx.x;

  //   expf();
}
}// namespace kernel

void launch_softmax_v1(float *input, float *output, int dim) {
}
}// namespace ml::operators::cuda