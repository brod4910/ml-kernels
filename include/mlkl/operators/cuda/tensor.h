#pragma once

#include <functional>
#include <iterator>
#include <numeric>

#include <mlkl/core/tensor.h>
#include <mlkl/utils/device.h>

#include <cuda_runtime_api.h>
#include <curand.h>

namespace mlkl::operators::cuda {
namespace kernel {
}// namespace kernel

Tensor empty(std::vector<int> &shape) {
  // Zero‐initialize the Tensor
  Tensor tensor;

  tensor.rank = shape.size();

  tensor.shape.reserve(tensor.rank);
  tensor.stride.reserve(tensor.rank);

  for (int i = 0; i < tensor.rank; ++i) {
    tensor.shape[i] = shape[i];
  }

  tensor.calculate_stride();

  cudaMalloc(&tensor.data, tensor.numel());
  CHECK_CUDA_ERROR();

  return tensor;
}

void fill(Tensor &tensor, int value) {
  cudaMemset(tensor.data, value, tensor.numel());
  CHECK_CUDA_ERROR();
}

void copy(Tensor &src, Device src_device, Tensor &dst, Device dst_device) {
  if (src_device == mlkl::Device::CUDA && dst_device == mlkl::Device::CPU) {
    cudaMemcpy(dst.data, src.data, src.numel(), cudaMemcpyDeviceToHost);
  } else if (src_device == mlkl::Device::CPU && dst_device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, src.numel(), cudaMemcpyHostToDevice);
  } else if (src_device == mlkl::Device::CUDA && dst_device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, src.numel(), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.data, src.data, src.numel(), cudaMemcpyHostToHost);
  }

  CHECK_CUDA_ERROR();
}

void destroy(Tensor &tensor) {
  cudaFree(tensor.data);
}

namespace {
void randn(float *data, size_t numel) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, data, numel);
  curandDestroyGenerator(prng);

  CHECK_CUDA_ERROR();
}
}// namespace

Tensor randn(std::vector<int> &shape) {
  auto tensor = empty(shape);

  randn(tensor.data, tensor.numel());

  CHECK_CUDA_ERROR();

  return tensor;
}

void randn(Tensor &tensor) {
  randn(tensor.data, tensor.numel());
}

// Tensor assert_correctness(Tensor &tensor, Tensor &ref, T epsilon = 1e-6);
}// namespace mlkl::operators::cuda