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

mlkl::Tensor create_tensor(std::vector<int> &shape) {
  // Zero‐initialize the Tensor
  mlkl::Tensor tensor;

  tensor.rank = shape.size();

  tensor.shape = new int[tensor.rank];
  tensor.stride = new int[tensor.rank];

  for (int i = 0; i < tensor.rank; ++i) {
    tensor.shape[i] = shape[i];
  }

  calculate_stride(tensor);

  cudaMalloc(&tensor.data, num_bytes(tensor));
  CHECK_CUDA_ERROR();

  return tensor;
}

void fill_tensor(mlkl::Tensor &tensor, int value) {
  cudaMemset(tensor.data, value, num_bytes(tensor));
  CHECK_CUDA_ERROR();
}

void copy(mlkl::Tensor &src, Device src_device, mlkl::Tensor &dst, Device dst_device) {
  if (src_device == mlkl::Device::CUDA && dst_device == mlkl::Device::CPU) {
    cudaMemcpy(dst.data, src.data, num_bytes(src), cudaMemcpyDeviceToHost);
  } else if (src_device == mlkl::Device::CPU && dst_device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, num_bytes(src), cudaMemcpyHostToDevice);
  } else if (src_device == mlkl::Device::CUDA && dst_device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, num_bytes(src), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.data, src.data, num_bytes(src), cudaMemcpyHostToHost);
  }

  CHECK_CUDA_ERROR();
}

Tensor randn(std::vector<int> &shape) {
  auto tensor = create_tensor(shape);
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, tensor.data, mlkl::num_bytes(tensor));
  curandDestroyGenerator(prng);

  CHECK_CUDA_ERROR();
}

void destroy(mlkl::Tensor &tensor) {
  cudaFree(tensor.data);
  delete tensor.shape;
  delete tensor.stride;
}

//
// Tensor assert_correctness(mlkl::Tensor &tensor, mlkl::Tensor &ref, T epsilon = 1e-6);
}// namespace mlkl::operators::cuda