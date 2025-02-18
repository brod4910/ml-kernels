#include "mlkl/core/tensor.h"
#include <mlkl/operators/cuda/tensor_ops.h>
#include <mlkl/utils/device.h>

#include <cuda_runtime_api.h>
#include <curand.h>

namespace mlkl::operators::cuda {
Tensor empty(std::vector<int> &shape) {
  // Zero‐initialize the Tensor
  Tensor tensor;
  tensor.device = Device::CUDA;
  tensor.rank = shape.size();

  tensor.shape.reserve(tensor.rank);
  tensor.stride.reserve(tensor.rank);

  for (int i = 0; i < tensor.rank; ++i) {
    tensor.shape.push_back(shape[i]);
  }

  if (tensor.rank > 0) {
    tensor.stride[tensor.rank - 1] = 1;
    for (int i = tensor.rank - 2; i >= 0; --i) {
      tensor.stride.push_back(tensor.stride[i + 1] * tensor.shape[i + 1]);
    }
  }

  cudaMalloc(&tensor.data, tensor.num_bytes());
  CHECK_CUDA_ERROR();

  return tensor;
}

void fill(Tensor &tensor, int value) {
  cudaMemset(tensor.data, value, tensor.num_bytes());
  CHECK_CUDA_ERROR();
}

void copy(Tensor &src, Tensor &dst) {
  if (src.device == mlkl::Device::CUDA && dst.device == mlkl::Device::CPU) {
    cudaMemcpy(dst.data, src.data, dst.num_bytes(), cudaMemcpyDeviceToHost);
  } else if (src.device == mlkl::Device::CPU && dst.device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, dst.num_bytes(), cudaMemcpyHostToDevice);
  } else if (src.device == mlkl::Device::CUDA && dst.device == mlkl::Device::CUDA) {
    cudaMemcpy(dst.data, src.data, dst.num_bytes(), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.data, src.data, dst.num_bytes(), cudaMemcpyHostToHost);
  }

  CHECK_CUDA_ERROR();
}

void destroy(Tensor &tensor) {
  cudaFree(tensor.data);
  CHECK_CUDA_ERROR();
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

  return tensor;
}

void randn(Tensor &tensor) {
  randn(tensor.data, tensor.numel());
}

// Tensor assert_correctness(Tensor &tensor, Tensor &ref, T epsilon = 1e-6);
}// namespace mlkl::operators::cuda