#pragma once

#include <functional>
#include <iterator>
#include <numeric>

#include <mlkl/tensor/tensor.h>
#include <mlkl/utils/device.h>

#include <cuda_runtime_api.h>
#include <curand.h>

namespace mlkl::operators::cuda {
template<typename T>
size_t num_bytes(Tensor<T> &tensor) {
  return std::accumulate(std::begin(tensor.shape), std::end(tensor.shape), 1, std::multiplies<int>()) * sizeof(T);
}

template<typename T>
void calculate_stride(Tensor<T> &tensor) {
  int stride = 1;
  for (int i = tensor.rank - 1; i > 0; --i) {
    tensor.stride[i] = stride;
    stride *= tensor.shape[i];
  }
}

template<typename T>
Tensor<T> create_tensor(int *shapeArray) {
  // Zero‐initialize the Tensor
  Tensor<T> tensor{};

  int rank = 0;
  while (shapeArray[rank] != 0) {
    ++rank;
  }
  tensor.rank = rank;

  tensor.shape = new int[rank];
  tensor.stride = new int[rank];

  for (int i = 0; i < rank; ++i) {
    tensor.shape[i] = shapeArray[i];
  }

  calculate_stride(tensor);

  cudaMalloc(&tensor.data, num_bytes<T>(tensor));
  CHECK_CUDA_ERROR();

  return tensor;
}

template<typename T>
void fill_tensor(Tensor<T> &tensor, T value) {
  cudaMemset(tensor.data, value, num_bytes<T>(tensor));
  CHECK_CUDA_ERROR();
}

template<typename T>
void copy(Tensor<T> &src, Device src_device, Tensor<T> &dst, Device dst_device) {
  if (src_device == CUDA && dst_device == CPU) {
    cudaMemcpy(dst.data, src.data, num_bytes<T>(src), cudaMemcpyDeviceToHost);
  } else if (src_device == CPU && dst_device == CUDA) {
    cudaMemcpy(dst.data, src.data, num_bytes<T>(src), cudaMemcpyHostToDevice);
  } else if (src_device == CUDA && dst_device == CUDA) {
    cudaMemcpy(dst.data, src.data, num_bytes<T>(src), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.data, src.data, num_bytes<T>(src), cudaMemcpyHostToHost);
  }

  CHECK_CUDA_ERROR();
}

template<typename T>
Tensor<T> randn(int *shape) {
  auto tensor = create_tensor<T>(shape);
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, tensor.data, num_bytes<T>(tensor));
  curandDestroyGenerator(prng);

  CHECK_CUDA_ERROR();
}

template<typename T>
Tensor<T> assert_correctness(Tensor<T> &tensor, Tensor<T> &ref, T epsilon = 1e-6) {
}
}// namespace mlkl::operators::cuda