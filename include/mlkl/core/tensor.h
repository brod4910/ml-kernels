#pragma once
#include <cstddef>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

using fp32 = float;
using bf16 = __nv_bfloat16;
using fp16 = __half;

namespace mlkl {
enum class Device {
  CPU,
  CUDA
};

enum class DType {
  F8,
  BF16,
  F16,
  F32
};

struct Tensor {
  void *data;

  int rank;
  std::vector<int> shape;
  std::vector<int> stride;

  Device device = Device::CPU;
  DType dtype = DType::F32;

  template<typename T>
  T *get_data() {
    return reinterpret_cast<T *>(data);
  }

  bf16 *bf16_();

  fp16 *fp16_();

  fp32 *fp32_();

  size_t dtype_size();

  size_t num_bytes();

  size_t numel();

  void to(Device device);
};

}// namespace mlkl