#include <cmath>

#include <mlkl/core/tensor.h>
#include <mlkl/operators/cpu/tensor_ops.h>
#include <mlkl/operators/cuda/tensor_ops.h>

#include <mlkl/core/tensor_ops.h>

namespace mlkl {
Tensor empty(std::vector<int> &shape, Device device) {
  if (device == mlkl::Device::CPU) {
    return operators::cpu::empty(shape);
  } else {
    return operators::cuda::empty(shape);
  }
}

void fill(Tensor &tensor, int value) {
  if (tensor.device == mlkl::Device::CPU) {
    return operators::cpu::fill(tensor, value);
  } else {
    return operators::cuda::fill(tensor, value);
  }
}

void destroy(Tensor &tensor) {
  if (tensor.device == mlkl::Device::CPU) {
    return operators::cpu::destroy(tensor);
  } else {
    return operators::cuda::destroy(tensor);
  }
}

Tensor randn(std::vector<int> &shape, Device device) {
  if (device == Device::CPU) {
    return operators::cpu::randn(shape);
  } else {
    return operators::cuda::randn(shape);
  }
}

void randn(Tensor &tensor) {
  if (tensor.device == Device::CPU) {
    return operators::cpu::randn(tensor);
  } else {
    return operators::cuda::randn(tensor);
  }
}

void to(Tensor &tensor, Device device) {
  if (tensor.device == device) {
    return;
  }

  if (device == Device::CPU && tensor.device == Device::CUDA) {
    auto temp = empty(tensor.shape, device);
    operators::cuda::copy(tensor, temp);
    destroy(tensor);
    tensor = temp;
  }
}

namespace {
bool same_shape(Tensor &a, Tensor &b) {
  if (a.rank != b.rank) {
    return false;
  }

  for (int i = 0; i < a.rank; ++i) {
    if (a.shape[i] != b.shape[i]) {
      return false;
    }
  }

  return true;
}
}// namespace

bool equals(Tensor &a, Tensor &b, float epsilon) {
  if (a.device != Device::CPU || b.device != Device::CPU) {
    return false;
  }

  if (a.numel() != b.numel() || !same_shape(a, b)) {
    return false;
  }

  float diff = .0f;

  for (int i = 0; i < a.numel(); ++i) {
    diff = fabs((double) a.data[i] - (double) b.data[i]);
    if (diff > epsilon) {
      return false;
    }
  }

  return true;
}

void copy(Tensor &src, Tensor &dst) {
  if (src.device == Device::CUDA || dst.device == Device::CUDA) {
    operators::cuda::copy(src, dst);
  } else {
    operators::cpu::copy(src, dst);
  }
}
}// namespace mlkl