#include <mlkl/core/tensor.h>
#include <mlkl/core/tensor_ops.h>

#include <vector>

namespace mlkl {
struct TensorAllocator {
  std::vector<Tensor> tensors_;
  Device device_;

  TensorAllocator(Device device) : device_(device) {}

  ~TensorAllocator() {
    for (auto tensor : tensors_) {
      destroy(tensor, device_);
    }
  }

  Tensor empty(std::vector<int> &shape) {
    auto tensor = create_tensor(shape, device_);
    tensors_.push_back(tensor);
    return tensor;
  }

  Tensor randn(std::vector<int> &shape) {
    return mlkl::randn(shape, device_);
  }

  Tensor copy(Tensor &tensor, Device device) {
  }
};
}// namespace mlkl