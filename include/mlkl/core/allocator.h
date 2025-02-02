#include <mlkl/core/tensor.h>
#include <mlkl/core/tensor_ops.h>

#include <vector>

namespace mlkl {
struct TensorAllocator {
  std::vector<Tensor> tensors;

  ~TensorAllocator() {
    for (auto tensor : tensors) {
      mlkl::destroy(tensor);
    }
  }

  Tensor empty(std::vector<int> &shape, Device device) {
    auto tensor = mlkl::empty(shape, device);
    tensors.push_back(tensor);
    return tensor;
  }

  Tensor randn(std::vector<int> &shape, Device device) {
    return mlkl::randn(shape, device);
  }
};
}// namespace mlkl