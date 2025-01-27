#pragma once

namespace mlkl {
enum Device {
  CPU,
  CUDA
};

template<typename T>
struct Tensor {
  T *data;

  int rank;
  int *shape;
  int *stride;
};
}// namespace mlkl