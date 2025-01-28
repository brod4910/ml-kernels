#pragma once

namespace mlkl::math {
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}
}// namespace mlkl::math