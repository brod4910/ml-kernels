//
// Created by Brian Rodriguez on 8/26/23.
//
//
// Created by Brian Rodriguez on 6/17/23.
//
#include "attention_cpu.h"

#include <iostream>
#include <tuple>
#include <vector>

int main() {
  // clang-format off
  std::vector<std::tuple<int, int, int, int, int>> matrix_sizes = {
    {1, 4, 128, 128, 64},
    {2, 4, 128, 128, 64},
    {2, 4, 256, 256, 64},
    {2, 4, 256, 256, 128},
  };
  // clang-format on

  for (const auto &[batch_size, num_heads, seq_len_q, seq_len_kv, head_dim] : matrix_sizes) {
    std::cout << "CPU" << std::endl;
    attention_cpu(batch_size, num_heads, seq_len_q, seq_len_kv, head_dim, 5);
  }
}
