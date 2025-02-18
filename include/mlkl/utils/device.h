#pragma once
#include <cuda_runtime_api.h>
#include <iostream>

#define CHECK_CUDA_ERROR() mlkl::utils::check_cuda_error(__FILE__, __LINE__)

namespace mlkl::utils {
void check_cuda_error(const char *file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(error) << " \""
              << cudaGetErrorString(error) << "\"" << std::endl;
    exit(-1);
  }
}
}// namespace mlkl::utils