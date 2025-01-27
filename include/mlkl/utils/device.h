#pragma once

#define CHECK_CUDA_ERROR() mlkl::utils::check_cuda_error(__FILE__, __LINE__)

namespace mlkl::utils {
#ifndef __CUDA__
void check_cuda_error(const char *file, int line) {}
#else
void check_cuda_error(const char *file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(error) << " \""
              << cudaGetErrorString(error) << "\"" << std::endl;
    exit(-1);
  }
}
#endif
}// namespace mlkl::utils