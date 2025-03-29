#include <cassert>
#include <iostream>

#include "mlkl/core/tensor.h"
#include "mlkl/core/tensor_ops.h"
#include <mlkl/mlkl.h>
#include <mlkl/operators/cuda/gemm.h>
#include <mlkl/utils/device.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#define CHECK_CUDNN(call)                                                \
  {                                                                      \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ \
                << std::endl;                                            \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  }

auto cudnn_kernel = [](mlkl::Tensor *qkv, mlkl::Tensor *output) {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnAttnDescriptor_t attnDesc;
  cudnnCreateAttnDescriptor(&attnDesc);

  int numHeads = 8;
  int batchSize = 32;
  int seqLength = 128;
  int embedDim = 512;
  int smScaler = 1.0 / sqrt(embedDim);

  cudnnAttnDescriptor_t attnDesc;
  cudnnCreateAttnDescriptor(&attnDesc);

  // Set up the attention descriptor
  int numHeads = 8;
  int embedDimPerHead = 64;
  float attnScale = 1.0 / sqrt(embedDimPerHead);// Scaling factor
  int batchSize = 32;
  int seqLength = 128;
  int tensorSize = batchSize * numHeads * seqLength * embedDimPerHead;

  float16_t *d_qkv;
  cudaMalloc(&d_qkv, sizeof(float16_t) * tensorSize * 3);// Q, K, V

  float16_t *d_output;
  cudaMalloc(&d_output, sizeof(float16_t) * tensorSize);

  float32_t *d_stats;
  cudaMalloc(&d_stats, sizeof(float32_t) * batchSize * numHeads * seqLength);

  cudnnSetAttnDescriptor(
    attnDesc,
    numHeads,
    embedDimPerHead,
    attnScale,
    CUDNN_DATA_HALF,                  // FP16 or BF16
    CUDNN_ATTN_QKV_LAYOUT_INTERLEAVED,// Interleaved QKV layout
    CUDNN_ATTN_SCALED_DOT_PRODUCT,
    batchSize,
    seqLength);

  size_t workspaceSize;
  void *workspace;

  cudnnGetAttnForwardWorkspaceSize(cudnn, attnDesc, &workspaceSize);
  cudaMalloc(&workspace, workspaceSize);

  cudnnAttnForward(
    cudnn,
    attnDesc,
    nullptr,  // Reserved space
    workspace,// Workspace
    d_qkv,    // Input QKV tensor
    d_output, // Output tensor
    d_stats,  // Softmax statistics (for training)
    nullptr); // Optional dropout descriptor

  cudaFree(d_qkv);
  cudaFree(d_output);
  cudaFree(workspace);
  cudaFree(d_stats);
  cudnnDestroyAttnDescriptor(attnDesc);
  cudnnDestroy(cudnn);
};

template<typename Kernel>
void test_kernel(const char *kernel_name,
                 Kernel kernel,
                 int M, int N, int K, float alpha, float beta, int num_runs = 10) {
}

void attention_cuda(int M, int N, int K, float alpha, float beta, int num_runs) {
  // Test CUDNN

  // Test custom kernels
}