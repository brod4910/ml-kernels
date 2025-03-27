#pragma once
#include <mlkl/operators/cpu/attention.h>
#include <mlkl/operators/cpu/gemm.h>
#include <mlkl/operators/cpu/softmax.h>
#include <mlkl/operators/cpu/tensor_ops.h>
#include <mlkl/operators/cpu/transpose.h>

#ifdef __AVX__
#include <mlkl/operators/avx/gemm.h>
#include <mlkl/operators/avx/transpose.h>
#endif

#include <mlkl/operators/cuda/bf16_gemm.h>
#include <mlkl/operators/cuda/gemm.h>
#include <mlkl/operators/cuda/softmax.h>
#include <mlkl/operators/cuda/tensor_ops.h>
