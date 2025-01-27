#pragma once
#include <mlkl/operators/cpu/create.h>
#include <mlkl/operators/cpu/gemm.h>
#include <mlkl/operators/cpu/softmax.h>
#include <mlkl/operators/cpu/transpose.h>

#ifdef __AVX__
#include <mlkl/operators/avx/gemm.h>
#include <mlkl/operators/avx/transpose.h>
#endif

#ifdef __CUDA__
#include <mlkl/operators/cuda/create.h>
#include <mlkl/operators/cuda/gemm.h>
#include <mlkl/operators/cuda/softmax.h>
#endif