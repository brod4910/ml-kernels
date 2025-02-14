enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(HEADERS sgemm_cpu.h sgemm_cuda.h sgemm_cuda.cu)

add_executable(sgemm-example sgemm-example.cpp ${HEADERS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})

target_compile_options(sgemm-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(sgemm-example PRIVATE 
$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

set_property(TARGET sgemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME} CUDA::cublas)
