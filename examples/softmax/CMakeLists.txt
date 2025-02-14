enable_language(CUDA)
find_package(CUDAToolkit)

set(HEADERS softmax_cpu.h softmax_cuda.h softmax_cuda.cu)
    
add_executable(softmax-example softmax-example.cpp ${HEADERS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})

target_compile_options(softmax-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(softmax-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)
set_property(TARGET softmax-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME})
