set(HEADERS softmax_cpu.h)

if (${BUILD_CUDA})
    set(HEADERS ${HEADERS} softmax_cuda.h softmax_cuda.cu)
endif()
    
add_executable(softmax-example softmax-example.cpp ${HEADERS})
    
if (${BUILD_CUDA})
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})
    target_compile_options(softmax-example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    target_compile_options(softmax-example PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
    set_property(TARGET softmax-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME}_cuda)
endif()

target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME}_cpu)
