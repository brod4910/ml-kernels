set(HEADERS sgemm_cpu.h)

if (${BUILD_AVX})
    set(HEADERS ${HEADERS} sgemm_avx.h)
endif ()

if (${BUILD_CUDA})
    set(HEADERS ${HEADERS} sgemm_cuda.h sgemm_cuda.cu)
endif()
    
add_executable(sgemm-example sgemm-example.cpp ${HEADERS})

if (${BUILD_CUDA})
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})
    target_compile_options(sgemm-example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    target_compile_options(sgemm-example PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
    set_property(TARGET sgemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cuda CUDA::cublas)
elseif(${BUILD_AVX})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_avx)
else()
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cpu)
endif()

