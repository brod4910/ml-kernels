set(HEADERS sgemm_cpu.h)

if (${BUILD_AVX})
    set(HEADERS ${HEADERS} sgemm_avx.h)
endif ()

if (${BUILD_CUDA})
    set(HEADERS ${HEADERS} sgemm_cuda.h sgemm_cuda.cu)
endif()
    
add_executable(sgemm-example sgemm-example.cpp ${HEADERS})
    
target_include_directories(sgemm-example PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (${BUILD_AVX})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_avx)
endif()

if (${BUILD_CUDA})
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})
    target_compile_options(sgemm-example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA__>)
    target_compile_options(sgemm-example PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-D__CUDA__>)
    set_property(TARGET sgemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cuda CUDA::cudart CUDA::cuda_driver)
    # Specify the CUDA compute capability
endif()

target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cpu)
