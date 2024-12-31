set(HEADERS softmax_cpu.h)

# if (${BUILD_AVX})
#     set(HEADERS ${HEADERS} softmax_avx.h)
# endif ()

# if (${BUILD_CUDA})
#     set(HEADERS ${HEADERS} softmax_cuda.h softmax_cuda.cu)
# endif()
    
add_executable(softmax-example softmax-example.cpp ${HEADERS})
    
target_include_directories(softmax-example PUBLIC ${PROJECT_SOURCE_DIR}/include)

# if (${BUILD_CUDA})
#     enable_language(CUDA)
#     find_package(CUDAToolkit)
#     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})
#     target_compile_options(softmax-example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA__>)
#     target_compile_options(softmax-example PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-D__CUDA__>)
#     set_property(TARGET softmax-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
#     target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME}_cuda CUDA::cudart CUDA::cuda_driver CUDA::cublas)
#     # Specify the CUDA compute capability
# endif()

target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME}_cpu)
