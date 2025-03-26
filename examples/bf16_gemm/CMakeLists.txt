enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(SOURCES bf16-gemm-example.cpp bf16_gemm_cpu.h bf16_gemm_cpu.cpp bf16_gemm_cuda.h bf16_gemm_cuda.cu)

# Create the executable
add_executable(bf16-gemm-example ${SOURCES})

# Ensure `.cu` files are compiled as CUDA
set_source_files_properties(bf16_gemm_cuda.cu PROPERTIES LANGUAGE CUDA)

# Enable proper CUDA compilation and linking
set_property(TARGET bf16-gemm-example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET bf16-gemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

# Apply compilation options
target_compile_options(bf16-gemm-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(bf16-gemm-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

# Link with CUDA and the project library
target_link_libraries(bf16-gemm-example PRIVATE ${PROJECT_NAME} CUDA::cublas CUDA::cudart)