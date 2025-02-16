enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(SOURCES sgemm-example.cpp sgemm_cpu.h sgemm_cpu.cpp sgemm_cuda.h sgemm_cuda.cu)

# Create the executable
add_executable(sgemm-example ${SOURCES})

# Ensure `.cu` files are compiled as CUDA
set_source_files_properties(sgemm_cuda.cu PROPERTIES LANGUAGE CUDA)

# Enable proper CUDA compilation and linking
set_property(TARGET sgemm-example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET sgemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

# Apply compilation options
target_compile_options(sgemm-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(sgemm-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

# Link with CUDA and the project library
target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME} CUDA::cublas CUDA::cudart)