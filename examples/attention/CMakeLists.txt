enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(SOURCES attention-example.cpp attention_cpu.h attention_cpu.cpp attention_cuda.h attention_cuda.cu)

# Create the executable
add_executable(attention-example ${SOURCES})

# Ensure `.cu` files are compiled as CUDA
set_source_files_properties(attention_cuda.cu PROPERTIES LANGUAGE CUDA)

# Enable proper CUDA compilation and linking
set_property(TARGET attention-example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET attention-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

# Apply compilation options
target_compile_options(attention-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(attention-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

# Link with CUDA and the project library
target_link_libraries(attention-example PRIVATE ${PROJECT_NAME} CUDA::cublas CUDA::cudart)