enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# List sources correctly (move .cu file out of HEADERS)
set(SOURCES softmax-example.cpp softmax_cpu.h softmax_cpu.cpp softmax_cuda.h softmax_cuda.cu)

# Create executable
add_executable(softmax-example ${SOURCES})

# Ensure `.cu` files are compiled as CUDA
set_source_files_properties(softmax_cuda.cu PROPERTIES LANGUAGE CUDA)

# Enable CUDA features for proper linking
set_property(TARGET softmax-example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET softmax-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

# Apply compilation options
target_compile_options(softmax-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
)
target_compile_options(softmax-example PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

# Link CUDA libraries
target_link_libraries(softmax-example PRIVATE ${PROJECT_NAME} CUDA::cudart)