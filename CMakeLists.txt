cmake_minimum_required(VERSION 3.21)

project(
    ml_kernels
    VERSION 0.0.1
    DESCRIPTION "A collection of machine learning kernels for CPU, AVX, and CUDA"
    HOMEPAGE_URL "https://github.com/brod4910/ml-kernels"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CUDA_ARCH "86;89" CACHE STRING "CUDA architecture")
option(BUILD_AVX "Build AVX library version" OFF)

# ---- Find Packages ----
find_package(fmt REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(PACKAGES fmt::fmt)

# ---- Collect Source Files ----
file(GLOB_RECURSE CORE_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/*.cpp)
file(GLOB_RECURSE CORE_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/core/*.h)
file(GLOB_RECURSE CPU_SRC_FILES ${PROJECT_SOURCE_DIR}/src/operators/cpu/*.cpp)
file(GLOB_RECURSE CPU_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/operators/cpu/*.h)
file(GLOB_RECURSE CU_SRC_FILES ${PROJECT_SOURCE_DIR}/src/operators/cuda/*.cu)
file(GLOB_RECURSE CUDA_SRC_FILES ${PROJECT_SOURCE_DIR}/src/operators/cuda/*.cpp)
file(GLOB_RECURSE CUDA_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/operators/cuda/*.h)

# ---- Define Compiler Options ----
set(CXX_DEBUG_OPTS $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra -Wpedantic>)
set(CXX_RELWITHDEBINFO_OPTS $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>)
set(CXX_RELEASE_OPTS $<$<CONFIG:Release>:-O3 -DNDEBUG -Wall -std=c++17 -ffast-math>)
set(CUDA_DEBUG_OPTS $<$<CONFIG:Debug>:-G -O0>)
set(CUDA_RELWITHDEBINFO_OPTS $<$<CONFIG:RelWithDebInfo>:-G -O2>)
set(CUDA_RELEASE_OPTS $<$<CONFIG:Release>:-O3 -lineinfo>)

# ---- Create a Combined CPU + CUDA Library ----
add_library(ml_kernels SHARED 
    ${CORE_SRC_FILES} ${CORE_HDR_FILES}
    ${CPU_SRC_FILES} ${CPU_HDR_FILES}
    ${CUDA_SRC_FILES} ${CUDA_HDR_FILES} ${CU_SRC_FILES}
)

# Ensure `.cu` files are compiled as CUDA code
set_source_files_properties(${CUDA_SRC_FILES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CUDA_HDR_FILES} PROPERTIES LANGUAGE CUDA)

# Enable CUDA features
set_target_properties(ml_kernels PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(ml_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(ml_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ml_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Apply compile options
target_compile_options(ml_kernels PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_OPTS} ${CXX_RELWITHDEBINFO_OPTS} ${CXX_RELEASE_OPTS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEBUG_OPTS} ${CUDA_RELWITHDEBINFO_OPTS} ${CUDA_RELEASE_OPTS}>
)

# Include directories
target_include_directories(ml_kernels PUBLIC 
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

# Link CUDA and other dependencies
target_link_libraries(ml_kernels PUBLIC ${PACKAGES} CUDA::cudart CUDA::cuda_driver CUDA::curand)

add_subdirectory(examples)