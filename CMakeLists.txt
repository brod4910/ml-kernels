cmake_minimum_required(VERSION 3.21)

project(
        ml_kernels
        VERSION 0.0.1
        DESCRIPTION "A collection of machine learning kernels for CPU, AVX and CUDA"
        HOMEPAGE_URL "https://github.com/brod4910/ml-kernels"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_CUDA "Build CUDA library version" OFF)
option(BUILD_AVX "Build AVX library version" OFF)

# ---- Packages ----
find_package(fmt REQUIRED)
#find_package(spdlog REQUIRED)

#set(PACKAGES fmt::fmt spdlog::spdlog)
set(PACKAGES fmt::fmt)

# ---- AVX Build  ----
if(${BUILD_AVX})
    set(AVX_LIB_NAME ${PROJECT_NAME}_avx)
    file(GLOB_RECURSE AVX_SRC_FILES ${PROJECT_SOURCE_DIR}/src/avx/*.cpp)
    file(GLOB_RECURSE AVX_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/avx/*.h)

    add_library(${AVX_LIB_NAME} SHARED ${AVX_SRC_FILES} ${AVX_HDR_FILES})

    target_compile_options(${AVX_LIB_NAME} PUBLIC -Wall -std=c++17 -O3 -mavx2 -mavx -march=native)
    target_link_libraries(${AVX_LIB_NAME} PRIVATE ${PACKAGES})
    target_include_directories(
            ${AVX_LIB_NAME}
            PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    )
endif()

# ---- CUDA Build  ----

if(${BUILD_CUDA})
    set(CUDA_LIB_NAME ${PROJECT_NAME}_cuda)
    option(CUDA_ARCH "CUDA architecture" "89")
    # Find CUDA package and include directories
    enable_language(CUDA)
    find_package(CUDAToolkit)

    # Specify the CUDA compute capability
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})  # Change the compute capability as needed

    # Add the CUDA sources to the project
    file(GLOB_RECURSE CU_SRC_FILES ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp)
    file(GLOB_RECURSE CUDA_SRC_FILES ${PROJECT_SOURCE_DIR}/src/cuda/*.cu)
    file(GLOB_RECURSE CU_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/cuda/*.h)
    file(GLOB_RECURSE CUDA_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/cuda/*.cuh)

    add_library(${CUDA_LIB_NAME} SHARED ${CU_SRC_FILES} ${CU_HDR_FILES} ${CUDA_SRC_FILES} ${CUDA_HDR_FILES})
    set_property(TARGET ${CUDA_LIB_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_link_libraries(${CUDA_LIB_NAME} PRIVATE ${PACKAGES} CUDA::cudart CUDA::cuda_driver)
    set_target_properties(${CUDA_LIB_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
    target_include_directories(
            ${CUDA_LIB_NAME}
            PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    )
endif()

# ---- CPU Build  ----
set(CPU_LIB_NAME ${PROJECT_NAME}_cpu)
file(GLOB_RECURSE CPU_SRC_FILES ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)
file(GLOB_RECURSE CPU_HDR_FILES ${PROJECT_SOURCE_DIR}/include/mlkl/cpu/*.h)

add_library(${CPU_LIB_NAME} SHARED ${CPU_SRC_FILES} ${CPU_HDR_FILES})
target_link_libraries(${CPU_LIB_NAME} PRIVATE ${PACKAGES})
target_include_directories(
        ${CPU_LIB_NAME}
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_options(${CPU_LIB_NAME} PUBLIC -Wall -std=c++17 -O3 -ffast-math)

add_subdirectory(examples)