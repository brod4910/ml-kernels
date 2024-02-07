cmake_minimum_required(VERSION 3.21)

project(
        ml_kernels
        VERSION 0.0.1
        DESCRIPTION "A C++ ML framework named after my dog"
        HOMEPAGE_URL "https://github.com/brod4910/luna"
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

# ---- Luna AVX Build  ----
if(${BUILD_AVX})
    file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/avx/*.cpp)
    file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/include/avx/*.h)

    add_library(avx SHARED ${SRC_FILES} ${HDR_FILES})

    target_compile_options(avx PUBLIC -Wall -std=c++17 -O3 -mavx2 -mavx -march=native)
    target_link_libraries(avx PRIVATE ${PACKAGES})
    target_include_directories(
            avx
            PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    )
endif()

# ---- Luna CUDA Build  ----

if(${BUILD_CUDA})
    option(CUDA_ARCH "CUDA architecture" "61")
    # Find CUDA package and include directories
    find_package(CUDA REQUIRED)

    # Specify the CUDA compute capability
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)  # Change the compute capability as needed

    # Add the CUDA sources to the project
    file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp)
    file(GLOB_RECURSE CUDA_SRC_FILES ${PROJECT_SOURCE_DIR}/include/cuda/*.cu)
    file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/include/cuda/*.h)
    file(GLOB_RECURSE CUDA_HDR_FILES ${PROJECT_SOURCE_DIR}/include/cuda/*.cuh)

    cuda_add_library(cuda SHARED ${SRC_FILES} ${HDR_FILES} ${CUDA_SRC_FILES} ${CUDA_HDR_FILES})
    set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_link_libraries(cuda PRIVATE ${PACKAGES})
    target_include_directories(
            cuda
            PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/cuda>"
    )
endif()

# ---- Luna CPU Build  ----
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)
file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/include/cpu/*.h)

add_library(cpu SHARED ${SRC_FILES} ${HDR_FILES})
target_link_libraries(luna_cpu PRIVATE ${PACKAGES})
target_include_directories(
        cpu
        PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_compile_options(cpu PUBLIC -Wall -std=c++17 -O3 -ffast-math)

add_subdirectory(examples)