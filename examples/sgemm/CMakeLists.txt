# Options
option(BUILD_AVX "Enable AVX support" OFF)
option(BUILD_CUDA "Enable CUDA support" OFF)
option(CUDA_ARCH "CUDA architecture" "86")

# Headers
set(HEADERS sgemm_cpu.h)

if (BUILD_AVX)
    set(HEADERS ${HEADERS} sgemm_avx.h)
endif()

check_language(CUDA)

if (BUILD_CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(HEADERS ${HEADERS} sgemm_cuda.h sgemm_cuda.cu)
    endif()
endif()

# Executable
add_executable(sgemm-example sgemm-example.cpp ${HEADERS})

# Include directories
target_include_directories(sgemm-example PUBLIC ${PROJECT_SOURCE_DIR}/include)

# AVX Support
if (BUILD_AVX)
    message(STATUS "Building with AVX support.")
    target_compile_options(sgemm-example PRIVATE -mavx2)
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_avx)
endif()

# CUDA Support
if (BUILD_CUDA)
    if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

        # Find CUDA Toolkit
        find_package(CUDAToolkit REQUIRED)

        if (CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit found.")
            
            # CUDA-specific configurations
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})
            set_property(TARGET sgemm-example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

            # CUDA compile and link options
            target_compile_options(sgemm-example PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA__>
                $<$<COMPILE_LANGUAGE:CXX>:-D__CUDA__>
            )
            target_link_libraries(sgemm-example PRIVATE
                ${PROJECT_NAME}_cuda
                CUDA::cudart
                CUDA::cuda_driver
                CUDA::cublas
            )
        else()
            message(WARNING "CUDA Toolkit not found. Disabling CUDA support.")
        endif()
    else()
        message(WARNING "CUDA compiler not found. Ensure nvcc and the CUDA Toolkit are installed.")
    endif()
endif()

# Always link CPU library
target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cpu)