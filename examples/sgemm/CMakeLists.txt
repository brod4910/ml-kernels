set(HEADERS sgemm_cpu.h)

if (${BUILD_AVX})
    set(HEADERS ${HEADERS} sgemm_avx.h)
endif ()

add_executable(sgemm-example sgemm-example.cpp ${HEADERS})

target_include_directories(sgemm-example PUBLIC ${PROJECT_SOURCE_DIR}/include)
if (${BUILD_AVX})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_avx)
endif()

if (${BUILD_CUDA})
    target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cuda)
endif()

target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cpu)
