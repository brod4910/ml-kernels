add_executable(sgemm-example sgemm-example.cpp)

target_include_directories(sgemm-example PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sgemm-example PRIVATE ${PROJECT_NAME}_cpu)
