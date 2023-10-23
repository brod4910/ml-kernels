add_executable(conv2d-example conv2d-example.cpp)

target_include_directories(conv2d-example PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(conv2d-example PRIVATE ${PROJECT_NAME})
