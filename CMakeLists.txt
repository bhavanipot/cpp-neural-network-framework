cmake_minimum_required(VERSION 3.20)
project(cpp_neural_network_framework LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_library(nnf src/tensor.cpp)
target_include_directories(nnf PUBLIC include)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE nnf)
