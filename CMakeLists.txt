cmake_minimum_required(VERSION 3.20)
project(cpp_neural_network_framework LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_library(nnf src/tensor.cpp)
target_include_directories(nnf PUBLIC include)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE nnf)

add_executable(train_line examples/train_line.cpp)
target_link_libraries(train_line PRIVATE nnf)

# tests: run with ctest --test-dir build
enable_testing()
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE nnf)
add_test(NAME tensor_tests COMMAND test_tensor)

add_executable(test_training tests/test_training.cpp)
target_link_libraries(test_training PRIVATE nnf)
add_test(NAME training_tests COMMAND test_training)
