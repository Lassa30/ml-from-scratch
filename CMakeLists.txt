# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

set(MLFS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MLFS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MLFS_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add naive_bayes
add_library(naive_bayes ${MLFS_INCLUDE}/models/naive_bayes.hpp ${MLFS_SRC}/models/naive_bayes.cpp)
target_link_libraries(naive_bayes Eigen3::Eigen)

# tiny examples
add_executable(naive_bayes_example ${MLFS_EXAMPLES}/naive_bayes_test.cpp)
target_link_libraries(naive_bayes_example naive_bayes)