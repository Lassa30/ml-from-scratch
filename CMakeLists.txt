# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

# mlfs variables
set(MLFS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MLFS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MLFS_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples)

include_directories(${MLFS_INCLUDE})

# add naive_bayes
add_library(naive_bayes ${MLFS_INCLUDE}/models/naive_bayes.hpp ${MLFS_SRC}/models/naive_bayes.cpp)
target_link_libraries(naive_bayes Eigen3::Eigen)

# nn constructor
add_library(model ${MLFS_INCLUDE}/model.hpp ${MLFS_SRC}/model.cpp)
add_library(optimizer ${MLFS_INCLUDE}/optimizer.hpp ${MLFS_SRC}/optimizer.cpp)
add_library(layers ${MLFS_INCLUDE}/layers.hpp ${MLFS_SRC}/layers.cpp)

# tiny examples
add_executable(naive_bayes_example ${MLFS_EXAMPLES}/naive_bayes_example.cpp)
target_link_libraries(naive_bayes_example naive_bayes)