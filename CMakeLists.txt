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
add_library(model ${MLFS_INCLUDE}/nn/model.hpp ${MLFS_SRC}/nn/model.cpp)
add_library(optimizer ${MLFS_INCLUDE}/nn/optimizer.hpp ${MLFS_SRC}/nn/optimizer.cpp)
add_library(layers ${MLFS_INCLUDE}/nn/layers.hpp ${MLFS_SRC}/nn/layers.cpp)
add_library(loss_function ${MLFS_INCLUDE}/nn/loss_function.hpp ${MLFS_SRC}/nn/loss_function.cpp)
target_link_libraries(
  model 
  layers 
  optimizer 
  loss_function
)

# tiny examples
add_executable(naive_bayes_example ${MLFS_EXAMPLES}/naive_bayes_example.cpp)
target_link_libraries(naive_bayes_example naive_bayes)