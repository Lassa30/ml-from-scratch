# // clang-format off

# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(MLFS_BUILD_EXAMPLES FALSE)
option(MLFS_BUILD_TESTS TRUE)

# mlfs cmake variables
set(MLFS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${MLFS_INCLUDE})

set(MLFS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# naive_bayes
find_package(Eigen3 REQUIRED)
add_library(naive_bayes ${MLFS_INCLUDE}/models/naive_bayes.hpp
                        ${MLFS_SRC}/models/naive_bayes.cpp)
target_link_libraries(naive_bayes Eigen3::Eigen)


# add_library(model ${MLFS_INCLUDE}/nn/model.hpp ${MLFS_SRC}/nn/model.cpp)
# add_library(optimizer ${MLFS_INCLUDE}/nn/optimizer.hpp
#                       ${MLFS_SRC}/nn/optimizer.cpp)
# add_library(layers ${MLFS_INCLUDE}/nn/layers.hpp ${MLFS_SRC}/nn/layers.cpp)
# add_library(loss_function ${MLFS_INCLUDE}/nn/loss_function.hpp
#                           ${MLFS_SRC}/nn/loss_function.cpp)

add_library(
  tensor
  ${MLFS_INCLUDE}/nn/tensor.hpp ${MLFS_SRC}/nn/tensor.cpp
  ${MLFS_INCLUDE}/nn/storage.hpp ${MLFS_SRC}/nn/storage.cpp
  ${MLFS_INCLUDE}/nn/utils.hpp ${MLFS_SRC}/nn/utils.cpp)

# add_library(autograd ${MLFS_INCLUDE}/nn/autograd.hpp
#                      ${MLFS_SRC}/nn/autograd.cpp)
# target_link_libraries(tensor autograd)

# examples
if(MLFS_BUILD_EXAMPLES)
  set(MLFS_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  add_executable(naive_bayes_example ${MLFS_EXAMPLES}/naive_bayes_example.cpp)
  target_link_libraries(naive_bayes_example naive_bayes)
endif()

if(MLFS_BUILD_TESTS)
  add_subdirectory(doctest)
  option(DOCTEST_NO_INSTALL ON)
  add_subdirectory(test)
endif()