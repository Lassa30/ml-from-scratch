# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Ofast)

# adding libraries
add_library(matrix INTERFACE ../include/utils/matrix.hpp)

add_library(nb INTERFACE ../include/models/naive_bayes.hpp)

add_library(utils INTERFACE ../include/utils/utils.hpp)

# include directories
target_include_directories(matrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(nb INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
