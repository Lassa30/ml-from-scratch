# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# if (${CMAKE_BUILD_TYPE} EQUAL "Release")
#     add_compile_options(-Ofast)
# endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# adding libraries
add_library(matrix ../include/utils/matrix.hpp ../src/utils/matrix.cpp)

add_library(nb ../include/models/naive_bayes.hpp ../src/models/naive_bayes.cpp)

add_library(linear_regression ../include/models/linear_regression.hpp ../src/models/linear_regression.cpp)

add_library(optimizer INTERFACE ../include/utils/optimizer.hpp)

add_library(utils INTERFACE ../include/utils/utils.hpp)

set(MLFS_HEADERS nb utils optimizer linear_regression matrix)
# include directories
foreach(mlfs_header IN LISTS MLFS_HEADERS)
    target_include_directories(${mlfs_header} INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    ) 
endforeach()

# subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
