# general info
cmake_minimum_required(VERSION 3.10)

project(ml-from-scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (${CMAKE_BUILD_TYPE} EQUAL "Release")
    add_compile_options(-Ofast)
endif()


# adding libraries
add_library(matrix INTERFACE ../include/utils/matrix.hpp)

add_library(nb INTERFACE ../include/models/naive_bayes.hpp)

add_library(utils INTERFACE ../include/utils/utils.hpp)

add_library(linear_regression INTERFACE ../include/models/linear_regression.hpp)

set(MLFS_HEADERS nb linear_regression matrix utils)

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
