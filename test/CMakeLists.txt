# // clang-format off
add_executable(test_tensor test_tensor.cpp)
add_executable(test_storage test_storage.cpp)

target_link_libraries(test_storage tensor doctest)
target_link_libraries(test_tensor tensor doctest)
