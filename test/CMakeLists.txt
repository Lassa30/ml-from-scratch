# // clang-format off
enable_testing()

add_executable(test_tensor test_tensor.cpp)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_storage test_storage.cpp)
add_test(NAME storage COMMAND test_storage)

target_link_libraries(test_storage tensor doctest)
target_link_libraries(test_tensor tensor doctest)
