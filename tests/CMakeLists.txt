add_executable(matrix_test matrix_test.cpp)
target_link_libraries(matrix_test matrix)

add_executable(naive_bayes_test naive_bayes_test.cpp)
target_link_libraries(naive_bayes_test nb matrix)

add_executable(optimizer_test optimizer_test.cpp)
target_link_libraries(optimizer_test matrix optimizer)
