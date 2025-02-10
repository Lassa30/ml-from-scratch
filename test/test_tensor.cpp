#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>

#include <nn/tensor.hpp>

/*
auto shape_1 = {1, 1, 1} // [DxWxH]
Tensor b = nn::tensor::ones(shape_1);
Tensor b1 = Tensor(shape_1, 42); // creates a scalar tensor: 42!
Tensor c = nn::tensor::zeros({1, 1, 1});

// copy ctor have to copy all metadata about tensor, but not the tensor itself!
Tensor a = b;

*/
TEST_CASE("basic constructors") { CHECK(1 == 1); }