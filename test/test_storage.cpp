#include <nn/storage.hpp>
#include <nn/tensor.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlfs::nn;

TEST_CASE("to begin with... an empty tensor") { Tensor a{}; }