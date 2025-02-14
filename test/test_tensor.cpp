#include <nn/tensor.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlfs::nn;

TEST_CASE("shape, stride, offset for an empty tensor.") {
  Tensor a{};

  SUBCASE("SUBCASE: shape") {
    CHECK(a.shape().empty());
    CHECK_THROWS(a.shape(0) == 0);
    CHECK_THROWS(a.shape(1));
  }

  SUBCASE("SUBCASE: stride") {
    CHECK(a.stride().empty());
    CHECK_THROWS(a.stride(0) == 0);
    CHECK_THROWS(a.stride(1));
  }

  SUBCASE("SUBCASE: offset") { CHECK(a.offset() == -1); }

  bool numel_is_zero = a.numel() == 0;
  bool memsize_is_zero = a.memsize() == 0;
  SUBCASE("numel, memsize") { CHECK(numel_is_zero * memsize_is_zero); }
}

TEST_CASE("Tensor transpose dim=0") {
  // Shape is Depth x Height x Width
  // So this tensor is a two vertically stacked column vectors xD
  Tensor a{Shape({2, 3, 1})};
  Tensor a_T = a.T();

  std::vector<int64_t> desired = {2, 3, 1};
  CHECK(a.shape().data() == desired);
  desired = {3, 1, 1};
  // TODO: it should work! Implement Stride 'n' Shape to work together
  CHECK(a.stride().data() == desired);
}

TEST_CASE("Tensor transpose dim=1") {}

TEST_CASE("Tensor transpose dim=2") {}

TEST_CASE("Tensor transpose dim=3") {}

TEST_CASE("Tensor views and reshape") {}

TEST_CASE("Tensor resize") {}

TEST_CASE("Tensor copy `n` move") {}