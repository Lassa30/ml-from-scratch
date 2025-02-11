#include <nn/tensor.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("shape, stride, offset for an empty tensor.") {
  mlfs::nn::Tensor a{};

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
  SUBCASE("numel, memsize") { CHECK(numel_is_zero * memsize_is_zero == 0); }
}

TEST_CASE("Tensor views and reshape") {}

TEST_CASE("Tensor resize") {}

TEST_CASE("Tensor copy `n` move") {}