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
}