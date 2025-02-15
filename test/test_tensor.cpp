#include <nn/tensor.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlfs::nn;

bool checkShape(const Tensor& tensor, const std::vector<std::int64_t> desired) {
  return tensor.shape().data() == desired;
}
bool checkStride(const Tensor& tensor,
                 const std::vector<std::int64_t> desired) {
  return tensor.stride().data() == desired;
}

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


TEST_CASE("Shape and Stride alignment.") {
  std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>
      shape_and_stride = {{{}, {}},         {{1}, {1}},
                          {{1, 1}, {1, 1}}, {{1, 1, 1}, {1, 1, 1}},
                          {{5, 5}, {5, 1}}, {{7, 15}, {15, 1}},
                          {{5, 1}, {1, 1}}, {{2, 2, 2, 2}, {8, 4, 2, 1}}};

  for (auto& [shape, stride] : shape_and_stride) {
    auto a = Tensor(Shape(shape));
    CHECK(checkShape(a, shape));
    CHECK(checkStride(a, stride));
  }
}

// TODO: refactor all those subcases to be the only function call.
// Also we can test basic shape constructors to avoid the lines:
// -------------------------------
// CHECK(checkShape(a, {5, 1}));
// CHECK(checkStride(a, {1, 1}));
// -------------------------------

// TEST_CASE("Tensor transpose") {
//
//   SUBCASE("scalar tensor") {
//     Tensor a{Shape({1})};
//     Tensor a_T = a.T();
//
//     std::vector<int64_t> desired = {1};
//
//     CHECK(a.shape().data() == desired);
//     CHECK(a.stride().data() == desired);
//     CHECK(a_T.shape().data() == desired);
//     CHECK(a_T.stride().data() == desired);
//   }
//
//   SUBCASE("vector tensor") {
//     Tensor a(Shape{5, 1});
//     Tensor a_T = a.T();
//
//     CHECK(checkShape(a, {5, 1}));
//     CHECK(checkStride(a, {1, 1}));
//
//     CHECK(checkShape(a_T, {1, 5}));
//     CHECK(checkStride(a_T, {1, 1}));
//   }
//
//   SUBCASE("matrix tensor") {
//     Tensor a(Shape{5, 2});
//     Tensor a_T = a.T();
//
//     CHECK(checkShape(a, {5, 2}));
//     CHECK(checkStride(a, {1, 1}));
//
//     CHECK(checkShape(a_T, {1, 5}));
//     CHECK(checkStride(a_T, {1, 1}));
//   }
// }


TEST_CASE("Tensor resize") {}

TEST_CASE("Tensor copy `n` move") {}