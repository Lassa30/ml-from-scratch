#include <nn/storage.hpp>
#include <nn/tensor.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlfs::nn;

/*
  Desired behaviour:
  ```c++
  Tensor(Shape{1, 1, 1}) // OK
  Tensor({1, 1, 1}) // OK


  ```
  If we need to create a tensor with shape {1, 1, 1} -- OK
  If we need to create a tensor with values {1, 1, 1} -- then we need also
  specify, which shape the data has(the stride is gonna be default OR ofc maybe
  set using set_stride() member function.)
  TODO: implement Tensor(data, shape, stride)
*/

TEST_CASE("to begin with... a scalar tensor") {
  Tensor a{Shape({1, 1, 1})};
  // TODO: Do I need this one???
  // Tensor b{{1,1,1}};

  CHECK(a.numel() == 1);
  CHECK(a.numel() * sizeof(float) == a.memsize());
  REQUIRE(a.data() != nullptr);  // to avoid dereferencing a.data()

  std::vector<float> desired = {0.0};
  CHECK(*a.data() == desired);
  CHECK(a.offset() == 0);
}

TEST_CASE("empty tensor edge cases.") {
  Tensor a{};
  CHECK(a.empty());
  CHECK(a.numel() == 0);
  CHECK(a.memsize() == 0);
  REQUIRE(a.data() == nullptr);
}