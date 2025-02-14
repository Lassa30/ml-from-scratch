#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Stride {
public:
  Stride() : data_{} {}

  Stride(const std::vector<int64_t>& stride) : data_{stride} {}

  Stride(const Stride&) = default;
  Stride(Stride&&) noexcept = default;

  Stride& operator=(const Stride&) = default;
  Stride& operator=(Stride&&) noexcept = default;

  ~Stride() = default;

  int64_t operator[](int64_t dim) const { return data_[dim]; }

  bool empty() const noexcept { return data_.empty(); }

  int64_t size() const noexcept { return data_.size(); }

  const std::vector<int64_t>& data() const noexcept { return data_; }

  Stride T() { return Stride(); }

private:
  std::vector<int64_t> data_;
};

class Shape {
public:
  Shape() : data_{} {}

  Shape(const std::vector<int64_t>& shape) {
    if (!ShapeIsValid(shape)) {
      throw std::invalid_argument(
          "Invalid shape-vector is provided.\nValid shape-vector satisfies any "
          "of the two provided conditions:\n1)It's empty.\n2)All elements are "
          ">= 0.");
    }
    data_ = shape;
  }

  Shape(const Shape&) = default;
  Shape(Shape&&) noexcept = default;

  Shape& operator=(const Shape&) = default;
  Shape& operator=(Shape&&) noexcept = default;

  ~Shape() = default;

  int64_t operator[](int64_t dim) const { return data_[dim]; }

  bool empty() const noexcept { return data_.empty(); }

  int64_t size() const noexcept { return data_.size(); }

  int64_t numel() const noexcept {
    if (!data_.empty()) {
      int64_t dim_product = 1;
      for (auto dim : data_) {
        dim_product *= dim;
      }
      return dim_product;
    }
    return 0;
  }

  const std::vector<int64_t>& data() const noexcept { return data_; }

  Shape T() { return Shape(); }

private:
  bool ShapeIsValid(const std::vector<int64_t>& given_shape) {
    return given_shape.empty() ||
           std::all_of(given_shape.begin(), given_shape.end(),
                       [](int64_t dim) { return dim > 0; });
  }
  std::vector<int64_t> data_;
};