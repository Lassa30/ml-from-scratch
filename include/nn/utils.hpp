#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Shape {
public:
  Shape() : data_{} {}

  Shape(const std::vector<std::int64_t>& shape) {
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

  std::int64_t operator[](std::int64_t dim) const { return data_.at(dim); }

  bool empty() const noexcept { return data_.empty(); }

  std::int64_t size() const noexcept { return data_.size(); }

  std::int64_t numel() const noexcept {
    if (!data_.empty()) {
      std::int64_t dim_product = 1;
      for (auto dim : data_) {
        dim_product *= dim;
      }
      return dim_product;
    }
    return 0;
  }

  const std::vector<std::int64_t>& data() const noexcept { return data_; }

private:
  bool ShapeIsValid(const std::vector<std::int64_t>& given_shape) {
    return given_shape.empty() ||
           std::all_of(given_shape.begin(), given_shape.end(),
                       [](std::int64_t dim) { return dim > 0; });
  }
  std::vector<std::int64_t> data_;
};

class Stride {
public:
  Stride() : data_{} {}
  Stride(const std::vector<std::int64_t>& stride) : data_{stride} {}

  Stride(const Shape& shape) {
    if (!shape.empty()) {
      int dims = shape.data().size();
      data_ = std::vector<std::int64_t>(dims, 1);
      std::int64_t current_stride = 1;
      for (int ind = dims - 1; ind >= 0; --ind) {
        stride_vector[ind] = current_stride;
        current_stride *= shape[ind];
      }
    }
    data_ = std::vector<std::int64_t>();
  }

  Stride(const Stride&) = default;
  Stride(Stride&&) noexcept = default;

  Stride& operator=(const Stride&) = default;
  Stride& operator=(Stride&&) noexcept = default;

  ~Stride() = default;

  std::int64_t operator[](std::int64_t dim) const { return data_.at(dim); }

  bool empty() const noexcept { return data_.empty(); }

  std::int64_t size() const noexcept { return data_.size(); }

  const std::vector<std::int64_t>& data() const noexcept { return data_; }

private:
  std::vector<std::int64_t> data_;
};