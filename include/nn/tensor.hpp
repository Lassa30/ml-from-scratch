#pragma once

#include <nn/storage.hpp>
#include <nn/utils.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using std::int64_t, std::uint64_t;

namespace mlfs {

namespace nn {

class Tensor {
private:
  class TensorImpl;
  std::shared_ptr<TensorImpl> impl_;

public:
  Tensor();
  Tensor(const Shape& shape);
  const Stride& stride() const noexcept;
  const Shape& shape() const noexcept;

  std::int64_t stride(std::int64_t dim) const;
  std::int64_t shape(std::int64_t dim) const;

  std::int64_t offset() const noexcept;
  std::int64_t numel() const noexcept;
  std::int64_t memsize() const noexcept;
  bool empty() const noexcept;

  const std::shared_ptr<std::vector<float>> data() const noexcept;

public:
  Tensor transpose();
  Tensor permute();
};

class Tensor::TensorImpl {
private:
  Shape shape_;
  Stride stride_;
  std::int64_t offset_;

  std::shared_ptr<Storage> storage_;

public:
  TensorImpl();
  TensorImpl(const Shape& shape);
  TensorImpl(const TensorImpl&) = default;
  TensorImpl(TensorImpl&&) noexcept = default;

  const Stride& stride() const noexcept;
  const Shape& shape() const noexcept;

  std::int64_t stride(std::int64_t dim) const;
  std::int64_t shape(std::int64_t dim) const;

  std::int64_t offset() const noexcept;
  std::int64_t numel() const noexcept;
  std::int64_t memsize() const noexcept;

  const std::shared_ptr<std::vector<float>> data() const noexcept;

  TensorImpl transpose();
  TensorImpl permute();

  bool empty() const noexcept;

private:
  bool checkShapeStrideValidity();
  Stride defaultStride(const Shape& shape);
};

namespace tensor {
// mlfs::nn::tensor::ones({SHAPE}} would live here...
}

}  // namespace nn
}  // namespace mlfs
