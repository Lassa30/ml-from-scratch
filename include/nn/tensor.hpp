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

  int64_t stride(int64_t dim) const;
  int64_t shape(int64_t dim) const;

  int64_t offset() const noexcept;
  int64_t numel() const noexcept;
  int64_t memsize() const noexcept;
  const std::shared_ptr<std::vector<float>> data() const noexcept;

public:
  Tensor T();
  bool empty() const noexcept;
};

class Tensor::TensorImpl {
private:
  Tensor& tensor_;

  Stride stride_;
  Shape shape_;
  int64_t offset_;

  std::shared_ptr<Storage> storage_;

public:
  TensorImpl(Tensor& tensor);
  TensorImpl(Tensor& tensor, const Shape& shape);

  const Stride& stride() const noexcept;
  const Shape& shape() const noexcept;

  int64_t stride(int64_t dim) const;
  int64_t shape(int64_t dim) const;

  int64_t offset() const noexcept;
  int64_t numel() const noexcept;
  int64_t memsize() const noexcept;

  const std::shared_ptr<std::vector<float>> data() const noexcept;

  Tensor T();
  bool empty() const noexcept;

private:
  bool checkShapeStrideValidity();
};

namespace tensor {
// mlfs::nn::tensor::ones({SHAPE}} would live here...
}

}  // namespace nn
}  // namespace mlfs
