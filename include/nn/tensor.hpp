#pragma once

#include <nn/storage.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using std::int64_t, std::uint64_t;

namespace mlfs {

namespace nn {

// TODO: copy and move for stride (?)
// TODO: do I really need a Stride and Shape classes?
class Stride {
public:
  Stride() : data_{} {}

  int64_t operator[](int64_t dim) const { return data_[dim]; }

  bool empty() const noexcept { return data_.empty(); }

  int64_t size() const noexcept { return data_.size(); }

private:
  std::vector<int64_t> data_;
};

class Tensor {
private:
  class TensorImpl;
  std::shared_ptr<TensorImpl> impl_;

public:
  Tensor();

  const Stride& stride() const noexcept;
  const std::vector<int64_t>& shape() const noexcept;

  int64_t stride(int64_t dim) const;
  int64_t shape(int64_t dim) const;
  int64_t offset() const noexcept;

  int64_t numel() const noexcept;
  int64_t memsize() const noexcept;
  std::shared_ptr<float> data();
};

class Tensor::TensorImpl {
private:
  Tensor& tensor_;

  Stride stride_;
  std::vector<int64_t> shape_;
  int64_t offset_;

  std::shared_ptr<Storage> storage_;

public:
  TensorImpl(Tensor& tensor);

  const Stride& stride() const noexcept;
  int64_t stride(int64_t dim) const;

  const std::vector<int64_t>& shape() const noexcept;
  int64_t shape(int64_t dim) const;

  int64_t offset() const noexcept;
  int64_t numel() const noexcept;
  int64_t memsize() const noexcept;
};

namespace tensor {
// mlfs::nn::tensor::ones({SHAPE}} would live here...
}

}  // namespace nn
}  // namespace mlfs
