#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nn/storage.hpp>

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
    std::unique_ptr<TensorImpl> impl;
    std::shared_ptr<Storage> storagePtr;

  public:
    Tensor();

    const Stride& stride() const noexcept;
    int64_t stride(int64_t dim) const;
    const std::vector<int64_t>& shape() const noexcept;
    int64_t shape(int64_t dim) const;
    int64_t offset() const noexcept;

    std::shared_ptr<float> data_();
    const std::shared_ptr<float> data();
};

class Tensor::TensorImpl {
  private:
    Tensor& tensor_;

    Stride stride_;
    std::vector<int64_t> shape_;
    int64_t offset_;

  public:
    TensorImpl(Tensor& tensor);

    const Stride& stride() const noexcept;
    int64_t stride(int64_t dim) const;

    const std::vector<int64_t>& shape() const noexcept;
    int64_t shape(int64_t dim) const;

    int64_t offset() const noexcept;
};

namespace tensor {
// mlfs::nn::tensor::ones({SHAPE}} would live here...
}

}  // namespace nn
}  // namespace mlfs