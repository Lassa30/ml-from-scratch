#include <nn/tensor.hpp>

#include <string>

namespace mlfs {
namespace nn {

Tensor::TensorImpl::TensorImpl(Tensor& tensor)
    : tensor_{tensor}, stride_{}, shape_{}, offset_{-1} {}

int64_t Tensor::TensorImpl::offset() const noexcept { return offset_; }

Tensor::Tensor()
    : impl{std::make_unique<TensorImpl>(*this)}
    , storagePtr{std::make_shared<Storage>()} {}

const Stride& Tensor::TensorImpl::stride() const noexcept { return stride_; }

int64_t Tensor::TensorImpl::stride(int64_t dim) const {
  if (dim < 0 || stride_.size() <= dim) {
    throw std::invalid_argument(
        std::string("Wrong dimension is provided: dim=") + std::to_string(dim));
  }
  return stride_[dim];
}

const std::vector<int64_t>& Tensor::TensorImpl::shape() const noexcept {
  return shape_;
}

int64_t Tensor::TensorImpl::shape(int64_t dim) const {
  if (dim < 0 || shape_.size() <= dim) {
    throw std::invalid_argument(
        std::string("Wrong dimension is provided: dim=") + std::to_string(dim));
  }
  return shape_[dim];
}

const Stride& Tensor::stride() const noexcept { return impl->stride(); }

int64_t Tensor::stride(int64_t dim) const { return impl->stride(dim); }

const std::vector<int64_t>& Tensor::shape() const noexcept {
  return impl->shape();
}

int64_t Tensor::shape(int64_t dim) const { return impl->shape(dim); }

int64_t Tensor::offset() const noexcept { return impl->offset(); }

}  // namespace nn
}  // namespace mlfs
