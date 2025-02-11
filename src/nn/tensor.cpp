#include <nn/tensor.hpp>

#include <string>

namespace mlfs {
namespace nn {

Tensor::TensorImpl::TensorImpl(Tensor& tensor)
    : tensor_{tensor}
    , stride_{}
    , shape_{}
    , offset_{-1}
    , storage_{std::make_shared<Storage>()} {}

const Stride& Tensor::TensorImpl::stride() const noexcept { return stride_; }
int64_t Tensor::TensorImpl::offset() const noexcept { return offset_; }

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

int64_t Tensor::TensorImpl::numel() const noexcept {
  if (shape().empty()) {
    int64_t dim_product = 1;
    for (auto dim : shape()) {
      dim_product *= dim;
    }
    return dim_product;
  }
  return 0;
}

int64_t Tensor::TensorImpl::memsize() const noexcept {
  return numel() * sizeof(float);
}

// Tensor definition
Tensor::Tensor() : impl_{std::make_shared<TensorImpl>(*this)} {}

const Stride& Tensor::stride() const noexcept { return impl_->stride(); }
int64_t Tensor::stride(int64_t dim) const { return impl_->stride(dim); }
const std::vector<int64_t>& Tensor::shape() const noexcept {
  return impl_->shape();
}
int64_t Tensor::shape(int64_t dim) const { return impl_->shape(dim); }
int64_t Tensor::offset() const noexcept { return impl_->offset(); }
int64_t Tensor::numel() const noexcept { return impl_->numel(); }
int64_t Tensor::memsize() const noexcept { return impl_->memsize(); }

}  // namespace nn
}  // namespace mlfs
