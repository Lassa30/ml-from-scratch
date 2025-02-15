#include <nn/tensor.hpp>

#include <string>

namespace mlfs {
namespace nn {

/*-----------------------------------------------------
__TensorImpl__
-----------------------------------------------------*/
Tensor::TensorImpl::TensorImpl()
    : shape_{}, stride_{}, offset_{-1}, storage_{std::make_shared<Storage>()} {}

Tensor::TensorImpl::TensorImpl(const Shape& shape)
    : shape_{shape}
    , stride_{shape}
    , offset_{0}
    , storage_{std::make_shared<Storage>(shape)} {}

const Stride& Tensor::TensorImpl::stride() const noexcept { return stride_; }
int64_t Tensor::TensorImpl::offset() const noexcept { return offset_; }

int64_t Tensor::TensorImpl::stride(int64_t dim) const {
  if (dim < 0 || stride_.size() <= dim) {
    throw std::invalid_argument(
        std::string("Wrong dimension is provided: dim=") + std::to_string(dim));
  }
  return stride_[dim];
}

const Shape& Tensor::TensorImpl::shape() const noexcept { return shape_; }

int64_t Tensor::TensorImpl::shape(int64_t dim) const {
  if (dim < 0 || shape_.size() <= dim) {
    throw std::invalid_argument(
        std::string("Wrong dimension is provided: dim=") + std::to_string(dim));
  }
  return shape_[dim];
}

int64_t Tensor::TensorImpl::numel() const noexcept { return shape_.numel(); }

int64_t Tensor::TensorImpl::memsize() const noexcept {
  return numel() * sizeof(float);
}

const std::shared_ptr<std::vector<float>>
Tensor::TensorImpl::data() const noexcept {
  return storage_->data();
}

Tensor::TensorImpl Tensor::TensorImpl::transpose() {
  return TensorImpl(shape_.transpose());
}

bool Tensor::TensorImpl::empty() const noexcept {
  return storage_->data() == nullptr;
}

/*-----------------------------------------------------
__Tensor__
-----------------------------------------------------*/
Tensor::Tensor() : impl_{std::make_shared<TensorImpl>()} {}
Tensor::Tensor(const Shape& shape)
    : impl_{std::make_shared<TensorImpl>(shape)} {}

const Stride& Tensor::stride() const noexcept { return impl_->stride(); }
int64_t Tensor::stride(int64_t dim) const { return impl_->stride(dim); }

const Shape& Tensor::shape() const noexcept { return impl_->shape(); }
int64_t Tensor::shape(int64_t dim) const { return impl_->shape(dim); }

int64_t Tensor::offset() const noexcept { return impl_->offset(); }
int64_t Tensor::numel() const noexcept { return impl_->numel(); }
int64_t Tensor::memsize() const noexcept { return impl_->memsize(); }
const std::shared_ptr<std::vector<float>> Tensor::data() const noexcept {
  return impl_->data();
}
bool Tensor::empty() const noexcept { return impl_->empty(); }
  
Tensor Tensor::transpose() {
  auto tensor_tmp = Tensor();
  tensor_tmp.impl_ = std::make_shared<Tensor::TensorImpl>(impl_->transpose());
  return tensor_tmp;
}

}  // namespace nn
}  // namespace mlfs
