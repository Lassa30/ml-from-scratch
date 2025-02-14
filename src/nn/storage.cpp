#include <stdexcept>

#include <nn/storage.hpp>
#include <nn/utils.hpp>

using namespace mlfs::nn;

Storage::StorageImpl::StorageImpl(Storage& storage)
    : storage_{storage}, data_{nullptr} {}
Storage::StorageImpl::StorageImpl(Storage& storage, const Shape& shape)
    : storage_{storage} {
  data_ = std::make_shared<std::vector<float>>(shape.numel());
}

Storage::StorageImpl::StorageImpl(Storage& storage,
                                  const std::vector<float>& data,
                                  const Shape& shape)
    : storage_{storage}, data_{std::make_shared<std::vector<float>>(data)} {}

const std::shared_ptr<std::vector<float>> Storage::StorageImpl::data() {
  return data_;
}

Storage::Storage() : impl_{std::make_shared<StorageImpl>(*this)} {}
Storage::Storage(const Shape& shape)
    : impl_{std::make_shared<StorageImpl>(*this, shape)} {}

Storage::Storage(const std::vector<float>& data, const Shape& shape) {
  if (data.size() != shape.numel()) {
    throw std::invalid_argument(
        "Sizes of the `data` and `shape` are incompetable.");
  }
  impl_ = std::make_shared<StorageImpl>(*this, data, shape);
}

const std::shared_ptr<std::vector<float>> Storage::data() {
  return impl_->data();
}
