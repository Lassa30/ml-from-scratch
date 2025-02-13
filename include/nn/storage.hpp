#pragma once

#include <nn/utils.hpp>

#include <memory>
#include <vector>

namespace mlfs {
namespace nn {

class Storage {
private:
  class StorageImpl;
  std::shared_ptr<StorageImpl> impl_;

public:
  Storage();

  Storage(const Shape& shape);
  Storage(const std::vector<float>& data, const Shape& shape);

  ~Storage() = default;

  Storage(const Storage&) = default;
  Storage& operator=(const Storage&) = default;

  Storage(Storage&&) noexcept = delete;
  Storage& operator=(Storage&&) noexcept = delete;

  const std::shared_ptr<std::vector<float>> data();
};

class Storage::StorageImpl {
private:
  Storage& storage_;
  std::shared_ptr<std::vector<float>> data_;
  bool checkSize(const std::vector<float>& data,
                 const Shape& shape) const noexcept;

public:
  StorageImpl(const StorageImpl&) = default;
  StorageImpl& operator=(const StorageImpl&) = default;

  StorageImpl(StorageImpl&&) noexcept = default;
  StorageImpl& operator=(StorageImpl&&) noexcept = default;

  StorageImpl(Storage& storage);
  StorageImpl(Storage& storage, const Shape& shape);
  StorageImpl(Storage& storage, const std::vector<float>& data,
              const Shape& shape);

  ~StorageImpl() = default;
  const std::shared_ptr<std::vector<float>> data();
};

}  // namespace nn
}  // namespace mlfs
