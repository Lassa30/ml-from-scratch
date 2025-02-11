#pragma once
#include <memory>

namespace mlfs {
namespace nn {

class Storage {
private:
  class StorageImpl {};

  std::unique_ptr<StorageImpl> impl;

public:
  Storage() = default;
  ~Storage() = default;

  int64_t size() const noexcept;
  int64_t memsize() const noexcept;

  std::shared_ptr<float> data__();
  const std::shared_ptr<float> data();
};

}  // namespace nn
}  // namespace mlfs
