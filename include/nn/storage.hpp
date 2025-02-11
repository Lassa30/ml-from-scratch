#pragma once

#include <memory>
#include <vector>

namespace mlfs {
namespace nn {

class Storage {
private:
  class StorageImpl {};
  std::unique_ptr<StorageImpl> impl_;

public:
  Storage() = default;
  Storage(const std::vector<float>& data) {};
  ~Storage() = default;

  const std::shared_ptr<float> data() { return nullptr; };
};

}  // namespace nn
}  // namespace mlfs
