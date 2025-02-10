#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlfs {

namespace nn {

template <class T>
class Tensor {
  public:
  private:
    std::vector<int> shape;

    std::vector<int> stride;
    class TensorImpl;
    std::unique_ptr<TensorImpl> impl;

    std::shared_ptr<T> data_;
};

}  // namespace nn
}  // namespace mlfs