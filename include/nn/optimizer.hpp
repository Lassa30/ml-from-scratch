#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <algorithm>
#include <memory>
#include <nn/layers.hpp>
#include <nn/loss_function.hpp>
#include <nn/model.hpp>
#include <random>
#include <set>
#include <vector>

namespace mlfs {
namespace nn {

class Optimizer {
  public:
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = delete;

    virtual Optimizer& operator=() = delete;
    virtual Optimzier&& operator=() = delete;

    virtual void zeroGrad() = 0;
    virtual void step() = 0;

  protected:
    std::unique_ptr<Model> model_;
};

}  // namespace nn
}  // namespace mlfs
#endif