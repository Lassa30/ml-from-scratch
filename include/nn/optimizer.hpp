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

  Optimizer& operator=() = delete;
  Optimzier&& operator=() = delete;

  Optimizer(const Optimizer&) = delete;
  Optimizer(Optimizer&&) = delete;

  virtual void zeroGrad() final;
  virtual void backward(const LossFunction& lossFn) final;

  virtual void step() = 0;
};

}  // namespace nn
}  // namespace mlfs
#endif