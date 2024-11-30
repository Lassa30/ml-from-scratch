#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <layers.hpp>
#include <model.cpp>

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace mlfs {
namespace nn {

class Optimizer {
public:
  virtual ~Optimizer() = default;
  Optimizer() = default;

  using VectorLayerSharedPtr = std::vector<std::shared_ptr<Layer>>;

  virtual void zeroGrad(VectorLayerSharedPtr &layers) = 0;
  virtual void step(VectorLayerSharedPtr &layers) = 0;
};

// class SGD : public Optimizer {
// public:
//   SGD() : lr_{1e-3};

//   SGD(double lr) : lr_{lr} {}

//   void zeroGrad(VectorLayerSharedPtr &layers) {}

//   void step(const MatrixXd &y, const MatrixXd &X, const LossFunction &lossFunc) override {
//     auto [weightsGrad, biasGrad] = lossFunc.computeGrad(y, X, weights_, bias_);
//     weights_ -= weightsGrad * lr_;
//     bias_ -= biasGrad * lr_;
//   }

//   double getLearningRate() { return lr_; }

// private:
//   double lr_;
// };
} // namespace nn
} // namespace mlfs
#endif