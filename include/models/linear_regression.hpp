#ifndef LINEAR_REGRESSION_09_11_2024
#define LINEAR_REGRESSION_09_11_2024

#include <utils/matrix.hpp>
#include <utils/optimizer.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace mlfs {
// Linear Regression with no regularization and MSE as a loss function.

class LinearRegression {
  std::unique_ptr<optim::Optimizer> optimizer_ = std::move(std::make_unique<optim::SGD>());
  std::unique_ptr<optim::LossFunction> loss_ = std::move(std::make_unique<optim::MSE>());
  optim::Reg reg_ = optim::Reg::No;

public:
  LinearRegression() = default;
  ~LinearRegression() = default;

  LinearRegression(std::unique_ptr<optim::Optimizer> &&optim, std::unique_ptr<optim::LossFunction> &&loss,
                   const optim::Reg &reg = optim::Reg::No) {
    optimizer_ = std::move(optim);
    loss_ = std::move(loss);
    reg_ = reg;
  }

  void train(const Matrix &features, const Matrix &target, const std::size_t &batch, const int &epochs, double st = 0,
             double fin = 0, int randomState = 42) {

    if (batch > features.rows()) {
      throw std::logic_error("In LinearRegression.train():\n\tThe batch size is greater, than "
                             "features size\n");
    }
    // random
    std::mt19937 gen(randomState);
    std::uniform_int_distribution<> intDis(0, features.rows() - 1);
    std::uniform_real_distribution<> realDis(-1, 1);

    // preparation
    optimizer_->zeroInit(features.cols());

    // SGD
    std::vector<int> idx(features.rows());
    std::iota(idx.begin(), idx.end(), 0);
    auto batchIdx = std::vector<int>(batch);

    for (int e = 0; e < epochs; ++e) {
      std::shuffle(idx.begin(), idx.end(), gen);

      for (auto i = 0; i <= features.rows() - batch; i += batch) {
        std::copy(idx.begin() + i, idx.begin() + i + batch - 1, batchIdx.begin());
        Matrix featuresBatch = utils::getBatch(features, batchIdx, batch);
        Matrix targetBatch = utils::getBatch(target, batchIdx, batch);

        optimizer_->update(targetBatch, featuresBatch, *loss_);
      }
    }
  }

  Matrix predict(const Matrix &features) const {
    return features.matmul((optimizer_->getWeights()).T()) + (optimizer_->getBias());
  }

  double score(const Matrix &prediction, const Matrix &target) const {
    return (loss_->computeLoss(target, prediction, optimizer_->getWeights(), optimizer_->getBias())).sum();
  }

  void printWeights() const {
    std::cout << "\nWEIGHTS:\n";
    optimizer_->getWeights().printMatrix();

    std::cout << "\nBIAS:\n";
    std::cout << optimizer_->getBias() << std::endl;
  }

  void setOptimizer(std::unique_ptr<optim::Optimizer> &&optim) { optimizer_ = std::move(optim); }
  void setLoss(std::unique_ptr<optim::LossFunction> &&loss) { loss_ = std::move(loss); }
};
} // namespace mlfs

#endif