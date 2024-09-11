#ifndef LINEAR_REGRESSION_09_11_2024
#define LINEAR_REGRESSION_09_11_2024

#include <utils/matrix.hpp>
#include <utils/optimizer.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace mlfs {
// Linear Regression with no regularization and MSE as a loss function.

class LinearRegression {
public:
  LinearRegression() = default;
  ~LinearRegression() = default;

  LinearRegression(std::unique_ptr<optim::Optimizer> &&optim, std::unique_ptr<optim::LossFunction> &&loss) {
    optimizer_ = std::move(optim);
    loss_ = std::move(loss);
  }

  void train(const Matrix &features, const Matrix &target, const std::size_t &batch, const int &epochs, double st = 0,
             double fin = 0, int randomState = 42) {

    if (batch > features.rows()) {
      throw std::logic_error("In LinearRegression.train():\n\tThe batch size is greater, than "
                             "features size\n");
    }
    // random
    std::mt19937 gen(randomState);
    std::uniform_int_distribution<> intDis(0, batch - 1);
    std::uniform_real_distribution<> realDis(-1, 1);

    // preparation
    optimizer_->zeroInit(features.cols());

    // SGD
    for (int e = 0; e < epochs; ++e) {

      for (auto i = 0; i < features.rows(); i += batch) {
        std::vector<int> idx(batch);
        std::generate(idx.begin(), idx.end(), [&]() { return intDis(gen); });

        const Matrix &featuresBatch = getBatch(features, idx, batch);
        const Matrix &targetBatch = getBatch(target, idx, batch);

        optimizer_->update(targetBatch, featuresBatch, *loss_);
      }
    }
  }

  Matrix predict(const Matrix &features) {
    return features.matmul((optimizer_->getWeights()).T()) + (optimizer_->getBias());
  }

  double score(const Matrix &prediction, const Matrix &target) {
    return ((target - prediction) * (target - prediction) / target.rows()).sum();
  }

  void printWeights() const {
    std::cout << "\nWEIGHTS:\n";
    optimizer_->getWeights().printMatrix();

    std::cout << "\nBIAS:\n";
    std::cout << optimizer_->getBias() << std::endl;
  }

  void setOptimizer(std::unique_ptr<optim::Optimizer> &&optim) { optimizer_ = std::move(optim); }
  void setLoss(std::unique_ptr<optim::LossFunction> &&loss) { loss_ = std::move(loss); }

private:
  // weights optimization
  std::unique_ptr<optim::Optimizer> optimizer_ = std::move(std::make_unique<optim::SGD>());
  std::unique_ptr<optim::LossFunction> loss_ = std::move(std::make_unique<optim::MSE>());

  Matrix getBatch(const Matrix &mat, const std::vector<int> &idx, const std::size_t batch_) const {
    std::vector<double> resVect;
    for (auto rowInd : idx) {
      auto rowVect = mat.getRow(rowInd).getData();
      resVect.insert(resVect.end(), rowVect.begin(), rowVect.end());
    }

    if (resVect.size() == batch_ * mat.cols()) {
      return Matrix(batch_, mat.cols(), resVect);
    } else {
      throw std::runtime_error("getBatch():\n\tVect size don't match...\n");
    }
  }
};
} // namespace mlfs
#endif