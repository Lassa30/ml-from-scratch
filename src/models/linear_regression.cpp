#include <models/linear_regression.hpp>
#include <utils/utils.hpp>

#include <iostream>
#include <random>
#include <stdexcept>

namespace mlfs {
LinearRegression::LinearRegression()
    : optimizer_{std::make_unique<optim::SGD>(1e-5)}, loss_{std::make_unique<optim::MSE>(optim::Reg::L2)} {}

LinearRegression::LinearRegression(std::unique_ptr<optim::Optimizer> &&optim,
                                   std::unique_ptr<optim::LossFunction> &&loss) {
  optimizer_ = std::move(optim);
  loss_ = std::move(loss);
}

void LinearRegression::train(const Matrix &features, const Matrix &target, const std::size_t &batch, const int &epochs,
                             int randomState) {

  if (batch > features.rows()) {
    throw std::logic_error("In LinearRegression.train():\n\tThe batch size is greater, than "
                           "features size\n");
  }
  // random
  std::mt19937 gen(randomState);
  std::uniform_int_distribution<> intDis(0, features.rows() - 1);
  std::uniform_real_distribution<> realDis(-1, 1);

  // SGD
  optimizer_->zeroInit(features.cols());
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

void LinearRegression::train(const Matrix &features, const Matrix &target, const std::size_t &batch,
                             const int &epochs) {
  train(features, target, batch, epochs, 42);
}

Matrix LinearRegression::predict(const Matrix &features) const {
  return features.matmul((optimizer_->getWeights()).T()) + (optimizer_->getBias());
}

double LinearRegression::score(const Matrix &prediction, const Matrix &target) const {
  return (loss_->computeLoss(target, prediction, optimizer_->getWeights(), optimizer_->getBias())).sum();
}

void LinearRegression::printWeights() const {
  std::cout << "\nWEIGHTS:\n";
  optimizer_->getWeights().printMatrix();

  std::cout << "\nBIAS:\n";
  std::cout << optimizer_->getBias() << std::endl;
}

void LinearRegression::setOptimizer(std::unique_ptr<optim::Optimizer> &&optim) { optimizer_ = std::move(optim); }
void LinearRegression::setLoss(std::unique_ptr<optim::LossFunction> &&loss) { loss_ = std::move(loss); }
} // namespace mlfs