#ifndef LINEAR_REGRESSION_09_11_2024
#define LINEAR_REGRESSION_09_11_2024

#include <utils/matrix.hpp>
#include <utils/optimizer.hpp>

#include <memory>

namespace mlfs {
// Linear Regression with no regularization and MSE as a loss function.

class LinearRegression {
  std::unique_ptr<optim::Optimizer> optimizer_;
  std::unique_ptr<optim::LossFunction> loss_;

public:
  LinearRegression();
  ~LinearRegression() = default;
  LinearRegression(std::unique_ptr<optim::Optimizer> &&optim, std::unique_ptr<optim::LossFunction> &&loss);

  void train(const Matrix &features, const Matrix &target, const std::size_t &batch, const int &epochs,
             int randomState);

  void train(const Matrix &features, const Matrix &target, const std::size_t &batch, const int &epochs);

  Matrix predict(const Matrix &features) const;

  double score(const Matrix &prediction, const Matrix &target) const;

  void printWeights() const;

  void setOptimizer(std::unique_ptr<optim::Optimizer> &&optim);
  void setLoss(std::unique_ptr<optim::LossFunction> &&loss);
};
} // namespace mlfs

#endif