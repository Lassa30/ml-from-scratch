#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <utils/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace mlfs {
namespace optim {

class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual std::pair<Matrix, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                                double bias) const = 0;
  virtual Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, double bias) const = 0;
};

class MSE : public LossFunction {
public:
  std::pair<Matrix, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                        double bias) const override {
    auto y_pred = X.matmul(weights.T()) + bias;
    // weights gradient
    Matrix dW = (X.T().matmul(y - y_pred)) * -2.0 / X.rows();
    // bias gradient
    double db = (y - y_pred).sum() * (-2.0) / X.rows();

    return {dW.T(), db};
  }

  Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, double bias) const override {
    return (y - (X.matmul(weights.T()) + bias)) * (y - (X.matmul(weights.T()) + bias)) / X.rows();
  }
};

class Optimizer {
public:
  virtual ~Optimizer() = default;
  Optimizer() = default;
  Optimizer(const Optimizer &) = delete;
  virtual Optimizer &operator=(const Optimizer &) = delete;

  virtual void update(const Matrix &y, const Matrix &X, const LossFunction &lossFunc) = 0;

  virtual void zeroInit(const std::size_t &dim) = 0;

  virtual const Matrix &getWeights() const { return weights_; }
  virtual double getBias() const { return bias_; }

protected:
  Matrix weights_;
  double bias_;

  bool isInit_;
};

class SGD : public Optimizer {
public:
  SGD() = default;

  SGD(const Matrix &weights, const double &bias, const double learningRate) {
    weights_ = weights;
    bias_ = bias;
    lr_ = learningRate;
    isInit_ = true;
  }
  SGD(double lr) : lr_{lr} {}

  void zeroInit(const std::size_t &dim) {
    if (isInit_) {
      return;
    }
    weights_ = Matrix(1, dim, std::vector<double>(dim, 0.0));
    bias_ = 0.0;

    isInit_ = true;
  }

  void update(const Matrix &y, const Matrix &X, const LossFunction &lossFunc) override {
    auto [weightsGrad, biasGrad] = lossFunc.computeGrad(y, X, weights_, bias_);
    weights_ -= weightsGrad * lr_;
    bias_ -= biasGrad * lr_;
  }

  double getLearningRate() { return lr_; }

private:
  double lr_ = 1e-3;
};

} // namespace optim
} // namespace mlfs
#endif
