#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <algorithm>
#include <memory>
#include <utils/matrix.hpp>
#include <vector>

// optimizer::optimizer class
// optimizer(& weights_, & bias_, *loss)
// optimizer::sgd
namespace mlfs {

class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual std::pair<const Matrix &, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                                        double bias) const = 0;
  virtual Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, double bias) const = 0;
};

class MSE : protected LossFunction {
public:
  std::pair<const Matrix &, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                                double bias) const override {
    auto y_pred = X.matmul(weights.T()) + bias;
    // weights gradient
    Matrix dW = (X.T().matmul(y - y_pred)) * -2.0 / X.rows();
    // bias gradient
    double db = ((y - y_pred) * (-2.0) / X.rows()).sum();

    return {dW, db};
  }
  Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, double bias) const override {
    return (y - (X.matmul(weights.T()) + bias)) * (y - (X.matmul(weights.T()) + bias)) / X.rows();
  }
};

class Optimizer {
protected:
  Matrix weights_;

  double bias_;

  std::unique_ptr<LossFunction> lossFunc_;

public:
  virtual ~Optimizer() = default;

  virtual void setLossFunction(std::unique_ptr<LossFunction> lossFunction) = 0;

  virtual void update(const Matrix &y, const Matrix &X) = 0;

  virtual const Matrix &getWeights() const { return weights_; }

  virtual const double &getBias() const { return bias_; }
};

class SGD : protected Optimizer {
public:
  SGD(const Matrix &weights, const double &bias, std::unique_ptr<LossFunction> lossFunc, const double learningRate);

  void update(const Matrix &y, const Matrix &X) override {
    auto [weightsGrad, biasGrad] = std::move(lossFunc_->computeGrad(y, X, weights_, bias_));
    weights_ -= weightsGrad * lr_;
    bias_ -= biasGrad * lr_;
  }

private:
  double lr_;
};

} // namespace mlfs
#endif
