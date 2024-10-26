#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <utils/matrix.hpp>

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace mlfs {

inline Matrix sign(const Matrix &mat) {
  if (mat.size() == 0)
    throw std::logic_error("In Matrix sign(const Matrix& mat):\n\tEmpty matrix is given.\n");

  std::vector<double> res(mat.size());
  for (auto i = 0; i < mat.size(); i++) {
    res[i] = (mat.getData()[i] > EPSILON) ? 1.0 : ((mat.getData()[i] < -EPSILON) ? -1.0 : 0);
  }

  return Matrix(mat.rows(), mat.cols(), res);
}

inline Matrix abs(const Matrix &mat) {
  if (mat.size() == 0)
    throw std::logic_error("In Matrix abs(const Matrix &mat):\n\tEmpty matrix is given.\n");

  std::vector<double> res(mat.size());
  for (auto i = 0; i < mat.size(); i++) {
    res[i] = std::abs(mat.getData()[i]);
  }

  return Matrix(mat.rows(), mat.cols(), res);
}

inline Matrix sigmoid(const Matrix &x) { return 1.0 / ((x * -1).exp() + 1); }
inline Matrix sigmoidInv(const Matrix &x) { return (x * -1).exp() + 1; }

// calculate softmax for each row
// ex:
// [ 1, 2, 3     [ 0.1 0.3 0.6
//  1, 1, 1  ~    0.3 0.3 0.3
//  2, 2, 2 ]      0.3 0.3 0.3 ]

inline Matrix softmax(const Matrix &x) {
  auto exp_x = x.exp();
  auto x_copy = x;
  std::vector<double> row_exp_sum(x.rows());

  // step 1: compute the sum of exponents for each row
  for (int j = 0; j < x.rows(); j++) {
    for (int i = 0; i < x.cols(); i++) {
      row_exp_sum[j] += exp_x.get(j, i);
    }
  }

  // step 2: divide each exp by exp_sum
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.cols(); j++) {
      exp_x[i * x.cols() + j] /= row_exp_sum[i];
    }
  }

  return exp_x;
}

namespace optim {

enum Reg {
  No,
  L1, // L1 = Sum(abs(x_i))
  L2  // L2 = Sum(x_i^2)
};

class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual std::pair<Matrix, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                                double bias) const = 0;
  virtual Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, const double &bias) const = 0;
};




// CrossEntropyLoss
class CEL : public LossFunction {
public:
  CEL(std::set<int> labels) : labels_{labels} {}

  // return pair: dW, db

  // find derivatives.
  std::pair<Matrix, double> computeGrad(const Matrix &y, const Matrix &X,
                                        const Matrix &weights,
                                        double bias) const;

  Matrix computeLoss(const Matrix &y, const Matrix &X, const Matrix &weights, const double &bias) const {
    auto y_one_hot = Matrix(X.rows(), labels_.size());
    // X: [NxM], y: [Nx1], y_one_hot: [NxC], softmax: [X @ W.T()] = [[NxM]*[CxM].T()]=[NxC], softmax[NxC] -> [NxC] logits GOOD!
    for (auto i = 0; i < y.rows(); i++) {
      int label = y.get(i, 1);
      y_one_hot[i * labels_.size() + label] = 1.0;
    }

    return (y_one_hot * (softmax(X.matmul(weights.T()) + bias)).log());  //* (-1.0 / X.rows());
  }

private:
  std::set<int> labels_;
};

class MSE : public LossFunction {
public:
  MSE() = default;
  MSE(const Reg &reg, const double &lmbda = 1e-3) : reg_{reg}, lambda_{lmbda} {}

  std::pair<Matrix, double> computeGrad(const Matrix &y, const Matrix &X, const Matrix &weights,
                                        double bias) const override {
    auto y_pred = X.matmul(weights.T()) + bias;

    // weights gradient
    Matrix dW = (X.T().matmul(y - y_pred)) * -2.0 / X.rows();

    // bias gradient
    double db = (y - y_pred).sum() * (-2.0) / X.rows();

    if (reg_ == Reg::L1) {
      dW += sign(weights).T() * lambda_;
      db += (bias > EPSILON) ? lambda_ : (bias < -EPSILON) ? -lambda_ : 0;
    } else if (reg_ == Reg::L2) {
      dW += weights.T() * 2.0 * lambda_;
      db += bias * 2.0 * lambda_;
    }
    return {dW.T(), db};
  }

  Matrix computeLoss(const Matrix &y, const Matrix &prediction, const Matrix &weights,
                     const double &bias) const override {
    auto loss = (y - prediction) * (y - prediction) / y.rows();

    if (reg_ == Reg::No) {
      return loss;
    } else if (reg_ == Reg::L1) {
      loss += abs(weights.T()).sum() * lambda_ + std::abs(bias) * lambda_;
    } else if (reg_ == Reg::L2) {
      loss += (weights * weights).sum() * 2.0 * lambda_;
      loss += bias * bias * 2.0 * lambda_;
    }
    return loss;
  }

protected:
  Reg reg_ = Reg::L2;
  double lambda_ = 1e-3;
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
  virtual bool isInit() const { return isInit_; }

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

    std::cout << "\nWeights are init\n";
    weights_.printMatrix();

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

  Matrix weights_;
  double bias_;

  bool isInit_;
};

} // namespace optim
} // namespace mlfs
#endif
