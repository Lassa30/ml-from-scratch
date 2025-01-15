#ifndef OPTIMIZER_HPP_2024_09_10
#define OPTIMIZER_HPP_2024_09_10

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <utils/MatrixXd.hpp>
#include <vector>

namespace mlfs {
namespace optim {

// class LossFunction {
//  public:
//   virtual ~LossFunction() = default;
//   virtual MatrixXd compute(const MatrixXd& y, const MatrixXd& y_pred) const = 0;
//   virtual MatrixXd gradient(const MatrixXd& y, const MatrixXd& y_pred) const = 0;
// };

// class MSE : public LossFunction {
//  public:
//   MSE() = default;
//   ~MSE() = default;

//   MatrixXd compute(const MatrixXd& y, const MatrixXd& y_pred) const override {
//     return (y - y_pred) * (y - y_pred) / y.rows();
//   };

//   MatrixXd gradient(const MatrixXd& y, const MatrixXd& y_pred) const override { return -2.0 * (y - y_pred) /
//   y.rows(); }
// };

// class CrossEntropyLoss : public LossFunction {
//  public:
//   CrossEntropyLoss() = default;
//   ~CrossEntropyLoss() = default;

//   std::pair<MatrixXd, double> gradient(const MatrixXd& y, const MatrixXd& y_pred) const;
//   MatrixXd compute(const MatrixXd& y, const MatrixXd& y_pred) const;
// }

// // ---------------------------------Optimizers-------------------------------------------

// class Optimizer {
//  public:
//   virtual ~Optimizer() = default;
//   Optimizer() = default;

//   virtual void update(const MatrixXd& y, const MatrixXd& X, const LossFunction& lossFunc) = 0;

//   virtual void zeroInit(const std::size_t& dim) = 0;

//   virtual const MatrixXd& getWeights() const { return weights_; }
//   virtual double getBias() const { return bias_; }
//   virtual bool isInit() const { return isInit_; }
// };

// class SGD : public Optimizer {
//  public:
//   SGD() = default;

//   SGD(double lr) : lr_{lr} {}

//   void zeroInit(const std::size_t& dim) {
//     weights_ = MatrixXd(1, dim, std::vector<double>(dim, 0.0));
//     bias_ = 0.0;

//     std::cout << "\nWeights are init\n";
//     weights_.printMatrixXd();
//   }

//   void update(const MatrixXd& y, const MatrixXd& X, const LossFunction& lossFunc) override {
//     auto [weightsGrad, biasGrad] = lossFunc.computeGrad(y, X, weights_, bias_);
//     weights_ -= weightsGrad * lr_;
//     bias_ -= biasGrad * lr_;
//   }

//   double getLearningRate() { return lr_; }

//  private:
//   double lr_ = 1e-3;
// };
}  // namespace optim
}  // namespace mlfs
#endif