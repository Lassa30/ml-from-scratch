#include <layers.hpp>
#include <model.cpp>

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace mlfs {
namespace nn {

class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual MatrixXd compute(const MatrixXd &y, const MatrixXd &y_pred) const = 0;
  virtual MatrixXd backward(const MatrixXd &y, const MatrixXd &y_pred) const = 0;
};

// class MSE : public LossFunction {
// public:
//   MSE() = default;
//   ~MSE() = default;

//   MatrixXd compute(const MatrixXd &y, const MatrixXd &y_pred) const override {
//     return (y - y_pred).rowwise() * (y - y_pred).rowwise() / y.rows();
//   };

//   MatrixXd gradient(const MatrixXd &y, const MatrixXd &y_pred) const override { return -2.0 * (y - y_pred).rowwise()
//   / y.rows(); }
// };

// class CrossEntropyLoss : public LossFunction {
// public:
//   CrossEntropyLoss() = default;
//   ~CrossEntropyLoss() = default;

//   std::pair<MatrixXd, double> gradient(const MatrixXd &y, const MatrixXd &y_pred) const;
//   MatrixXd compute(const MatrixXd &y, const MatrixXd &y_pred) const;
// }
} // namespace nn
} // namespace mlfs