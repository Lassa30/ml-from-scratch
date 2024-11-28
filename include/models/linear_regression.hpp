#ifndef LINEAR_REGRESSION_09_11_2024
#define LINEAR_REGRESSION_09_11_2024

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utils/MatrixXd.hpp>
#include <utils/optimizer.hpp>

namespace mlfs {

class LinearRegression {
public:
  ~LinearRegression() = default;
  LinearRegression();

  // dL_dP is LossFunction's gradient w.r.t. prediction of the model. Just dL/d(Prediction)
  // std::pair<MatrixXd, double> backward(const MatrixXd &prediction, const MatrixXd &dL_dP) const;

  MatrixXd predict_proba(const MatrixXd &features) const;
  MatrixXd predict(const MatrixXd &features) const;
  // weights and biases
  // std::tuple parameters =
  //     std::make_tuple(std::unordered_map<std::string, MatrixXd>(), std::unordered_map<std::string, double>());

private:
  double threshold;
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};
} // namespace mlfs

#endif