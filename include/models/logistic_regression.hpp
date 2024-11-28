#ifndef LOGISTIC_REGRESSION_2024_10_26
#define LOGISTIC_REGRESSION_2024_10_26

#include <utils/MatrixXd.hpp>
#include <utils/optimizer.hpp>

namespace mlfs {
class LogisticRegression {
public:
  LogisticRegression();
  ~LogisticRegression();

  void train(const MatrixXd &features, const MatrixXd &target);
  MatrixXd predict_proba(const MatrixXd &features);

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};
} // namespace mlfs

#endif