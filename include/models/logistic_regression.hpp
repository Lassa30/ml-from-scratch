#ifndef LOGISTIC_REGRESSION_2024_10_26
#define LOGISTIC_REGRESSION_2024_10_26

#include <utils/matrix.hpp>
#include <utils/optimizer.hpp>

namespace mlfs {
class LogisticRegression {
public:
  LogisticRegression();
  LogisticRegression(const double &learningRate);
  ~LogisticRegression();

  void train(const Matrix &features, const Matrix &target);
  Matrix predict_proba(const Matrix &features);
  void setThreshold(const double threshold);
  double getThreshold();

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};
} // namespace mlfs

#endif