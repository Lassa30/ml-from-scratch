#include <models/logistic_regression.hpp>
#include <utils/MatrixXd.hpp>

namespace mlfs {

class LogisticRegression::Impl {
public:
  Impl();

  MatrixXd train(const MatrixXd &features, const MatrixXd &target);

private:
  double learning_rate_;
};

LogisticRegression::LogisticRegression() : pImpl_{std::make_unique<Impl>()} {};

} // namespace mlfs