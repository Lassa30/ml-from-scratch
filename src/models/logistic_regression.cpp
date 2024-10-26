#include <utils/matrix.hpp>
#include <models/logistic_regression.hpp>

namespace mlfs {

class LogisticRegression::Impl {
public:
  Impl() : learning_rate_{0.01} {}

  Matrix train(const Matrix &features, const Matrix &target);

private:
 double learning_rate_;
};

LogisticRegression::LogisticRegression() : pImpl_{std::make_unique<Impl>()} {};

} // namespace mlfs